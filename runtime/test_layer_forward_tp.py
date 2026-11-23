import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from .layer_eviction import RequestLayerEvictionKV
from .layer_eviction_scheduler import LayerEvictionSchedulerBridge
from .layer_eviction_tp import DemotionProtocolError, GlooDemotionTransport, LayerEvictionTP
from .layer_forward_tp import LayerForwardTP
from .layer_hybrid_pool import LayerGroup, LayerLayout
from .test_layer_eviction_tp import Exchange, Transport
from .test_layer_hybrid_kv import CPUMHAPool


def build(transport, rid="request"):
    groups = tuple(LayerGroup(f"l{i}", (i,), "resident", 16) for i in range(2))
    layout = LayerLayout(2, 4, 4, 16, 2, 4, groups, 2 * 17 * 64)
    kv = RequestLayerEvictionKV(layout, 256, 4, "cpu", max_context=64,
                                host_budget_bytes=8192, pool_factory=CPUMHAPool)
    table = torch.full((4, 64), -1, dtype=torch.int32)
    tp = LayerEvictionTP(kv, transport, config_digest="model-v1")
    bridge = LayerEvictionSchedulerBridge(kv, table, tp_controller=tp)
    return bridge, LayerForwardTP(bridge), bridge.open_request(1, request_id=rid)


def begin(rank_state, epoch, chunk=(4, 5, 6, 7, 8), *, mode="prefill", wrong=False):
    bridge, ctrl, lease = rank_state
    old = bridge.adapter._state(lease, 0).logical
    bridge.req_to_token[lease.row, :len(old) + len(chunk)] = torch.tensor(old + list(chunk))
    if wrong:
        bridge.req_to_token[lease.row, len(old)] = 42
    return ctrl.begin(epoch, [lease], [chunk], mode=mode)


def write(bridge, ticket, layer, chunk=(4, 5, 6, 7, 8)):
    ids = torch.tensor(chunk)
    k = ids.reshape(-1, 1, 1).repeat(1, 1, 4).bfloat16() + layer
    bridge.set_kv_buffer(ticket, SimpleNamespace(layer_id=layer), ids, k, -k)


def run_forward(state, ticket):
    bridge, ctrl, _ = state
    ctrl.start(ticket)
    for layer in range(2):
        write(bridge, ticket, layer)
    ctrl.complete(ticket)


def gloo_forward_worker(rank, address, output):
    dist.init_process_group("gloo", init_method=address, rank=rank, world_size=2,
                            timeout=timedelta(seconds=25))
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
        state = build(GlooDemotionTransport(group))
        bridge, _, lease = state
        before = bridge.adapter.accounting()
        if rank == 1:
            with patch.object(bridge.adapter._kv._pages, "free_pages", return_value=0):
                rejected = begin(state, 1)
        else:
            rejected = begin(state, 1)
        rolled_back = before == bridge.adapter.accounting()
        ticket = begin(state, 2)
        run_forward(state, ticket)
        bridge.tp.require_decode_ready()
        output.put((rank, rejected is None, rolled_back, bridge.adapter._state(lease, 0).logical))
    finally:
        dist.destroy_process_group()


class ForwardTPTests(unittest.TestCase):
    def single(self):
        transport = SimpleNamespace(rank=0, world_size=1, all_gather=lambda value: [value])
        return build(transport)

    def test_ticket_allocation_oom_before_publication_leaves_no_pages(self):
        state = self.single()
        bridge, ctrl, _ = state
        before = bridge.adapter.accounting()
        with patch("runtime.layer_eviction_scheduler.ForwardTicket", side_effect=MemoryError("ticket OOM")):
            self.assertIsNone(begin(state, 1))
        self.assertEqual(bridge.adapter.accounting(), before)
        run_forward(state, begin(state, 2))

    def test_wrong_thread_cannot_inject_collective(self):
        state = self.single()
        bridge, ctrl, _ = state
        ticket = begin(state, 1)
        with patch.object(bridge.tp._transport, "all_gather", side_effect=AssertionError("foreign collective")) as gather:
            with ThreadPoolExecutor(1) as executor:
                with self.assertRaises(DemotionProtocolError):
                    executor.submit(ctrl.start, ticket).result(timeout=5)
            gather.assert_not_called()
        self.assertTrue(bridge.tp.poisoned)

    def test_arming_failure_and_supervisor_error_still_close_gate(self):
        state = self.single()
        bridge, ctrl, _ = state
        ticket = begin(state, 1)
        ctrl.on_fatal = lambda exc: (_ for _ in ()).throw(ValueError("supervisor callback failed"))
        with patch.object(bridge, "begin_model", side_effect=RuntimeError("arming failure")):
            with self.assertRaises(DemotionProtocolError):
                ctrl.start(ticket)
        self.assertTrue(bridge.tp.poisoned)

    def test_incremental_history_digest_is_independent_of_chunking(self):
        left, right = self.single(), self.single()
        run_forward(left, begin(left, 1))
        for epoch, chunk in enumerate(((4, 5), (6, 7, 8)), 1):
            ticket = begin(right, epoch, chunk)
            right[1].start(ticket)
            for layer in range(2):
                write(right[0], ticket, layer, chunk)
            right[1].complete(ticket)
        for layer in range(2):
            a = left[0].adapter._state(left[2], layer)
            b = right[0].adapter._state(right[2], layer)
            self.assertEqual(a.logical, b.logical)
            self.assertEqual(a.logical_digest.hexdigest(), b.logical_digest.hexdigest())

    def setup_ranks(self, rids=("request", "request"), transport_type=Transport):
        self.executors = [ThreadPoolExecutor(1) for _ in range(2)]
        for executor in self.executors:
            self.addCleanup(executor.shutdown)
        exchange = Exchange()
        self.ranks = [build(transport_type(rank, exchange), rid) for rank, rid in enumerate(rids)]

    def pair(self, fn):
        def run(rank):
            try:
                return fn(rank, self.ranks[rank])
            except DemotionProtocolError as exc:
                return exc

        futures = [executor.submit(run, rank) for rank, executor in enumerate(self.executors)]
        return [future.result(timeout=10) for future in futures]

    def assert_closed(self, results):
        for (bridge, _, _), result in zip(self.ranks, results):
            self.assertIsInstance(result, DemotionProtocolError)
            self.assertTrue(bridge.tp.poisoned)
            with self.assertRaises(DemotionProtocolError):
                bridge.tp.require_decode_ready()

    def test_oom_rolls_back_and_next_epoch_completes(self):
        self.setup_ranks()
        before = [bridge.adapter.accounting() for bridge, _, _ in self.ranks]
        pages = self.ranks[1][0].adapter._kv._pages
        with patch.object(pages, "free_pages", return_value=0):
            self.assertEqual(self.pair(lambda _, state: begin(state, 1)), [None, None])
        self.assertEqual([bridge.adapter.accounting() for bridge, _, _ in self.ranks], before)
        for bridge, _, lease in self.ranks:
            self.assertEqual(bridge.adapter._state(lease, 0).logical, [])
            bridge.tp.require_decode_ready()
        tickets = self.pair(lambda _, state: begin(state, 2))
        self.assertEqual(self.pair(lambda rank, state: run_forward(state, tickets[rank])), [None, None])
        for bridge, _, lease in self.ranks:
            for layer in range(2):
                self.assertEqual(bridge.adapter._state(lease, layer).logical, [4, 5, 6, 7, 8])

    def test_rid_mismatch(self):
        self.setup_ranks(rids=("one", "two"))
        self.assert_closed(self.pair(lambda _, state: begin(state, 1)))

    def test_wrong_table_is_voted_to_healthy_rank(self):
        self.setup_ranks()
        self.assert_closed(self.pair(lambda rank, state: begin(state, 1, wrong=rank == 1)))

    def test_writer_requires_start_and_cancel_releases_reservation(self):
        self.setup_ranks()
        before = [bridge.adapter.accounting() for bridge, _, _ in self.ranks]
        tickets = self.pair(lambda _, state: begin(state, 1))
        def reject_write(rank, state):
            with self.assertRaises(RuntimeError):
                write(state[0], tickets[rank], 0)

        self.pair(reject_write)
        self.assertEqual(self.pair(lambda rank, state: state[1].cancel(tickets[rank])), [None, None])
        self.assertEqual([bridge.adapter.accounting() for bridge, _, _ in self.ranks], before)
        self.assert_closed(self.pair(lambda _, state: begin(state, 1)))

    def test_completed_ticket_is_stale(self):
        self.setup_ranks()
        tickets = self.pair(lambda _, state: begin(state, 1))
        self.pair(lambda rank, state: run_forward(state, tickets[rank]))
        for (bridge, _, _), ticket in zip(self.ranks, tickets):
            with self.assertRaises((ValueError, DemotionProtocolError)):
                write(bridge, ticket, 0)

    def test_cancel_after_start_fails_closed(self):
        self.setup_ranks()
        tickets = self.pair(lambda _, state: begin(state, 1))
        self.pair(lambda rank, state: state[1].start(tickets[rank]))
        self.assert_closed(self.pair(lambda rank, state: state[1].cancel(tickets[rank])))

    def test_missing_layers_on_one_rank_fails_completion_everywhere(self):
        self.setup_ranks()
        tickets = self.pair(lambda _, state: begin(state, 1))
        self.pair(lambda rank, state: state[1].start(tickets[rank]))
        def write_layers(rank, state):
            write(state[0], tickets[rank], 0)
            if rank == 0:
                write(state[0], tickets[rank], 1)

        self.pair(write_layers)
        self.assert_closed(self.pair(lambda rank, state: state[1].complete(tickets[rank])))

    def test_forward_and_demote_share_epoch_and_proposal(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                self.setup_ranks()
                tickets = self.pair(lambda _, state: begin(state, 1))
                self.pair(lambda rank, state: run_forward(state, tickets[rank]))
                if mixed:
                    results = self.pair(lambda rank, state: (
                        begin(state, 2, (9,), mode="decode") if rank == 0
                        else state[0].demote(2, state[2], 0, 8)
                    ))
                else:
                    results = self.pair(lambda _, state: state[0].demote(1, state[2], 0, 8))
                self.assert_closed(results)

    def test_decode_requires_resolver_on_every_rank(self):
        for missing_resolver in (False, True):
            with self.subTest(missing_resolver=missing_resolver):
                self.setup_ranks()

                def execute(rank, state):
                    bridge, ctrl, _ = state
                    run_forward(state, begin(state, 1))
                    ticket = begin(state, 2, (9,), mode="decode")
                    ctrl.start(ticket)
                    for layer in range(2):
                        write(bridge, ticket, layer, (9,))
                        if missing_resolver and rank == 1 and layer == 1:
                            continue
                        locations = bridge.swap_in_selected_blocks(
                            ticket, torch.tensor([1]), torch.tensor([6]),
                            torch.tensor([[1, 0]], dtype=torch.int32), layer, 4)
                        k, _ = bridge.get_kv_buffer(layer)
                        self.assertEqual(k[locations[0, 1], 0, 0].item(), 9 + layer)
                    ctrl.complete(ticket)

                results = self.pair(execute)
                if missing_resolver:
                    self.assert_closed(results)
                else:
                    self.assertEqual(results, [None, None])

    def test_model_exception_notifies_supervisor_without_collective(self):
        self.setup_ranks()
        failures = []
        bridge, ctrl, _ = self.ranks[1]
        ctrl.on_fatal = failures.append
        tickets = self.pair(lambda _, state: begin(state, 1))
        error = RuntimeError("local model error")

        def model():
            raise error

        def execute(rank, state):
            _, ctrl, _ = state
            if rank == 0:
                ctrl.start(tickets[rank])
                return
            ctrl.run(tickets[rank], model)

        healthy, failed = self.pair(execute)
        self.assertIsNone(healthy)
        self.assertIsInstance(failed, DemotionProtocolError)
        self.assertIs(failed.__cause__, error)
        self.assertEqual(failures, [error])
        self.assertTrue(bridge.tp.poisoned)
        # No completion vote on a model exception; the supervisor stops peers.
        self.assertFalse(self.ranks[0][0].tp.poisoned)

    def test_collective_failure_fails_closed(self):
        class BadTransport(Transport):
            def all_gather(self, value):
                super().all_gather(value)
                raise TimeoutError("control group timeout")

        self.setup_ranks(transport_type=BadTransport)
        self.assert_closed(self.pair(lambda _, state: begin(state, 1)))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo unavailable")
    def test_gloo_oom_and_retry(self):
        ctx = multiprocessing.get_context("spawn")
        with TemporaryDirectory() as directory:
            address = (Path(directory) / "rendezvous").as_uri()
            output = ctx.Queue()
            processes = [ctx.Process(target=gloo_forward_worker, args=(i, address, output))
                         for i in range(2)]
            try:
                for proc in processes:
                    proc.start()
                results = sorted(output.get(timeout=45) for _ in processes)
                self.assertEqual(results, [(i, True, True, [4, 5, 6, 7, 8]) for i in range(2)])
                for proc in processes:
                    proc.join(timeout=10)
                    self.assertEqual(proc.exitcode, 0)
            finally:
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)
                output.close()
