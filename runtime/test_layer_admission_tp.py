import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.distributed as dist

from .layer_admission import DemotionProposal, ExcludedTarget, plan_admission, require_current_snapshot
from .layer_eviction_policy import LayerEvictionPolicy, LayerKey
from .layer_eviction_tp import DemotionProtocolError, GlooDemotionTransport
from .test_layer_forward_tp import build, write
from .test_layer_eviction_tp import Exchange, Transport


def prepare_native(bridge, lease, chunk):
    old = bridge.adapter._state(lease, 0).logical
    bridge.req_to_token[lease.row, :len(old)+len(chunk)] = torch.tensor(old+list(chunk))


def gloo_admission_worker(rank, address, output, multi_h=False):
    """Real two-process CPU/Gloo control and CPU BF16 pools; no CUDA claims."""
    dist.init_process_group("gloo", init_method=address, rank=rank, world_size=2,
                            timeout=timedelta(seconds=25))
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
        bridge, ctrl, old = build(GlooDemotionTransport(group), rid="old")
        initial = tuple(range(4, 24))
        prepare_native(bridge, old, initial)
        ticket = ctrl.begin(0, [old], [initial], mode="prefill")
        ctrl.run(ticket, lambda: [write(bridge, ticket, l, initial) for l in range(2)])
        epochs = [bridge.tp._last_epoch]
        pages = bridge.adapter._kv._pages
        old_pages = tuple(set(pages.page_ids(old, f"l{l}")) for l in range(2))
        new = bridge.open_request(2, request_id="incoming")
        chunk = tuple(range(64, 116))
        prepare_native(bridge, new, chunk)
        proposals = tuple(DemotionProposal(LayerKey(1, old.generation, layer),
                          8, 1, 0, 0, True) for layer in range(2))
        if multi_h:
            # H12 is cheaper on BOTH ranks but frees only one page; H8 frees
            # the two pages needed for this same incoming prepared forward.
            proposals = tuple(DemotionProposal(LayerKey(1, old.generation, layer),
                h, (8 + rank if h == 8 else 1 + rank), 0, 0, True)
                for layer in range(2) for h in (8, 12, 16))
        snapshots = ctrl.collect_admission(1, [new], [chunk], mode="prefill", proposals=proposals)
        epochs.append(bridge.tp._last_epoch)
        exclusions = ()
        if multi_h:
            page_bytes = bridge.adapter.layout.page_tokens * bridge.adapter.layout.kv_bytes_per_token_per_layer
            for snapshot in snapshots:
                assert snapshot.needed_bytes == (13 * page_bytes,) * 2
                assert snapshot.free_bytes == (11 * page_bytes,) * 2
                assert len(snapshot.candidates) == 4
                assert len(snapshot.excluded) == 2
                assert all(type(e) is ExcludedTarget and e.hot_capacity == 16 for e in snapshot.excluded)
                for candidate in snapshot.candidates:
                    p = candidate.proposal
                    assert p.hot_capacity in (8, 12)
                    assert sum(candidate.reclaim_bytes) == (2 if p.hot_capacity == 8 else 1) * page_bytes
                    assert p.switch_cost == (8 + snapshot.rank if p.hot_capacity == 8 else 1 + snapshot.rank)
            exclusions = tuple((s.rank, e.key.layer, e.hot_capacity, e.reason)
                               for s in snapshots for e in s.excluded)
        policy = LayerEvictionPolicy(trigger_free_bytes=(0,)*4,
            target_free_bytes=(1,)*4, max_migrations=2, cooldown=1)
        decision = plan_admission(policy, snapshots, now=10)
        assert decision.status == "planned"
        assert all(s.world_size == 2 for s in snapshots)
        if multi_h:
            assert decision.actions == tuple((LayerKey(1, old.generation, layer), 8) for layer in range(2))
            assert decision.plan.actions == decision.actions
        require_current_snapshot(bridge, snapshots[rank], [new], [chunk],
                                 mode="prefill", proposals=proposals)
        for epoch, (key, capacity) in enumerate(decision.actions, 2):
            assert bridge.demote(epoch, old, key.layer, capacity).committed
            epochs.append(bridge.tp._last_epoch)
        current = ctrl.collect_admission(4, [new], [chunk], mode="prefill", proposals=proposals)
        epochs.append(bridge.tp._last_epoch)
        if multi_h:
            assert all(not s.candidates and len(s.excluded) == 6 for s in current)
            assert all(type(e) is ExcludedTarget for s in current for e in s.excluded)
            assert all(bridge.adapter._state(old, layer).hot.capacity == 8 for layer in range(2))
        policy.commit(decision.plan, succeeded=decision.plan.keys,
                      free_bytes=tuple(v for s in current for v in s.free_bytes), now=11)
        ready = plan_admission(policy, current, now=11)
        assert ready.status == "main_pages_ready"
        ticket = ctrl.begin(5, [new], [chunk], mode="prefill")
        ctrl.run(ticket, lambda: [write(bridge, ticket, l, chunk) for l in range(2)])
        epochs.append(bridge.tp._last_epoch)
        reused_pages = []
        for layer in range(2):
            reused = old_pages[layer] & set(pages.page_ids(new, f"l{layer}"))
            assert reused
            reused_pages.append(len(reused))
            k, v = bridge.get_kv_buffer(layer)
            assert k.device.type == v.device.type == "cpu"
            assert k.dtype == v.dtype == torch.bfloat16
            locs = bridge.adapter._kv.get_prefill_mapping(layer)[list(chunk)]
            torch.testing.assert_close(k[locs, 0, 0], torch.tensor(chunk).bfloat16()+layer)
            torch.testing.assert_close(v[locs, 0, 0], -k[locs, 0, 0])
        bridge.tp.require_decode_ready()
        result = (rank, ready.status, len(bridge.adapter._state(new, 0).logical))
        if multi_h:
            actions = tuple((key.row, key.generation, key.layer, h) for key, h in decision.actions)
            result += (dict(actions=actions, epochs=tuple(epochs), exclusions=exclusions,
                            reused_pages=tuple(reused_pages)),)
        output.put(result)
    finally:
        dist.destroy_process_group()


class AdmissionTPTests(unittest.TestCase):
    def pair(self, callback):
        exchange = Exchange()
        states = [build(Transport(rank, exchange)) for rank in range(2)]
        def run(rank):
            try:
                return callback(rank, states[rank])
            except DemotionProtocolError as exc:
                return exc
        with ThreadPoolExecutor(2) as executor:
            results = list(executor.map(run, range(2)))
        return states, results

    def assert_closed(self, states, results):
        self.assertTrue(all(isinstance(r, DemotionProtocolError) for r in results))
        self.assertTrue(all(s[0].tp.poisoned for s in states))

    def test_one_rank_invalid_table_votes_before_exiting(self):
        def run(rank, state):
            bridge, ctrl, lease = state
            prepare_native(bridge, lease, (4, 5))
            if rank:
                bridge.req_to_token[lease.row, 0] = 99
            return ctrl.collect_admission(1, [lease], [(4, 5)], mode="prefill", proposals=())
        self.assert_closed(*self.pair(run))

    def test_admission_inspection_cannot_mix_with_forward_collective(self):
        def run(rank, state):
            bridge, ctrl, lease = state
            prepare_native(bridge, lease, (4, 5))
            if rank:
                return ctrl.begin(1, [lease], [(4, 5)], mode="prefill")
            return ctrl.collect_admission(1, [lease], [(4, 5)], mode="prefill", proposals=())
        self.assert_closed(*self.pair(run))

    def test_disagreeing_target_capacities_fail_before_reservation(self):
        def run(rank, state):
            bridge, ctrl, lease = state
            prepare_native(bridge, lease, (4, 5))
            proposals = (DemotionProposal(LayerKey(1, lease.generation, 0),
                                         8 if rank else 12, 1, 0, 0, True),)
            return ctrl.collect_admission(1, [lease], [(4, 5)], mode="prefill", proposals=proposals)
        self.assert_closed(*self.pair(run))

    def gloo_run(self, *, multi_h=False):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        with TemporaryDirectory() as tmp:
            address = Path(tmp, "gloo-init").resolve().as_uri()
            processes = [context.Process(target=gloo_admission_worker,
                         args=(rank, address, queue, multi_h)) for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                results = sorted(queue.get(timeout=40) for _ in processes)
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                return results
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
                queue.close()

    def test_gloo_reclaim_then_real_collective_reservation_and_bf16_writes(self):
        self.assertEqual(self.gloo_run(), [(0, "main_pages_ready", 52), (1, "main_pages_ready", 52)])

    def test_gloo_multih_common_small_h_then_reserve_bf16_and_page_reuse(self):
        results = self.gloo_run(multi_h=True)
        self.assertEqual([r[:3] for r in results], [(0, "main_pages_ready", 52), (1, "main_pages_ready", 52)])
        first, second = (r[3] for r in results)
        self.assertEqual(first, second)
        self.assertEqual([(a[2], a[3]) for a in first["actions"]], [(0, 8), (1, 8)])
        self.assertEqual(first["epochs"], (0, 1, 2, 3, 4, 5))
        self.assertEqual([(r, layer, h) for r, layer, h, _ in first["exclusions"]],
                         [(0, 0, 16), (0, 1, 16), (1, 0, 16), (1, 1, 16)])
        self.assertTrue(all(count > 0 for count in first["reused_pages"]))
