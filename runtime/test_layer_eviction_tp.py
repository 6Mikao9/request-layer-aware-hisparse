import multiprocessing
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from .layer_eviction import RequestLayerEvictionKV
from .layer_eviction_tp import DemotionProtocolError, GlooDemotionTransport, LayerEvictionTP
from .layer_hybrid_pool import LayerGroup, LayerLayout
from .test_layer_hybrid_kv import CPUMHAPool


def worker(transport, digest="model-v1"):
    layout = LayerLayout(1, 2, 4, 16, 2, 4, (LayerGroup("only", (0,), "resident", 12),), 832)
    kv = RequestLayerEvictionKV(layout, 64, 4, "cpu", max_context=32,
                                 host_budget_bytes=512, pool_factory=CPUMHAPool)
    lease = kv.open_request(1)
    values = torch.zeros((20, 1, 4), dtype=torch.bfloat16)
    kv.append(lease, 0, range(4, 24), values, values)
    return LayerEvictionTP(kv, transport, config_digest=digest), lease


class Exchange:
    def __init__(self, failed_phase=None):
        self.failed_phase = failed_phase
        self.records = [None, None]
        self.barrier = Barrier(2, timeout=5)


class Transport:
    world_size = 2

    def __init__(self, rank, exchange):
        self.rank = rank
        self.exchange = exchange

    def all_gather(self, value):
        self.exchange.records[self.rank] = value
        self.exchange.barrier.wait()
        result = tuple(self.exchange.records)
        self.exchange.barrier.wait()
        if value[0] == self.exchange.failed_phase:
            raise TimeoutError("control group timeout")
        return result


def gloo_worker(rank, address, output):
    dist.init_process_group("gloo", init_method=address, rank=rank,
                            world_size=2, timeout=timedelta(seconds=25))
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
        tp, lease = worker(GlooDemotionTransport(group))
        result = tp.demote(1, lease, 0, 8)
        tp.require_decode_ready()
        output.put((rank, result.reclaimed_bytes_by_rank, tp.kv.mode(lease, 0)))
    finally:
        dist.destroy_process_group()


class DemotionTPTests(unittest.TestCase):
    def setup_ranks(self, *, different_digest=False, failed_phase=None):
        exchange = Exchange(failed_phase)
        self.ranks = [
            worker(Transport(i, exchange), "different" if i and different_digest else "model-v1")
            for i in range(2)
        ]

    def run_epoch(self, epochs=(1, 1), capacities=(8, 8)):
        def run(rank):
            tp, lease = self.ranks[rank]
            try:
                return tp.demote(epochs[rank], lease, 0, capacities[rank])
            except DemotionProtocolError as exc:
                return exc

        with ThreadPoolExecutor(2) as pool:
            return list(pool.map(run, range(2)))

    def assert_closed(self, results):
        for (tp, _), result in zip(self.ranks, results):
            self.assertIsInstance(result, DemotionProtocolError)
            self.assertTrue(tp.poisoned)
            with self.assertRaises(DemotionProtocolError):
                tp.require_decode_ready()

    def test_commit_and_epoch_reuse(self):
        self.setup_ranks()
        for (tp, lease), result in zip(self.ranks, self.run_epoch()):
            self.assertTrue(result.committed)
            self.assertEqual(result.reclaimed_bytes_by_rank, (128, 128))
            self.assertEqual(tp.kv.mode(lease, 0), "hot")
            tp.require_decode_ready()
        self.assert_closed(self.run_epoch())

    def test_identity_and_configuration_mismatch(self):
        for epochs, capacities, different_digest in [
            ((1, 2), (8, 8), False), ((1, 1), (8, 12), False), ((1, 1), (8, 8), True),
        ]:
            with self.subTest(epochs=epochs, capacities=capacities, digest=different_digest):
                self.setup_ranks(different_digest=different_digest)
                self.assert_closed(self.run_epoch(epochs, capacities))
                for tp, _ in self.ranks:
                    self.assertEqual(tp.kv.accounting()["host_reserved_bytes"], 0)

    def test_inactive_request_votes_before_preparation(self):
        self.setup_ranks()
        tp, lease = self.ranks[1]
        tp.kv.finish(lease)
        self.assert_closed(self.run_epoch())

    def test_recycled_row_generation_mismatch(self):
        self.setup_ranks()
        tp, lease = self.ranks[1]
        tp.kv.finish(lease)
        self.ranks[1] = tp, tp.kv.open_request(lease.row)
        self.assert_closed(self.run_epoch())

    def test_prepare_rejection_cancels_both_and_consumes_epoch(self):
        self.setup_ranks()
        with patch.object(self.ranks[1][0].kv, "prepare_demote", side_effect=MemoryError):
            for (tp, lease), result in zip(self.ranks, self.run_epoch()):
                self.assertFalse(result.committed)
                self.assertEqual(result.reclaimed_bytes_by_rank, (0, 0))
                self.assertEqual(tp.kv.mode(lease, 0), "resident")
                self.assertEqual(tp.kv.accounting()["host_reserved_bytes"], 0)
                tp.require_decode_ready()
        self.assert_closed(self.run_epoch())

    def test_unsafe_prepare_and_commit_fail_closed(self):
        for method in ("prepare_demote", "commit_demote"):
            with self.subTest(method=method):
                self.setup_ranks()
                with patch.object(self.ranks[1][0].kv, method, side_effect=RuntimeError):
                    self.assert_closed(self.run_epoch())
                tp, lease = self.ranks[0]
                self.assertEqual(tp.kv.mode(lease, 0),
                                 "hot" if method == "commit_demote" else "resident")

    def test_cancel_failure(self):
        self.setup_ranks()
        with (
            patch.object(self.ranks[1][0].kv, "prepare_demote", side_effect=MemoryError),
            patch.object(self.ranks[0][0].kv, "cancel_demote", side_effect=RuntimeError),
        ):
            self.assert_closed(self.run_epoch())

    def test_collective_timeout(self):
        for phase in ("propose", "prepare", "commit"):
            with self.subTest(phase=phase):
                self.setup_ranks(failed_phase=phase)
                self.assert_closed(self.run_epoch())

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo unavailable")
    def test_gloo_process_group(self):
        ctx = multiprocessing.get_context("spawn")
        with TemporaryDirectory() as directory:
            address = (Path(directory) / "rendezvous").as_uri()
            output = ctx.Queue()
            processes = [ctx.Process(target=gloo_worker, args=(i, address, output)) for i in range(2)]
            try:
                for proc in processes:
                    proc.start()
                results = sorted(output.get(timeout=45) for _ in processes)
                self.assertEqual(results, [(0, (128, 128), "hot"), (1, (128, 128), "hot")])
                for proc in processes:
                    proc.join(timeout=10)
                    self.assertEqual(proc.exitcode, 0)
            finally:
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)
                output.close()
