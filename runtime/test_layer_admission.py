from dataclasses import replace
import unittest

import torch

from .layer_admission import (
    DemotionProposal, capture_admission, plan_admission, require_current_snapshot,
)
from .layer_eviction_policy import LayerEvictionPolicy, LayerKey
from .layer_hybrid_pool import PageBudgetExceeded
from . import test_layer_eviction_scheduler as fixtures


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SchedulerTests()

    def scenario(self):
        bridge = self.fixture.build(pages=6)
        resident = bridge.open_request(1, request_id="resident")
        incoming = bridge.open_request(2, request_id="incoming")
        self.fixture.prefill(bridge, resident, count=20)
        chunk = tuple(range(64, 76))
        bridge.req_to_token[incoming.row, :len(chunk)] = torch.tensor(chunk)
        proposals = tuple(DemotionProposal(LayerKey(resident.row, resident.generation, layer),
                                          8, 1, 0, 0, True) for layer in range(2))
        return bridge, resident, incoming, chunk, proposals

    def snapshot(self, scenario, *, rank=0):
        bridge, _, incoming, chunk, proposals = scenario
        return capture_admission(bridge, rank, [incoming], [chunk], mode="prefill",
                                 proposals=proposals)

    def decide(self, *snapshots):
        domains = sum(len(s.groups) for s in snapshots)
        policy = LayerEvictionPolicy(trigger_free_bytes=(0,) * domains,
                                     target_free_bytes=(1,) * domains, max_migrations=2)
        return plan_admission(policy, snapshots, now=10)

    def test_decode_needs_page_only_at_boundary(self):
        bridge = self.fixture.build()
        lease = bridge.open_request(1, request_id="decode")
        self.fixture.prefill(bridge, lease, count=3)
        bridge.req_to_token[1, 3] = 7
        demand = bridge.inspect_forward([lease], [(7,)], mode="decode")
        self.assertEqual(demand.needed_pages, (0, 0))
        ticket = bridge.begin_forward([lease], [(7,)], mode="decode")
        for layer in range(2):
            self.fixture.write(bridge, ticket, layer)
            bridge.swap_in_selected_blocks(ticket, torch.tensor([1]), torch.tensor([4]),
                                           torch.tensor([[0, -1]], dtype=torch.int32), layer, 4)
        bridge.end_forward(ticket)
        bridge.req_to_token[1, 4] = 8
        demand = bridge.inspect_forward([lease], [(8,)], mode="decode")
        self.assertEqual(demand.needed_pages, (1, 1))

    def test_no_partial_plan_when_one_layer_is_uncovered(self):
        bridge, old, incoming, chunk, proposals = self.scenario()
        decision = self.decide(self.snapshot((bridge, old, incoming, chunk, proposals[:1])))
        self.assertEqual(decision.status, "no_complete_plan")
        self.assertEqual(decision.actions, ())
        bridge.demote(1, old, 0, 8)
        demand = bridge.inspect_forward([incoming], [chunk], mode="prefill")
        self.assertGreaterEqual(demand.free_pages[0], demand.needed_pages[0])
        self.assertLess(demand.free_pages[1], demand.needed_pages[1])
        with self.assertRaises(PageBudgetExceeded):
            bridge.begin_forward([incoming], [chunk], mode="prefill")

    def test_projection_then_real_demotion_reservation_and_writer(self):
        scenario = self.scenario()
        bridge, old, incoming, chunk, proposals = scenario
        pages = bridge.adapter._kv._pages
        before = [set(pages.page_ids(old, f"l{layer}")) for layer in range(2)]
        snapshot = self.snapshot(scenario)
        decision = self.decide(snapshot)
        self.assertEqual(decision.status, "planned")
        self.assertEqual(set(decision.actions), {(proposal.key, 8) for proposal in proposals})
        self.assertEqual([set(pages.page_ids(old, f"l{i}")) for i in range(2)], before)
        with self.assertRaises(PageBudgetExceeded):
            bridge.begin_forward([incoming], [chunk], mode="prefill")
        require_current_snapshot(bridge, snapshot, [incoming], [chunk], mode="prefill",
                                 proposals=proposals)
        for epoch, (key, capacity) in enumerate(decision.actions, 1):
            self.assertTrue(bridge.demote(epoch, old, key.layer, capacity).committed)
        self.assertEqual(self.decide(self.snapshot(scenario)).status, "main_pages_ready")
        ticket = bridge.begin_forward([incoming], [chunk], mode="prefill")
        for layer in range(2):
            self.assertTrue(before[layer] & set(pages.page_ids(incoming, f"l{layer}")))
            self.fixture.write(bridge, ticket, layer)
            locs = bridge.get_prefill_mapping(ticket, layer)[list(chunk)]
            k, v = bridge.get_kv_buffer(layer)
            self.assertEqual(k.dtype, torch.bfloat16)
            torch.testing.assert_close(k[locs, 0, 0], torch.tensor(chunk).bfloat16() + layer)
            torch.testing.assert_close(v[locs, 0, 0], -k[locs, 0, 0])
        bridge.end_forward(ticket)

    def test_prefill_and_scheduler_ineligible_candidates_are_excluded(self):
        bridge, old, _, _, proposals = self.scenario()
        bridge.req_to_token[old.row, 20] = 24
        snapshot = capture_admission(bridge, 0, [old], [(24,)], mode="prefill",
                                     proposals=(proposals[0], replace(proposals[1], decode_eligible=False)))
        self.assertEqual(snapshot.candidates, ())
        self.assertEqual({(e.key, e.hot_capacity): e.reason for e in snapshot.excluded}, {
            (proposals[0].key, proposals[0].hot_capacity): "participates_in_pending_prefill",
            (proposals[1].key, proposals[1].hot_capacity): "scheduler_not_decode_eligible",
        })

    def test_snapshot_revalidation_includes_allocator_state(self):
        for changed in ("generation", "history", "pages", "host"):
            with self.subTest(changed=changed):
                scenario = self.scenario()
                bridge, old, incoming, chunk, proposals = scenario
                extra = bridge.open_request(3, request_id="extra")
                if changed == "history":
                    self.fixture.prefill(bridge, extra, start=100, count=1)
                snapshot = self.snapshot(scenario)
                if changed == "generation":
                    bridge.finish(extra)
                    bridge.open_request(3, request_id="extra")
                elif changed == "history":
                    self.fixture.prefill(bridge, extra, start=101, count=1)
                elif changed == "pages":
                    bridge.adapter._kv._pages.grow(extra, {"l0": 1})
                else:
                    bridge.adapter._host_budget -= 1
                with self.assertRaisesRegex(ValueError, "became stale"):
                    require_current_snapshot(bridge, snapshot, [incoming], [chunk],
                                             mode="prefill", proposals=proposals)

    def test_host_budget_is_cumulative_per_rank_and_uses_max_context(self):
        snapshot = self.snapshot(self.scenario())
        host_charge = 64 * 16
        self.assertEqual(tuple(c.host_reserve_bytes for c in snapshot.candidates),
                         (host_charge, host_charge))
        low_host = replace(snapshot, host_free_bytes=2 * host_charge - 1)
        self.assertEqual(self.decide(low_host).status, "no_complete_plan")
        self.assertEqual(self.decide(replace(snapshot, world_size=2), replace(low_host, rank=1, world_size=2)).status,
                         "no_complete_plan")
        self.assertEqual(self.decide(replace(low_host, host_free_bytes=2 * host_charge)).status,
                         "planned")

    def test_rank_and_layer_free_pages_are_independent_domains(self):
        snapshot = self.snapshot(self.scenario())
        rank0 = replace(snapshot, world_size=2, free_bytes=(256, 256))
        rank1 = replace(snapshot, rank=1, world_size=2, free_bytes=(0, 0), candidates=())
        decision = self.decide(rank0, rank1)
        self.assertEqual(decision.domains, ((0, "l0"), (0, "l1"), (1, "l0"), (1, "l1")))
        self.assertEqual(decision.status, "no_complete_plan")
        self.assertEqual(decision.actions, ())

    def test_requires_snapshot_from_every_rank_and_checks_local_rank(self):
        scenario = self.scenario()
        snapshot = self.snapshot(scenario)
        with self.assertRaises(ValueError):
            self.snapshot(scenario, rank=1)
        with self.assertRaises(ValueError):
            self.decide(replace(snapshot, world_size=2))

    def test_pending_decode_demotion_removes_own_page_demand(self):
        bridge, resident, incoming, _, proposals = self.scenario()
        self.fixture.prefill(bridge, incoming, start=64, count=4)
        leases, chunks = [resident, incoming], [(24,), (68,)]
        bridge.req_to_token[resident.row, 20] = 24
        bridge.req_to_token[incoming.row, 4] = 68
        proposals = tuple(replace(p, hot_capacity=12) for p in proposals)
        snapshot = capture_admission(bridge, 0, leases, chunks, mode="decode",
                                     proposals=proposals)
        self.assertEqual(snapshot.needed_bytes, (128, 128))
        self.assertEqual(snapshot.free_bytes, (0, 0))
        self.assertEqual(snapshot.candidates[0].reclaim_bytes, (64, 0))
        self.assertEqual(snapshot.candidates[0].demand_relief_bytes, (64, 0))
        decision = self.decide(snapshot)
        self.assertEqual(decision.status, "planned")
        self.assertEqual(set(decision.actions), {(p.key, 12) for p in proposals})
        self.assertEqual(decision.plan.projected_free_bytes, (64, 64))
        self.assertEqual(decision.plan.required_free_bytes_after, (64, 64))
        for epoch, (key, capacity) in enumerate(decision.actions, 1):
            self.assertTrue(bridge.demote(epoch, resident, key.layer, capacity).committed)
        demand = bridge.inspect_forward(leases, chunks, mode="decode")
        self.assertEqual(demand.needed_pages, (1, 1))
        self.assertEqual(demand.free_pages, (1, 1))
        ticket = bridge.begin_forward(leases, chunks, mode="decode")
        bridge.abort_forward(ticket)
