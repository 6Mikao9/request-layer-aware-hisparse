"""Real-call semantic oracle for active capacity changes, with no raw data output.

Historical long-context KV is independently gathered from pinned host rows;
the newest token and every short-context token use their GPU pre-state. The
two stores differ intentionally: latest host slots can be allocated but stale.
"""

from .shadow import MUTABLE, READ_ONLY, ShadowEngine, byte_equal, span_bytes

SCHEMA = "hisparse-selected-kv-oracle-v1"
CHECKS = ("selected_validity", "selected_k", "selected_v", "selected_routing",
          "active_slot_identity", "lru_permutation", "unchanged_tail",
          "short_unchanged", "inactive_rows", "lru_order", "active_tags")


def reference_lru(tags, order, selections):
    """Pinned native batch-LRU transition, independently computed before the op.

    Hits retain their old relative order. Misses in sparse-selection order
    replace the earliest evictable slots. Remaining stale slots, loaded slots,
    then hits form the new LRU-to-MRU order.
    """
    if sorted(order) != list(range(len(tags))) or len(set(selections)) != len(selections):
        raise RuntimeError("Invalid pre-state LRU permutation or duplicate selection")
    present = {token: slot for slot, token in enumerate(tags) if token >= 0}
    if len(present) != sum(token >= 0 for token in tags):
        raise RuntimeError("Duplicate logical tokens in the active pre-state")
    wanted = set(selections)
    hits = [slot for slot in order if tags[slot] in wanted]
    evictable = [slot for slot in order if tags[slot] not in wanted]
    misses = [token for token in selections if token not in present]
    if len(misses) > len(evictable):
        raise RuntimeError("Insufficient active slots for exact sparse selection")
    victims = evictable[:len(misses)]
    after_tags = list(tags)
    for token, slot in zip(misses, victims):
        after_tags[slot] = token
        present[token] = slot
    after_order = evictable[len(misses):] + victims + hits
    return after_tags, after_order, [present[token] for token in selections], victims


class SelectionOracle(ShadowEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report["schema"] = SCHEMA
        self.report["scope_note"] = (
            "Every actual overlay swap call, exact selected KV bytes against host historical rows or GPU pre-state. "
            "Newest token always uses GPU pre-state; logical sparse block order is unchanged. "
            "Checks active-slot identity, LRU/tail/readonly state and actual coordinator lifecycle. "
            "Not an inference performance measurement; no raw selections or tensor values are written.")
        self.save()

    def check(self, layer_id, kwargs, live, stock, *, final_layer):
        t = self.torch
        hmax, active = kwargs["hot_buffer_size"], kwargs["active_hot_buffer_size"]
        if kwargs.get("skip_io", False) or not 0 < active <= hmax:
            raise RuntimeError("Oracle requires real IO and a valid active capacity")
        if kwargs.get("num_real_reqs") is None or not 0 <= layer_id < 60:
            raise RuntimeError("Oracle requires actual row count and absolute layer")
        self.synchronize()
        before = {key: self.copy("before_" + key, kwargs[key]) for key in MUTABLE + READ_ONLY}
        nreal = int(before["num_real_reqs"].item())
        if not 0 <= nreal <= kwargs["top_k_device_locs"].size(0):
            raise RuntimeError("Invalid active row count")
        if kwargs["top_k_device_locs"].size(1) != kwargs["top_k_blocks"].size(1) * kwargs["sparse_block_size"]:
            raise RuntimeError("Oracle requires the exact flattened sparse-selection width")
        if active < kwargs["top_k_device_locs"].size(1):
            raise RuntimeError("Active capacity must cover the full selected width, including padding")
        rows = []
        if before["device_buffer_k"].size(0) != before["device_buffer_v"].size(0):
            raise RuntimeError("Expected matching K/V physical token domains")
        writable = t.zeros(before["device_buffer_k"].size(0), dtype=t.bool,
                           device=before["device_buffer_k"].device)
        claimed_physical_slots = set()
        # Gather references BEFORE committing live. Host rows are immutable in
        # this eager/no-overlap interval after the device synchronization.
        for out_row in range(nreal):
            rid = int(before["req_pool_indices"][out_row].item())
            seq = int(before["seq_lens"][out_row].item())
            if seq <= 0 or not 0 <= rid < before["device_buffer_tokens"].size(0):
                raise RuntimeError("Invalid real request row or sequence length")
            blocks = before["top_k_blocks"][out_row].long()
            offsets = t.arange(kwargs["sparse_block_size"], device=blocks.device, dtype=t.int64)
            logical = (blocks[:, None] * kwargs["sparse_block_size"] + offsets).flatten()
            valid = ((blocks[:, None] >= 0) & (logical.reshape(-1, offsets.numel()) < seq)).flatten()
            tokens = logical[valid]
            if tokens.unique().numel() != tokens.numel():
                raise RuntimeError("The pinned kernel requires unique valid sparse selections")
            locations = before["device_buffer_locs"][rid]
            if seq <= hmax:
                reference_locs = locations[tokens].long()
                host_rows = None
                older = t.zeros(tokens.shape, dtype=t.bool, device=tokens.device)
                state = None
            else:
                older = tokens != seq - 1
                reference_locs = locations[hmax].long().expand(tokens.shape)
                active_locs = locations[:active].long()
                if (active_locs.unique().numel() != active or int(active_locs.min()) < 0
                        or int(active_locs.max()) >= writable.numel()):
                    raise RuntimeError("Active cache has invalid or nonunique physical slots")
                latest_loc = int(locations[hmax].item())
                physical = set(active_locs.cpu().tolist())
                if latest_loc in physical or physical.intersection(claimed_physical_slots):
                    raise RuntimeError("Active request slots overlap each other or the reserved latest slot")
                if latest_loc in claimed_physical_slots:
                    raise RuntimeError("Reserved latest slot overlaps another request")
                claimed_physical_slots.update(physical | {latest_loc})
                state = reference_lru(before["device_buffer_tokens"][rid, :active].cpu().tolist(),
                                      before["lru_slots"][rid, :active].long().cpu().tolist(),
                                      tokens[older].cpu().tolist())
                if state[3]:
                    writable[active_locs[t.tensor(state[3], device=active_locs.device, dtype=t.int64)]] = True
                host_rows = before["host_cache_locs"][rid, tokens[older]].long().cpu()
                if host_rows.numel() and (int(host_rows.min()) < 0
                        or int(host_rows.max()) >= kwargs["host_cache_k"].size(0)
                        or int(host_rows.max()) >= kwargs["host_cache_v"].size(0)):
                    raise RuntimeError("Historical selection lacks an allocated host KV row")
            if reference_locs.numel() and (int(reference_locs.min()) < 0
                    or int(reference_locs.max()) >= before["device_buffer_k"].size(0)
                    or int(reference_locs.max()) >= before["device_buffer_v"].size(0)):
                raise RuntimeError("Invalid pre-state device reference location")
            expected = {}
            for component in ("k", "v"):
                device_key = "device_buffer_" + component
                expected[component] = before[device_key][reference_locs].clone()
                if seq > hmax and host_rows.numel():
                    payload = kwargs["host_cache_" + component].index_select(0, host_rows)
                    expected[component][older] = payload.to(expected[component].device)
            rows.append((out_row, rid, seq, tokens, valid, older, reference_locs, expected, state))

        self.report["scratch_span_bytes_max"] = max(self.report["scratch_span_bytes_max"],
                                                    sum(span_bytes(v) for v in self.scratch.values()))
        live(**kwargs)  # Exactly one commit; stock's state machine is not the semantic oracle.
        self.synchronize()
        self.report["observed_allocated_bytes_max"] = max(self.report["observed_allocated_bytes_max"],
                                                          int(self.allocated_bytes()))
        self.report["checked_calls"] += 1
        stats = self.layer_stats.setdefault(layer_id, dict(layer_id=layer_id, checked=0,
            seq_min=None, seq_max=None, long_rows=0, short_rows=0, token_replacement_calls=0,
            active_capacity=active, historical_selected=0, expected_misses=0))
        if stats["active_capacity"] != active:
            raise RuntimeError("Layer capacity changed during a frozen-policy diagnostic")
        stats["checked"] += 1
        stats["token_replacement_calls"] += int(not byte_equal(t, kwargs["device_buffer_tokens"],
                                                               before["device_buffer_tokens"]))
        passed = dict.fromkeys(CHECKS, True)
        request_rows = []
        for out_row, rid, seq, tokens, valid, older, reference_locs, expected, state in rows:
            request_rows.append(rid)
            stats["seq_min"] = seq if stats["seq_min"] is None else min(stats["seq_min"], seq)
            stats["seq_max"] = seq if stats["seq_max"] is None else max(stats["seq_max"], seq)
            stats["long_rows"] += int(seq > hmax)
            stats["short_rows"] += int(seq <= hmax)
            output = kwargs["top_k_device_locs"][out_row]
            got = output[valid].long()
            in_bounds = (not got.numel() or (int(got.min()) >= 0
                and int(got.max()) < kwargs["device_buffer_k"].size(0)
                and int(got.max()) < kwargs["device_buffer_v"].size(0)))
            passed["selected_validity"] &= bool(t.all(output[~valid] == -1)) and in_bounds
            if in_bounds:
                passed["selected_k"] &= byte_equal(t, kwargs["device_buffer_k"][got], expected["k"])
                passed["selected_v"] &= byte_equal(t, kwargs["device_buffer_v"][got], expected["v"])
            else:
                passed["selected_k"] = passed["selected_v"] = False
            if seq <= hmax:
                passed["selected_routing"] &= bool(t.equal(got, reference_locs))
                passed["short_unchanged"] &= (
                    byte_equal(t, kwargs["device_buffer_tokens"][rid], before["device_buffer_tokens"][rid])
                    and byte_equal(t, kwargs["lru_slots"][rid], before["lru_slots"][rid]))
            else:
                stats["historical_selected"] += int(older.sum().item())
                stats["expected_misses"] += len(state[3])
                locations = before["device_buffer_locs"][rid, :active].long()
                sorted_locs, permutation = locations.sort()
                if sorted_locs.unique().numel() != active:
                    raise RuntimeError("Active cache does not have unique physical slots")
                positions = t.searchsorted(sorted_locs, got[older]).clamp(max=active - 1)
                belongs = bool(t.all(sorted_locs[positions] == got[older]))
                passed["selected_routing"] &= belongs and bool(t.equal(got[~older], reference_locs[~older]))
                reference_slots = t.tensor(state[2], device=locations.device, dtype=t.int64)
                passed["selected_routing"] &= bool(t.equal(got[older], locations[reference_slots]))
                passed["active_slot_identity"] &= belongs and bool(t.equal(
                    kwargs["device_buffer_tokens"][rid, permutation[positions]].long(), tokens[older]))
                passed["active_tags"] &= bool(t.equal(kwargs["device_buffer_tokens"][rid, :active].long(),
                    t.tensor(state[0], device=locations.device, dtype=t.int64)))
                passed["lru_order"] &= bool(t.equal(kwargs["lru_slots"][rid, :active].long(),
                    t.tensor(state[1], device=locations.device, dtype=t.int64)))
                passed["unchanged_tail"] &= byte_equal(t, kwargs["device_buffer_tokens"][rid, active:],
                                                       before["device_buffer_tokens"][rid, active:])
            passed["lru_permutation"] &= bool(t.equal(kwargs["lru_slots"][rid, :active].sort().values.long(),
                                                       t.arange(active, device=output.device, dtype=t.int64)))
            passed["unchanged_tail"] &= byte_equal(t, kwargs["lru_slots"][rid, active:], before["lru_slots"][rid, active:])

        active_rows = t.zeros(kwargs["device_buffer_tokens"].size(0), dtype=t.bool,
                              device=kwargs["device_buffer_tokens"].device)
        if request_rows:
            active_rows[t.tensor(request_rows, dtype=t.int64, device=active_rows.device)] = True
        passed["inactive_rows"] &= bool(t.all(kwargs["top_k_device_locs"][nreal:] == -1))
        for key in ("device_buffer_tokens", "lru_slots"):
            passed["inactive_rows"] &= byte_equal(t, kwargs[key][~active_rows], before[key][~active_rows])
        for key in ("device_buffer_k", "device_buffer_v"):
            # Latest slots, short-request rows, inactive tails, other request
            # rows and unused storage must all remain unchanged by this op.
            passed["unchanged_tail"] &= byte_equal(t, kwargs[key][~writable], before[key][~writable])
        for key, value in passed.items():
            self.record_check(key, value, layer_id)
        for key in READ_ONLY:
            self.record_check("readonly_" + key, byte_equal(t, kwargs[key], before[key]), layer_id)
        if layer_id == final_layer or not all(passed.values()) or self.report["mismatches"]:
            self.save()
