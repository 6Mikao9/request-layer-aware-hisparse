"""Cost model and pressure controller of elastic per-layer HiSparse (issue #13). Pure Python, no torch.

Step time is modelled as in A100_HISPARSE_GRAPH_AND_SLO_20261124.md: ``T(S, B) = base(B) + sum_{l in S} t_l(B)``
where ``base`` is the step with every switchable layer resident and ``t_l`` the extra time of offloaded layer l:

    t_l(B, ctx) = exposed_l * (fixed_us + per_request_us * B + B * min(selected_rows, ctx) * miss_l(ctx)
                               * bytes_per_miss / h2d_bandwidth)

``miss_l`` comes from measured per-layer miss rates by context bin, the bandwidth and fixed cost from a fit of the
native swap kernel against misses (``profile_from_measurements``), and ``base`` is learned online from the steps
the server actually ran (weighted least squares on ``measured - offload_ms``), so the model only has to be right
about the difference between two offload sets. An SLO check multiplies by ``tail_factor`` and adds a margin.

``Controller.tick`` turns one observation of allocator and queue state into a target (level, logical tokens, hot
rows, admission cap). It grows capacity from free chunks first, offloads the next group only when that is needed
and the SLO model allows it (otherwise requests queue and admission is capped), and backfills a group only after
``dwell_steps`` of low demand. It is deterministic given its inputs; rank 0 runs it and broadcasts the decision.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path


def profile_from_measurements(path):
    """Profile fields from the repository's A100 aggregate (data/a100-hisparse-graph-and-c4-20261124.json):
    per-layer miss rates by context bin for MiniMax (and DeepSeek-V4 C4 layers), and the swap-kernel cost fit
    ``us = fixed + misses * per_miss`` over every MiniMax layer and wave."""
    data = json.loads(Path(path).read_text())
    rates, xs, ys = {}, [], []
    for wave in data["minimax"]["waves"]:
        upper = int(wave["input_length_bin_ki"].split("-")[1]) * 1024
        steps = len(wave["step_ordinals"])
        timing = {t["layer"]: t["fused_swap_kernel_us"] for t in wave["timing_by_layer"]}
        rates[str(upper)] = {}
        for row in wave["miss_by_layer"]:
            if row["lookups"]:
                rates[str(upper)][str(row["layer"])] = row["misses"] / row["lookups"]
                xs.append(row["misses"] / steps)
                ys.append(timing[row["layer"]])
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    c4 = {}
    for wave in data["deepseek"]["waves"]:
        upper = wave["context_tokens"][1]
        c4[str(upper)] = {str(r["main_layer"]): r["miss_rate"] for r in wave["layers"]}
    return dict(minimax=dict(miss_rate=rates, fixed_us=max(0.0, my - slope * mx),
                             h2d_gbps=512 / slope / 1e3, bytes_per_miss=512, selected_rows=2048),
                deepseek_v4=dict(miss_rate=c4, selected_rows=data["deepseek"]["top_k_c4_records"]))


MEASURED_SECTION = {"minimax_m3": "minimax", "deepseek_v4": "deepseek_v4"}


def load_profile(config, root):
    """The config's profile with ``profile.measurements`` (repo-relative) filling every field the config left at
    its default: per-layer miss rates, and for MiniMax the swap-kernel fit."""
    profile = dict(config.profile)
    if profile.get("measurements") is None:
        return profile
    section = MEASURED_SECTION.get(config.model)
    if section is None:
        raise ValueError(f"profile.measurements has no {config.model} section; give profile.miss_rate instead")
    measured = profile_from_measurements(Path(root) / profile["measurements"])[section]
    for key, value in measured.items():
        if key not in profile["explicit"]:
            profile[key] = value
    return profile


class CostModel:
    def __init__(self, profile):
        self.p = profile
        self._bins = [(int(k), v) for k, v in profile["miss_rate"].items()]

    def miss_rate(self, layer, ctx):
        for upper, layers in self._bins:
            if ctx <= upper:
                return layers.get(str(layer), self.p["default_miss_rate"])
        if self._bins:
            return self._bins[-1][1].get(str(layer), self.p["default_miss_rate"])
        return self.p["default_miss_rate"]

    def layer_ms(self, layer, batch, ctx):
        p = self.p
        if batch <= 0:
            return 0.0
        misses = batch * min(p["selected_rows"], ctx) * self.miss_rate(layer, ctx)
        us = p["fixed_us"] + p["per_request_us"] * batch + misses * p["bytes_per_miss"] / (p["h2d_gbps"] * 1e3)
        return p["exposed"].get(str(layer), 1.0) * us / 1e3

    def offload_ms(self, layers, batch, ctx):
        return sum(self.layer_ms(layer, batch, ctx) for layer in layers)


class BaseEstimator:
    """Online fit of ``base(B) = a + b * B`` with exponential forgetting; seeded by ``profile.base_ms``."""

    def __init__(self, prior=None, decay=0.98, slope=0.5):
        self.decay, self.prior_slope = decay, (prior[1] if prior else slope)
        self.prior = prior
        self.w = self.wx = self.wxx = self.wy = self.wxy = 0.0

    def observe(self, batch, base_ms):
        d = self.decay
        self.w, self.wx, self.wxx = d * self.w + 1, d * self.wx + batch, d * self.wxx + batch * batch
        self.wy, self.wxy = d * self.wy + base_ms, d * self.wxy + batch * base_ms

    def coefficients(self):
        if self.w == 0:
            return tuple(self.prior) if self.prior else None
        mx, my = self.wx / self.w, self.wy / self.w
        var = self.wxx / self.w - mx * mx
        slope = (self.wxy / self.w - mx * my) / var if var > 1e-6 * max(1.0, mx * mx) else self.prior_slope
        slope = max(0.0, slope)
        return my - slope * mx, slope

    def __call__(self, batch):
        coef = self.coefficients()
        return None if coef is None else coef[0] + coef[1] * batch


@dataclass(frozen=True)
class Observation:
    step: int
    level: int
    tokens: int                # logical capacity the allocator may hand out
    tokens_used: int
    tokens_peak: int           # one past the highest live logical location
    hot_rows: int
    hot_used: int
    hot_peak: int
    running: int
    mean_context: float
    waiting: int
    waiting_tokens: int        # prompt tokens of the queued requests
    retracted: int = 0
    step_ms: float | None = None
    busy: bool = False         # staging, chunked prefill or a transition in flight
    prefill_rows: int = 0      # hot rows the next prefill chunk takes (stock HiSparse prefills into hot rows)


@dataclass(frozen=True)
class Decision:
    level: int
    tokens: int
    hot_rows: int
    admission_cap: int | None
    reason: str

    def changes(self, obs):
        return (self.level, self.tokens, self.hot_rows) != (obs.level, obs.tokens, obs.hot_rows)


@dataclass(frozen=True)
class Geometry:
    """What the controller may assume about this rank: the chunk pool and how rows map to chunks."""
    layout: object             # elastic_memory.Layout
    chunks: int
    hot_per_request: int       # padded device buffer rows of one request (all offloaded layers share them)
    max_running: int
    hot_min: int = 0


class Controller:
    def __init__(self, config, geometry, cost=None):
        self.config, self.geometry = config, geometry
        self.cost = cost or CostModel(config.profile)
        self.base = BaseEstimator(config.profile["base_ms"])
        self.c = config.controller
        self.last_change = -10 ** 9
        self.low_since = None
        self.batch_cap = None
        self._caps = {}

    # ---- model -------------------------------------------------------------------------------------------
    def predict_ms(self, level, batch, ctx):
        base = self.base(batch)
        if base is None:
            return None
        return base + self.cost.offload_ms(self.config.offloaded(level), batch, ctx)

    def slo_ok(self, level, batch, ctx):
        slo = self.config.slo
        if slo is None:
            return True
        predicted = self.predict_ms(level, batch, ctx)
        return predicted is None or predicted * slo["tail_factor"] + slo["margin_ms"] <= slo["tpot_ms"]

    def admission_cap(self, level, ctx):
        """Largest running batch to admit: the SLO bound (with ``slo_admission``) and the batch of the operating
        point the throughput guard last chose; None when neither applies."""
        cap = self.batch_cap
        if self.config.feature("slo_admission"):
            slo_cap = 0
            for batch in range(1, self.geometry.max_running + 1):
                if not self.slo_ok(level, batch, ctx):
                    break
                slo_cap = batch
            cap = max(slo_cap, 1) if cap is None else min(cap, max(slo_cap, 1))
        return cap

    def observe(self, obs):
        """Feed the measured step of the current level into the base fit (offload cost removed)."""
        if obs.step_ms is not None and obs.running > 0:
            offload = self.cost.offload_ms(self.config.offloaded(obs.level), obs.running, obs.mean_context)
            self.base.observe(obs.running, max(0.0, obs.step_ms - offload))

    # ---- capacity ----------------------------------------------------------------------------------------
    def capacity(self, level, hot_rows):
        key = (level, hot_rows)
        if key not in self._caps:
            self._caps[key] = self.geometry.layout.max_tokens_for(
                self.geometry.chunks, hot_rows=hot_rows, offloaded=self.config.offloaded(level))
        return self._caps[key]

    def _align(self, value, unit):
        return -(-int(value) // unit) * unit

    def _hot_rows_for(self, obs, batch):
        """Hot rows a running batch needs (one padded device buffer per request, plus the next prefill chunk),
        with the high-watermark slack."""
        g = self.geometry
        need = max(obs.hot_peak, batch * g.hot_per_request + obs.prefill_rows)
        want = self._align(need / self.c["high_watermark"], g.layout.page)
        return min(g.layout.hot_rows, max(g.hot_min, want))

    def operating_point(self, level, obs, per_new):
        """(batch, hot rows, tokens) of ``level`` at the largest batch whose tokens and hot rows both fit the
        chunk pool (hot rows only cost chunks for offloaded layers). The running requests need their live rows
        plus ``horizon_steps`` of growth, each admitted one ``per_new`` tokens; batch 0 when even the running
        requests do not fit."""
        live = max(obs.tokens_peak, obs.tokens_used) + obs.running * self.c["horizon_steps"]

        def fit(batch):
            rows = max(obs.hot_rows if batch <= obs.running else 0, self._hot_rows_for(obs, batch))
            cap = self.capacity(level, rows)
            room = batch * self.geometry.hot_per_request <= self.geometry.layout.hot_rows
            return room and cap >= live + max(0, batch - obs.running) * per_new, rows, cap

        lo, hi = obs.running, self.geometry.max_running
        ok, rows, cap = fit(lo)
        if not ok:
            return 0, rows, cap
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fit(mid)[0]:
                lo = mid
            else:
                hi = mid - 1
        _, rows, cap = fit(lo)
        return lo, rows, cap

    # ---- decision ----------------------------------------------------------------------------------------
    def tick(self, obs):
        self.observe(obs)
        c, cfg, layout = self.c, self.config, self.geometry.layout
        ctx = max(1.0, obs.mean_context)
        room = max(0, self.geometry.max_running - obs.running)
        admit = min(obs.waiting, room)
        admit_tokens = obs.waiting_tokens if obs.waiting <= room else obs.waiting_tokens * room / obs.waiting
        need = obs.tokens_used + obs.running * c["horizon_steps"] + admit_tokens
        hot = self._hot_rows_for(obs, obs.running + admit)
        hold = Decision(obs.level, obs.tokens, obs.hot_rows, self.admission_cap(obs.level, ctx), "hold")
        if obs.busy:
            return replace(hold, reason="busy")
        pressured = obs.retracted > 0 or need > c["high_watermark"] * obs.tokens or hot > obs.hot_rows

        if pressured:
            self.low_since = None
            desired = obs.running + admit
            per_new = (admit_tokens / admit if admit else ctx) + c["horizon_steps"]
            points = []     # (level, batch, hot rows, tokens) at each level's largest feasible batch
            for level in range(obs.level, cfg.max_level + 1):
                if level > obs.level and not self.slo_ok(level, desired, ctx):
                    break           # one more group would break the SLO: stay and let requests queue
                batch, rows, cap = self.operating_point(level, obs, per_new)
                if batch >= max(1, obs.running):
                    points.append((level, batch, rows, cap))
                    if batch >= desired:
                        break
            if not points:
                return replace(hold, reason="pressure without a feasible level: requests queue")
            target = points[-1] if points[-1][1] >= desired else max(points, key=lambda p: (p[1], -p[0]))
            if cfg.feature("throughput_guard") and target[0] > obs.level:
                target = self._best_throughput(points, target, ctx)
            level, batch, rows, cap = target
            # Back only what the chosen batch needs (with the watermark slack): chunks left over keep the leaving
            # layers' rows readable (stale hits, cheaper backfill) until a later grow reclaims them.
            live = max(obs.tokens_peak, obs.tokens_used) + obs.running * c["horizon_steps"]
            demand = live + max(0, batch - obs.running) * per_new
            tokens = min(cap, max(obs.tokens, self._align(demand / c["high_watermark"], layout.page)))
            # Admitting past the chosen point would force the next offload once those requests grow.
            self.batch_cap = batch if cfg.feature("throughput_guard") else None
            if (level, tokens, rows) == (obs.level, obs.tokens, obs.hot_rows):
                return replace(hold, admission_cap=self.admission_cap(level, ctx),
                               reason="pressure at the best feasible capacity: requests queue")
            if level != obs.level:
                self.last_change = obs.step
            return Decision(level, tokens, rows, self.admission_cap(level, ctx),
                            "grow" if level == obs.level else f"offload to level {level}")

        if (cfg.feature("backfill") and obs.level > c["min_level"]
                and obs.step - self.last_change >= c["dwell_steps"]):
            prev = obs.level - 1
            hot = max(hot, obs.hot_rows)
            cap = self.capacity(prev, hot)
            if cfg.mode == "smooth":    # the paced copy backs both regions of the returning group until it ends
                cap = min(cap, layout.max_tokens_for(self.geometry.chunks, hot_rows=hot,
                                                     offloaded=cfg.offloaded(obs.level),
                                                     filling=cfg.offloaded(obs.level) - cfg.offloaded(prev)))
            compaction = cfg.feature("compaction")
            if cap > 0 and need <= c["low_watermark"] * cap and (obs.tokens_peak <= cap or compaction):
                if self.low_since is None:
                    self.low_since = obs.step
                if obs.step - self.low_since >= c["dwell_steps"]:
                    self.low_since, self.last_change, self.batch_cap = None, obs.step, None
                    floor = self._align(obs.tokens_used if compaction else obs.tokens_peak, layout.page)
                    tokens = min(cap, max(self._align(need / c["high_watermark"], layout.page), floor))
                    return Decision(prev, tokens, hot, self.admission_cap(prev, ctx), f"backfill to level {prev}")
                return replace(hold, reason="low demand, dwelling before backfill")
        self.low_since = None
        return hold

    def _best_throughput(self, points, target, ctx):
        """Among points up to ``target``, the one the model expects to decode the most tokens per second at its
        largest batch; a higher level must beat the best lower one by ``min_gain``. Offloading more only pays
        while the larger batch outweighs the extra swap-ins."""
        best, best_rate = None, -1.0
        for point in points:
            level, batch = point[0], point[1]
            if level > target[0]:
                break
            step = self.predict_ms(level, batch, ctx)
            if step is None:
                return target
            rate = batch / step
            if best is None or rate > best_rate * (1 + self.c["min_gain"]):
                best, best_rate = point, rate
        return best

    def initial(self, hot_rows=None):
        """Starting state: ``controller.initial_level`` with all the logical tokens it can back."""
        level = self.c["initial_level"]
        hot = self.geometry.layout.hot_rows if hot_rows is None else hot_rows
        cap = self.capacity(level, hot)
        if cap <= 0:
            raise ValueError(f"initial level {level} cannot back any logical token with {self.geometry.chunks} "
                             "chunks; raise the KV budget or start at a higher level")
        return Decision(level, cap, hot, None, "initial")


def backfill_rows_per_step(config, step_ms, row_bytes):
    """Rows one paced backfill step may copy so the H2D stream stays under ``controller.backfill_gbps``."""
    return max(1, int(config.controller["backfill_gbps"] * 1e6 * step_ms / max(1, row_bytes)))


def describe_levels(config, geometry, hot_rows):
    """Per-level capacity table for receipts and dry runs."""
    out = []
    for level in range(len(config.groups) + 1):
        offloaded = config.offloaded(level)
        out.append(dict(level=level, offloaded=sorted(offloaded),
                        tokens=geometry.layout.max_tokens_for(geometry.chunks, hot_rows=hot_rows,
                                                              offloaded=offloaded),
                        bytes_per_token=round(geometry.layout.bytes_per_token(offloaded), 3)))
    return out
