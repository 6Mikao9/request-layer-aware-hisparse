"""Explicit diagnostic sampling; absent configuration leaves serving unchanged.

Each candidate profile has its own frozen type/H and cost table. They are
inspected against the SAME prepared native batch, never executed by this probe.
Costs are not reused across H values. No request text or raw IDs are exported.
Synchronous all-rank collection and disk writes are diagnostic overhead, not
performance measurements. Runtime/source hashes are declared provenance, not
proof of a live process's full source identity.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat

from benchmarks.artifact_paths import is_private_path
from .layer_service_config import _hash, _int, _keys, parse_service_config


def _label(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("Diagnostic identifiers must be short opaque labels, not paths/request text")
    return value


def _output_path(value):
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError("Explicit absolute diagnostic output directory required")
    path = PureWindowsPath(value) if PureWindowsPath(value).is_absolute() else PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or is_private_path(path):
        raise ValueError("Diagnostic output must be an absolute non-private-area path without traversal")
    return value


@dataclass(frozen=True)
class CatalogDiagnostics:
    output_directory: str
    run_id: str
    workload_id: str
    runtime_sha256: str
    first_prepared: int
    every_prepared: int
    max_captures: int
    profiles: tuple
    digest: str


def parse_catalog_diagnostics(value, base_service):
    _keys(value, "schema output_directory run_id workload_id runtime_sha256 first_prepared "
          "every_prepared max_captures profiles", "catalog_diagnostics")
    if value["schema"] != "layer-catalog-probe-v1":
        raise ValueError("Unknown catalog diagnostic schema")
    output = _output_path(value["output_directory"])
    run, workload = _label(value["run_id"]), _label(value["workload_id"])
    runtime_sha = _hash(value["runtime_sha256"])
    first = _int(value["first_prepared"], "first zero-based prepared boundary", 0)
    every = _int(value["every_prepared"], "prepared interval")
    maximum = _int(value["max_captures"], "maximum captures")
    if maximum > 10000:
        raise ValueError("Diagnostic capture count is bounded to 10000 per run")
    profiles = value["profiles"]
    if type(profiles) is not list or not 1 <= len(profiles) <= 16:
        raise ValueError("Declare 1..16 explicit H/cost profiles")
    parsed, names = [], set()
    for profile in profiles:
        _keys(profile, "name type_policy cost_model", "diagnostic profile")
        name = _label(profile["name"])
        if name in names:
            raise ValueError("Duplicate diagnostic profile name")
        names.add(name)
        # Recursive parser sees NO diagnostic field. Geometry, request manifest
        # and host budget are unchanged; only explicit candidate policy/costs
        # differ. Pressure parsing validates cost coverage and H geometry even
        # when the outer run intentionally remains resident_only.
        candidate = dict(base_service, mode="pressure",
                         type_policy=profile["type_policy"], cost_model=profile["cost_model"])
        parsed.append((name, parse_service_config(candidate)))
    # The offline builder assigns one option to (request generation, layer, H).
    # Reject aliases at configuration time instead of silently choosing between
    # two potentially different cost estimates for the same action. None covers
    # unlisted/warmup requests using each profile's declared default table.
    dispatch_labels = (None, *sorted(set(base_service["request_types"].values())))
    for label in dispatch_labels:
        tables = [c.type_policy.resolve(label)[1] for _, c in parsed]
        for layer in range(parsed[0][1].layer_count):
            capacities = [table[layer] for table in tables]
            if len(set(capacities)) != len(capacities):
                raise ValueError("Diagnostic profiles repeat H for a dispatch type/layer")
    digest = sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode()).hexdigest()
    return CatalogDiagnostics(output, run, workload, runtime_sha, first, every,
                              maximum, tuple(parsed), digest)


def _existing_directory(path):
    # Inspect only the explicitly named ancestry, never enumerate directories.
    if not path.is_absolute() or is_private_path(path):
        raise ValueError("Output path must be absolute in the execution container")
    for part in reversed((path,) + tuple(path.parents)):
        info = part.lstat()
        if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("Diagnostic output ancestors must be real directories")


class ConfiguredCatalogProbe:
    def __init__(self, config, runtime, scheduler, service_sha256):
        self.config, self.runtime, self.scheduler = config, runtime, scheduler
        self.service_sha256 = _hash(service_sha256)
        self.providers = tuple(c.make_proposals(runtime, scheduler) for _, c in config.profiles)
        self.boundary_index = self.sample_count = 0
        self.pending_metadata = None
        self.last_receipt = None
        self.directory = Path(config.output_directory) / config.run_id
        if runtime.tp._transport.rank == 0:
            _existing_directory(self.directory.parent)
            # Every attempt needs a fresh run ID; never append to an old run.
            self.directory.mkdir(mode=0o700, exist_ok=False)

    def __call__(self, binding, handle, common_now):
        from analysis.capture_bundle import CaptureMetadata
        from .layer_scheduler_binding import CatalogProbeRequest
        if (binding.handoff is not self.runtime.handoff or binding.scheduler is not self.scheduler
                or type(common_now) not in (int, float) or not isfinite(common_now) or common_now < 0):
            raise ValueError("Diagnostic probe must use the owning scheduler and common clock")
        self.runtime.handoff._validate(handle, "prepared")
        index = self.boundary_index
        self.boundary_index += 1
        c = self.config
        if (self.sample_count >= c.max_captures or index < c.first_prepared
                or (index - c.first_prepared) % c.every_prepared):
            return None
        # Providers have their own observation clocks/counters and cannot mutate
        # the serving policy. resident_since is a conservative first observation,
        # NOT a reconstructed true residency start. No hidden model-time claim.
        rounds = tuple(provider("prepared", handle.requests, common_now) for provider in self.providers)
        sample = self.sample_count
        self.sample_count += 1
        deadline = binding.policy._next_plan_at
        if deadline == float("-inf"):
            deadline = None
        self.pending_metadata = CaptureMetadata(self.service_sha256, c.digest,
            tuple(name for name, _ in c.profiles), sample, index, common_now,
            deadline, binding.policy._min_residency, binding.policy._max_migrations)
        meta_sha = sha256(json.dumps(asdict(self.pending_metadata), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return CatalogProbeRequest(f"{c.run_id}-{sample:06d}", c.workload_id, c.runtime_sha256, rounds, meta_sha)

    def persist(self, captures):
        from analysis.capture_bundle import write_capture_bundle
        c, meta = self.config, self.pending_metadata
        if self.runtime.tp._transport.rank != 0 or meta is None:
            raise RuntimeError("Only rank zero can persist its selected pending sample")
        expected = f"{c.run_id}-{meta.sample_index:06d}"
        if (not captures or any(v.capture_id != expected or v.workload_id != c.workload_id
                               or v.runtime_sha256 != c.runtime_sha256 for v in captures)
                or any(len(v.rounds) != len(c.profiles) for v in captures)):
            raise ValueError("Collected capture differs from configured pending sample")
        receipt = write_capture_bundle(captures, self.directory / (expected + ".json"), metadata=meta)
        self.last_receipt = receipt
        self.pending_metadata = None
        return receipt


def build_catalog_callbacks(config, runtime, scheduler, service_sha256):
    probe = ConfiguredCatalogProbe(config, runtime, scheduler, service_sha256)
    return probe, probe.persist
