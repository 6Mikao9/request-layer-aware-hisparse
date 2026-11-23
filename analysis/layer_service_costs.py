"""Hash-pinned calibration receipts -> frozen service costs; CPU and offline only.

No fitting, solver, interpolation, execution or raw conversation reader lives
here. A receipt contains already aggregated calibration values; its estimator
and coverage claim remain the producer's responsibility. SHA checks establish
identity, not the truth of a measurement or an end-to-end serving benefit.

Input files (relative paths confined beneath the manifest directory):
  manifest: schema, kind, unit, horizon_steps, protocol, scope, runtime_sources,
    context_definition, interval_convention, exposed_definition,
    switch_definition, receipts=[{path, sha256}].
  scope: model_identity, runtime_sha256, tp, layers, ranks, page_tokens,
    selected_tokens, dtype, execution_path, hardware.
  runtime_sources: nonempty [{path, sha256}]; runtime_sha256 is the SHA of
    encode_json(runtime_sources), in canonical path order. The caller supplies
    the independently pinned expected runtime fingerprint too.
  receipt: schema, contract_sha256, kind, estimator, rows.
    contract_sha256 hashes encode_json(manifest without receipts).
  row: request_type, rank, layer, H, context_low, context_high, switch_us,
    exposed_us, independent_origins, repetitions.

The receipt schema is layer-service-cost-receipt-v1. Measured receipts must
describe paired incremental exposed cache cost over the entire horizon after
resident->hot, plus the completed transition's switch cost. Summed kernel
durations, miss rates and old fused-overlay measurements are not that contract.
Surrogate receipts explicitly assert a model of the same target quantities.

Only exact type/rank/layer/H/range cells are selected. Missing cells reject the
whole range by default; explicit incomplete='exclude' records the omission.
No neighboring ranges are merged. Absent request types remain uncovered.
The output source_sha256 hashes encode_json(provenance); save these exact bytes
beside a launch configuration. A pressure config with zero ranges cannot launch.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath

from .mixed_type_budget import encode_json
from benchmarks.artifact_paths import is_private_path
from runtime.layer_service_config import (
    _hash, _int, _keys, _number, _object, _string, parse_service_config,
)


SCHEMA = "layer-service-cost-curves-v1"
RECEIPT_SCHEMA = "layer-service-cost-receipt-v1"
EXECUTION_PATH = "independent-layer-pages/native-split-store/eager/no-prefetch/resident-to-hot/v1"
EXPOSED_DEFINITION = "paired-incremental-exposed-us-over-entire-post-transition-horizon"
SWITCH_DEFINITION = "completed-resident-to-hot-transition-us"


def _digest(value):
    return sha256(encode_json(value)).hexdigest()


def _json(raw):
    def reject(_):
        raise ValueError("Non-finite JSON number")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=reject)


def _read_pin(path, digest):
    # This module only consumes named code/calibration artifacts, not captures
    # of prompts, generated output, token IDs, or arbitrary directory scans.
    if is_private_path(Path(path).absolute()):
        raise ValueError("Private paths are not calibration artifacts")
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != _hash(digest):
        raise ValueError("Artifact SHA mismatch")
    return raw


def _reference(root, ref):
    _keys(ref, "path sha256", "artifact reference")
    name = _string(ref["path"], "relative artifact path")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or "\\" in name or ":" in name
            or ".." in relative.parts or is_private_path(relative)
            or relative.as_posix() != name):
        raise ValueError("Artifact path must be canonical and confined to the bundle")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact symlink escapes the bundle")
    return _read_pin(path, ref["sha256"])


def _completed_interval(row, definition, convention, max_context):
    low = _int(row["context_low"], "context low", 0)
    high = _int(row["context_high"], "context high", 0)
    if convention == "(lo,hi]":
        low += 1
    if definition == "next_decode_seq_len=completed+1":
        low -= 1
        high -= 1
    if not 0 <= low <= high <= max_context:
        raise ValueError("Converted completed-history range outside context limit")
    return low, high


def export_frozen_costs(manifest_path, expected_sha256, *, service_config,
                        runtime_sha256, incomplete="error"):
    """Return (cost_model JSON, provenance JSON), without writing any files.

    service_config is a JSON config template accepted by parse_service_config;
    its old costs are replaced, never used to infer new costs. Runtime SHA is a
    separately trusted fingerprint of the source-file reference list. Hardware
    is currently the policy's pinned A100 target; this is not a topology model.
    """
    if incomplete not in ("error", "exclude"):
        raise ValueError("incomplete must be error or exclude")
    target = parse_service_config(service_config)
    path = Path(manifest_path).resolve()
    root = path.parent
    manifest = _json(_read_pin(path, expected_sha256))
    _keys(manifest, "schema kind unit horizon_steps protocol scope runtime_sources "
          "context_definition interval_convention exposed_definition switch_definition receipts", "curve manifest")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Expected rank-preserving cost curve manifest; miss summaries are not costs")
    if manifest["kind"] not in ("measured", "surrogate") or manifest["unit"] != "us":
        raise ValueError("Explicit measured/surrogate microsecond costs required")
    _int(manifest["horizon_steps"], "horizon_steps")
    _string(manifest["protocol"], "calibration protocol")
    if (manifest["exposed_definition"] != EXPOSED_DEFINITION
            or manifest["switch_definition"] != SWITCH_DEFINITION):
        raise ValueError("Costs must specify switch completion and whole-horizon exposed cost")
    if manifest["context_definition"] not in ("completed_before_forward", "next_decode_seq_len=completed+1"):
        raise ValueError("Ambiguous completed-history versus seq_len context")
    if manifest["interval_convention"] not in ("[lo,hi]", "(lo,hi]"):
        raise ValueError("Explicit integer interval convention required")
    scope = manifest["scope"]
    _keys(scope, "model_identity runtime_sha256 tp layers ranks page_tokens selected_tokens dtype execution_path hardware", "scope")
    expected = dict(model_identity=target.model_identity, runtime_sha256=_hash(runtime_sha256),
        tp=target.expected_tp_size, layers=list(range(60)), ranks=list(range(target.expected_tp_size)),
        page_tokens=128, selected_tokens=2048, dtype="bfloat16", execution_path=EXECUTION_PATH, hardware="A100")
    # Encoding equality also rejects booleans masquerading as integer indices.
    if encode_json(scope) != encode_json(expected):
        raise ValueError("Runtime/TopK/TP/all-layer/all-rank/execution scope mismatch")
    if target.layout.topk_blocks * target.layout.block_tokens != 2048:
        raise ValueError("Target service must keep TopK=2048")
    sources = manifest["runtime_sources"]
    if type(sources) is not list or not sources:
        raise ValueError("Runtime source file receipts required")
    for ref in sources:
        _reference(root, ref)
    names = [ref["path"] for ref in sources]
    if names != sorted(set(names)) or _digest(sources) != runtime_sha256:
        raise ValueError("Runtime source fingerprint mismatch or duplicate/noncanonical paths")
    contract = {k: v for k, v in manifest.items() if k != "receipts"}
    contract_sha = _digest(contract)
    refs = manifest["receipts"]
    if type(refs) is not list or not refs:
        raise ValueError("Calibration receipts required")
    tables = dict(target.type_policy.tables)
    cells, ranges, origins = {}, set(), []
    seen_paths = set(names)
    for ref in refs:
        raw = _reference(root, ref)
        if ref["path"] in seen_paths:
            raise ValueError("Repeated artifact path")
        seen_paths.add(ref["path"])
        receipt = _json(raw)
        _keys(receipt, "schema contract_sha256 kind estimator rows", "calibration receipt")
        if (receipt["schema"] != RECEIPT_SCHEMA or receipt["contract_sha256"] != contract_sha
                or receipt["kind"] != manifest["kind"]):
            raise ValueError("Receipt contract mismatch; cannot relabel evidence")
        _string(receipt["estimator"], "frozen estimator/model description")
        if type(receipt["rows"]) is not list or not receipt["rows"]:
            raise ValueError("Nonempty rank-preserving calibration rows required")
        origins.append(dict(path=ref["path"], sha256=ref["sha256"], estimator=receipt["estimator"]))
        for row in receipt["rows"]:
            _keys(row, "request_type rank layer H context_low context_high switch_us exposed_us independent_origins repetitions", "calibration row")
            label = row["request_type"]
            if type(label) is not str or label not in tables:
                raise ValueError("Unknown request type in calibration")
            rank = _int(row["rank"], "rank", 0)
            layer = _int(row["layer"], "layer", 0)
            hot = _int(row["H"], "H")
            if rank >= target.expected_tp_size or layer >= 60 or not 2048 <= hot <= target.type_policy.hmax or hot % 128:
                raise ValueError("Illegal rank/layer/capacity cell")
            low, high = _completed_interval(row, manifest["context_definition"],
                manifest["interval_convention"], target.max_context)
            for key in ("switch_us", "exposed_us"):
                _number(row[key], key)
            minimum = 1 if manifest["kind"] == "measured" else 0
            for key in ("independent_origins", "repetitions"):
                _int(row[key], key, minimum)
            key = label, low, high, rank, layer, hot
            if key in cells:
                raise ValueError("Duplicate calibration cell; aggregate upstream with a frozen estimator")
            cells[key] = row
            ranges.add((label, low, high))
    selected, excluded = [], []
    for label, low, high in sorted(ranges):
        missing = [(rank, layer, tables[label][layer])
                   for rank in range(target.expected_tp_size) for layer in range(60)
                   if (label, low, high, rank, layer, tables[label][layer]) not in cells]
        if missing:
            if incomplete == "error":
                raise ValueError(f"Incomplete frozen-H range {label} [{low},{high}]: {len(missing)} missing cells")
            excluded.append(dict(request_type=label, min_tokens=low, max_tokens=high,
                reason="missing_frozen_H_cells", missing_rank_layer_H=[list(k) for k in missing]))
            continue
        rank_costs = []
        for rank in range(target.expected_tp_size):
            rows = [cells[label, low, high, rank, layer, tables[label][layer]] for layer in range(60)]
            rank_costs.append({name: [r[name] for r in rows] for name in ("switch_us", "exposed_us")})
        selected.append(dict(request_type=label, min_tokens=low, max_tokens=high, ranks=rank_costs))
    for a, b in zip(selected, selected[1:]):
        if a["request_type"] == b["request_type"] and a["max_tokens"] >= b["min_tokens"]:
            raise ValueError("Selected completed-context ranges overlap")
    provenance = dict(schema="layer-service-cost-export-v1", manifest_sha256=expected_sha256,
        contract=contract, contract_sha256=contract_sha, receipts=origins,
        policy_sha256=target.type_policy.digest, frozen_H={k: list(v) for k,v in target.type_policy.tables},
        incomplete_action=incomplete, excluded_ranges=excluded,
        uncovered_request_types=sorted(set(tables) - {r["request_type"] for r in selected}),
        selected_ranges=selected, context_output="inclusive completed_before_forward",
        limitation="File identity and declared calibration contract checked; no measurement authenticity or serving benefit proved")
    costs = dict(kind=manifest["kind"], unit="us", horizon_steps=manifest["horizon_steps"],
        protocol=manifest["protocol"], source_sha256=_digest(provenance),
        policy_sha256=target.type_policy.digest, ranges=selected)
    if selected or target.mode == "resident_only":
        check = deepcopy(service_config)
        check["cost_model"] = costs
        parse_service_config(check)
    return costs, provenance
