"""Produce a new independent source checkout, never edit the frozen input.

No package installation, compilation or GPU execution. Indexer port is separate.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REGISTRY = "vllm/v1/attention/backends/registry.py"
INSERT = ('    HYBRID_TRITON_SM80 = (\n'
          '        "vllm.v1.attention.backends.mla.hybrid_triton_sm80.HybridTritonSM80Backend"\n'
          '    )\n')
ANCHOR = '    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"\n'


def prepare(source, output):
    source = Path(source).absolute()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be new; no existing work is overwritten")
    if source == output or source in output.parents:
        raise ValueError("Output cannot be inside source")
    pins = json.loads((HERE / "attention-source-pins.json").read_text(encoding="utf-8"))
    for rel, expected in pins.items():
        path = source / rel
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source mismatch: {rel}")
    old = (source / REGISTRY).read_text(encoding="utf-8")
    if old.count(ANCHOR) != 1 or "HYBRID_TRITON_SM80" in old:
        raise ValueError("Registry anchor mismatch")
    additions = {
        "vllm/v1/attention/backends/mla/hybrid_triton_sm80.py": HERE / "hybrid_triton_sm80.py",
        "vllm/v1/attention/ops/hybrid_sm80_abi.py": HERE / "abi_probe.py",
        "vllm/v1/attention/ops/triton_mla_sparse_kernel.py": HERE / "third-party/patches/vllm-pr-38476/files/vllm/v1/attention/ops/triton_mla_sparse_kernel.py",
    }
    kernel_sha = hashlib.sha256(additions["vllm/v1/attention/ops/triton_mla_sparse_kernel.py"].read_bytes()).hexdigest()
    if kernel_sha != "c1a0186f43fd08ffcc397ad1731a8757979de4dc54d0b91ee99fe0afcc91a518":
        raise ValueError("Third-party kernel mismatch")
    for rel in additions:
        if (source / rel).exists(): raise ValueError(f"New file already exists: {rel}")
    # Explicit output is an isolated copy. Leave incomplete copies marked as
    # such on I/O failure; never create a success receipt before all files exist.
    def extended(path):
        return "\\\\?\\" + str(path) if os.name == "nt" else str(path)
    shutil.copytree(extended(source), extended(output), symlinks=True)
    new = old.replace(ANCHOR, INSERT + ANCHOR)
    (output / REGISTRY).write_text(new, encoding="utf-8", newline="\n")
    changes = {REGISTRY: (old, new)}
    for rel, path in additions.items():
        data = path.read_text(encoding="utf-8")
        (output / rel).parent.mkdir(parents=True, exist_ok=True)
        (output / rel).write_text(data, encoding="utf-8", newline="\n")
        changes[rel] = ("", data)
    patch = "".join("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="a/" + rel if before else "/dev/null", tofile="b/" + rel))
        for rel, (before, after) in changes.items())
    (output / "SM80_ATTENTION.patch").write_text(patch, encoding="utf-8", newline="\n")
    receipt = dict(schema="sm80-hybrid-attention-port-v1", complete_service=False,
                   gpu_tested=False, source_pins=pins,
                   changed={rel: hashlib.sha256((output / rel).read_bytes()).hexdigest()
                            for rel in changes},
                   pending=["SM80 indexer/model-kernel port", "full Hybrid extension build",
                            "GPU kernel/lifecycle gates", "actual INT4 model load and TP8 correctness"])
    (output / "SM80_ATTENTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, default=HERE / "hybrid")
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.source_root, args.output_root), indent=2))
