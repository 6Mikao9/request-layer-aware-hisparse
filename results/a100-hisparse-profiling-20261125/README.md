# A100 HiSparse profiling · 2026-11-25

This bundle supports two separate optimization efforts. Issue #2 changes the scope of the hot buffer and its replacement policy. Issue #3's fixed-placement arm moves 20 layers to direct GPU-resident reads and uses the original HiSparse swap path on the other 40.

## Issue #2 · Cross-layer shared LRU

The matched CUDA Graph trace uses the same 48 real MiniMax inputs, frozen decode prefix, and equal KV hot-slot capacity on A100 TP8. Each arm contributes steps 17–24 across all 60 layers and eight ranks. The whole-step value is the earliest-to-latest kernel envelope across the eight ranks of each replay. Kernel columns are medians across layer, rank, and step cells.

| Path | Graph step | Swap kernel | Sparse attention | Merge |
| --- | ---: | ---: | ---: | ---: |
| Original HiSparse | 99.022 ms | 506.675 µs | 40.352 µs | 2.784 µs |
| Isolated arena | 100.821 ms | 554.342 µs | 40.448 µs | 2.752 µs |
| Shared arena | 135.490 ms | 1,327.226 µs | 46.080 µs | 2.752 µs |

[Nsight Systems export](issue2/nsys-three-arm.json.gz) · [Layer and step plots](issue2/nsys-three-arm.png)

One-GPU controlled probes use B48 and 60 layers, with matching initial state, selections, and logical KV payload. Across 0%, 12.5%, and 25% misses, sharing adds 721.92–724.48 µs to the swap kernel versus the isolated arena. At 0% misses, Nsight Compute reports 1.60 MB, 2.81 MB, and 190.80 MB of GPU DRAM reads for original, isolated, and shared. The zero-miss comparison isolates a large management cost. Logical KV payload and GPU DRAM counter bytes are separate quantities.

[Nsight Compute counters](issue2/ncu-controlled-miss.json.gz) · [Controlled-miss plots](issue2/ncu-controlled-miss.png)

Natural generation ran three rounds within each arm's independent worker. The original, isolated, and shared paths measured p50 TPOT of 99.96, 103.01, and 139.38 ms and throughput of 480.3, 464.5, and 343.7 token/s. The source was pinned at `3f9d0d8`, with MiniMax MXFP8 weights, BF16 KV, Top-K 2,048, 4,096 hot rows per layer, and page size 128. [Natural-generation results](issue2/natural-three-arm.json)

Optimization targets for the contributed code are the full-domain hit search and the stable LRU update. Give each change its own switch and preserve hit, victim, and payload semantics in the controlled probe. Compare original, isolated, and shared on the real same-batch Graph workload after checking kernel correctness. Include a zero-miss point to expose fixed management costs, and account for arena metadata in the equal-HBM service comparison. [Issue #2](https://github.com/6Mikao9/request-layer-aware-hisparse/issues/2)

## Issue #3 · Fixed layer placement

One complete natural-generation **B8** pair on A100 TP8 uses the same held-out 32–64 Ki-token cohort, BF16 KV, a common `mem_fraction_static=0.7828`, and the original [PR #5](https://github.com/6Mikao9/request-layer-aware-hisparse/pull/5) implementation at `f3c07ad`. Both arms have a verified full-batch CUDA Graph window after warmup.

| Metric | Original HiSparse | Fixed T40 | Change |
| --- | ---: | ---: | ---: |
| Steady-state decode throughput | 224.5859 token/s | 280.0024 token/s | +24.67% |
| TPOT p50 | 35.4317 ms | 28.4330 ms | −19.75% |
| TPOT p95 | 35.4406 ms | 28.4528 ms | −19.72% |
| TTFT p50 | 15.1671 s | 15.6155 s | +2.96% |
| Full-B8 steps | 155 | 133 | Both clear 128 |

The matching original A/A throughput differs by 2.59%. Full-output hash mismatch already reaches 8/8 on A/A, so report the performance pair with this quality-check limitation. Other frozen cohorts produced natural-EOS windows below the clean timing threshold. B32 at the original budget needs 858,112 logical rows and the fixed arm provides 764,672. The first result covers one B8 pair. The revised-budget B32 validation and full-model Nsight attribution are pending. [B8 timing and scope](issue3/b8-matched-progress.json)

The single-GPU attention reader probe matches the selected K/V and query while changing the source addressing. Graph event times for original hot-slot versus resident direct are 15.36 versus 16.38 µs at B8 and 37.38 versus 38.40 µs at B32. This probe runs the sparse attention reader with KV prepopulated on the device. Full-model swap savings require separate profiling of the two actual services. [Reader Nsight Compute counters](issue3/ncu-attention-reader.json)

For the next comparison, keep layer placement and resident K/V layout fixed when switching `resident_direct`. Compare calibrated, reversed, and random layer lists at the same resident layer count to isolate selection policy. The planned full-model trace checks all eight ranks for exactly 60 versus 40 native swap calls per step and follows the attention and collective dependencies before attributing changes in decode time. [Issue #3](https://github.com/6Mikao9/request-layer-aware-hisparse/issues/3)

## Reproduce the single-GPU profiles

Use the study's pinned SGLang A100 image, CUDA 13.4.59, SM80, and Nsight Compute 2026.3.0. Set `HISPARSE_REPRO_SOURCE` to an independent checkout at `3f9d0d8` for the Issue #2 benchmark module. Keep this bundle available from the current checkout. Run each command with a newly named output directory and one assigned GPU visible. [Environment and adapter](../../docs/implementation/agent-environment.md)

```bash
repo=$(git rev-parse --show-toplevel)
bundle="$repo/results/a100-hisparse-profiling-20261125"
export PYTHONPATH="${HISPARSE_REPRO_SOURCE:?Set the checkout at 3f9d0d8}${PYTHONPATH:+:$PYTHONPATH}"
bash "$bundle/reproduce/run_issue2_ncu_b48.sh" "$PWD/issue2-ncu-new"
bash "$bundle/reproduce/run_issue3_reader_ncu.sh" "$PWD/issue3-ncu-new" \
  "$bundle/reproduce/profile_issue3_attention.py"
```

`plot_matched_issue2.py` renders the shipped Nsight Systems export with a local font passed through `--font`. The artifacts include timing, profiler counters, and aggregate counts. The full-model workload, generated tokens, and raw process traces are kept in the experiment workspace. [Artifact checksums](SHA256SUMS)
