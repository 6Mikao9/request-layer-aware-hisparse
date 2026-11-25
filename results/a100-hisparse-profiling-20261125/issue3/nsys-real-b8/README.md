# Fixed-layer HiSparse on A100 · B8 Nsight Systems

One independent stock → T40 diagnostic pair used the same eight held-out MiniMax requests in the 32–64 Ki-token range. Both services used A100 TP8, BF16 KV and `mem_fraction_static=0.7828`. The original PR #5 arm at `f3c07ad` offloaded 40 layers through the native swap kernel and read the other 20 directly from resident GPU KV. The original HiSparse control swapped across all 60 layers. [Implementation](https://github.com/6Mikao9/request-layer-aware-hisparse/pull/5)

| GPU measure | Original HiSparse | Fixed T40 |
| --- | ---: | ---: |
| All-rank Graph envelope p50 | 33.437 ms | 26.761 ms |
| All-rank Graph envelope p95 | 35.909 ms | 29.274 ms |
| Verified consecutive Graph steps after warmup | 128 | 128 |
| Native swap layers per step and rank | 60 | 40 |
| Sum of per-layer median swap kernel times on the per-step bottleneck rank | 9.925 ms | 5.443 ms |
| Swap kernel times on the 20 layers made resident, same summary | 4.473 ms | 0.000 ms |

The GPU envelope p50 was **19.96% lower**, a difference of **6.675 ms**. The first-to-last CUDA kernel duration covers all eight ranks of each verified replay. The per-layer medians sum within each arm after selecting the longest-running rank for each replay. The swap sums are descriptive and do not partition the full Graph envelope. The plot also shows swap time without overlapping kernels on that rank and intervals from one attention-merge end to the next. The first interval starts at the first kernel of the replay.

The reducer proved the correct batch bucket, rank and device coverage, ordered request/sequence-length agreement within each arm, 60 attention/merge chains, and the exact configured swap mask. The two GPU windows were selected independently after warmup. The profiled client windows, 204 and 130 full-batch steps, separately establish that both workloads reached stable B8.

The clean natural-generation pair measured 35.432 → 28.433 ms TPOT p50 and 224.586 → 280.002 token/s. The GPU trace agrees with the direction and approximate magnitude of that single pair. The first experiment covers a single context range and memory budget. Multi-cohort and B32 capacity/performance conclusions require separate evidence.

[All-rank JSON aggregate](diagnostic-pair.json.gz) · [Layer comparison and Graph envelope](layer-profile.png) · [Plotting script](plot_issue3_nsys_pair.py)
