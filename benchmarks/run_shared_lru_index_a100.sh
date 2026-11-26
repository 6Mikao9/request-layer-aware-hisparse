#!/usr/bin/env bash
# Issue #6 A100 recipe for the shared hot-pool resolver switches. One idle A100, the pinned SGLang
# image, the repository root as the working directory. Stages never share a process: JIT build
# receipts, correctness gates, clean CUDA Graph timing, then Nsight Compute and Nsight Systems.
#   bash benchmarks/run_shared_lru_index_a100.sh NEW_OUTPUT_DIR
set -Eeuo pipefail
umask 077
out=$1
mkdir "$out"
arms=runtime/fixtures/shared_lru_arms
py=(python3 -B)
{ nvcc --version; ncu --version; nsys --version; "${py[@]}" --version; } > "$out/toolchain.txt" 2>&1

# 1. JIT build and toolchain receipts (compile only).
for arm in shared_arena isolated_arena shared_arena_indexed isolated_arena_indexed; do
    "${py[@]}" -m runtime.shared_lru_device build-receipt --allow-cuda-build --config "$arms/$arm.json" \
        --output "$out/build-$arm.json"
done

# 2. Correctness: every switch set against KernelMirror, eager and Graph, small and production geometry.
"${py[@]}" -m runtime.cuda_shared_lru_correctness --allow-cuda-test --output "$out/gate.json"
"${py[@]}" -m runtime.cuda_shared_lru_production_correctness --allow-cuda-test --all-planes \
    --output "$out/production-gate.json"

# 3. Clean timing, no profiler: native, isolated, old shared and optimized shared at 0, 2 and 4 of 16
#    miss blocks (0%, 12.5%, 25%); each technique alone; then the ring rebuild call.
"${py[@]}" -m runtime.benchmark_shared_lru_scan --allow-cuda-test --arms stock isolated shared \
    --features none indexed hit_index incremental_lru compact_owner --batches 48 --layers 60 \
    --miss-blocks 0 2 4 --layouts contiguous fragmented --warmup 5 --repeats 50 --output "$out/timing.json"
"${py[@]}" -m runtime.benchmark_shared_lru_scan --allow-cuda-test --arms isolated shared \
    --features indexed incremental_lru --ring-states rebuild --batches 48 --layers 60 --miss-blocks 0 2 4 \
    --warmup 5 --repeats 50 --output "$out/timing-rebuild.json"

# 4. Nsight Compute: one graph-node launch per arm and miss count, kernel replay, cold caches.
metrics=(
    gpu__time_duration.sum dram__bytes_read.sum dram__bytes_write.sum lts__t_sectors_lookup_miss.sum
    lts__t_sector_hit_rate.pct sm__warps_active.avg.pct_of_peak_sustained_active
    sm__maximum_warps_per_active_cycle_pct launch__occupancy_limit_shared_mem launch__occupancy_limit_registers
    launch__registers_per_thread launch__shared_mem_per_block_dynamic launch__waves_per_multiprocessor
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
)
metric_csv=$(IFS=,; printf '%s' "${metrics[*]}")
cases=("stock none" "isolated none" "shared none" "shared indexed")
for misses in 0 2 4; do
    for item in "${cases[@]}"; do
        read -r arm features <<< "$item"
        case_dir="$out/ncu-${arm}-${features}-miss${misses}"
        mkdir "$case_dir"
        kernel=$([ "$arm" = stock ] && echo load_cache_to_device_buffer_kernel || echo rlh_shared_lru::resolve_kernel)
        ncu --clock-control none --cache-control all --replay-mode kernel --graph-profiling node \
            --kernel-name-base demangled --kernel-name "regex:.*${kernel}.*" --launch-skip 1 --launch-count 1 \
            --metrics "$metric_csv" --export "$case_dir/profile" \
            "${py[@]}" -m runtime.benchmark_shared_lru_scan --allow-cuda-test --arms "$arm" --features "$features" \
            --no-reference --batches 48 --layers 60 --miss-blocks "$misses" --warmup 1 --repeats 2 \
            --output "$case_dir/bench.json" > "$case_dir/ncu.log" 2>&1
        ncu --import "$case_dir/profile.ncu-rep" --page raw --csv > "$case_dir/metrics.csv"
    done
done

# 5. Nsight Systems: graph nodes, stream order and launch queue delay of every replay.
for misses in 0 2 4; do
    for item in "${cases[@]}"; do
        read -r arm features <<< "$item"
        case_dir="$out/nsys-${arm}-${features}-miss${misses}"
        mkdir "$case_dir"
        nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --output "$case_dir/trace" \
            "${py[@]}" -m runtime.benchmark_shared_lru_scan --allow-cuda-test --arms "$arm" --features "$features" \
            --no-reference --batches 48 --layers 60 --miss-blocks "$misses" --warmup 3 --repeats 20 \
            --output "$case_dir/bench.json" > "$case_dir/nsys.log" 2>&1
        nsys export --type sqlite --output "$case_dir/trace.sqlite" "$case_dir/trace.nsys-rep"
        "${py[@]}" -m benchmarks.nsys_graph_summary "$case_dir/trace.sqlite" --output "$case_dir/graph-summary.json"
    done
done
find "$out" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
