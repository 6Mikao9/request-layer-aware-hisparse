#!/usr/bin/env bash
# Production geometry with deterministic hit/miss states and one GPU.
set -Eeuo pipefail
umask 077
out=$1
mkdir "$out"
cd "$out"
ncu --version > ncu.version
nsys --version > nsys.version
python3 -B -m runtime.benchmark_shared_lru_scan --allow-cuda-test \
  --arms stock isolated shared --batches 48 --layers 60 --miss-blocks 0 2 4 \
  --warmup 3 --repeats 10 --output unprofiled.json > unprofiled.log 2>&1

metrics=(
    gpu__time_duration.sum
    lts__t_sector_hit_rate.pct
    lts__t_sectors_lookup_miss.sum
    sm__warps_active.avg.pct_of_peak_sustained_active
    launch__registers_per_thread
    launch__shared_mem_per_block_dynamic
    launch__waves_per_multiprocessor
    launch__occupancy_limit_shared_mem
    launch__occupancy_limit_registers
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    dram__bytes_read.sum
    dram__bytes_write.sum
)
metric_csv=$(IFS=,; printf '%s' "${metrics[*]}")
for misses in 0 2 4; do
    for arm in stock isolated shared; do
        if test -e /work/YIELD; then
            exit 75
        fi
        case_dir="${arm}-miss${misses}"
        mkdir "$case_dir"
        if test "$arm" = stock; then
            kernel=load_cache_to_device_buffer_kernel
        else
            kernel=rlh_shared_lru::resolve_kernel
        fi
        ncu --clock-control none --cache-control all --replay-mode kernel --graph-profiling node \
          --kernel-name-base demangled --kernel-name "regex:.*${kernel}.*" \
          --launch-skip 1 --launch-count 1 --metrics "$metric_csv" \
          --export "$case_dir/profile" python3 -B -m runtime.benchmark_shared_lru_scan \
          --allow-cuda-test --arms "$arm" --batches 48 --layers 60 --miss-blocks "$misses" \
          --warmup 1 --repeats 2 --output "$case_dir/bench.json" > "$case_dir/ncu.log" 2>&1
        shopt -s nullglob
        reports=("$case_dir/"*.ncu-rep "$case_dir/"*.ncu-repz)
        test "${#reports[@]}" = 1
        ncu --import "${reports[0]}" --page raw --csv > "$case_dir/metrics.csv"
        sha256sum "${reports[0]}" "$case_dir/bench.json" "$case_dir/metrics.csv" > "$case_dir/SHA256SUMS"
    done
done
