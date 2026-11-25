#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

output=$1
script=$2
mkdir "$output"
ncu --version > "$output/ncu.version"
metrics=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_lookup_miss.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
for batch in 8 32; do
    python3 -B "$script" --allow-cuda-test --batch "$batch" --output "$output/b${batch}-events.json"
    for arm in stock_hot resident_direct; do
        case_dir="$output/b${batch}-$arm"
        mkdir "$case_dir"
        ncu --clock-control none --cache-control all --replay-mode kernel --graph-profiling node \
            --nvtx --nvtx-include 'issue3-attention/' --kernel-name-base demangled \
            --kernel-name 'regex:.*_gqa_share_sparse_decode_kernel_impl.*' --launch-count 1 \
            --metrics "$metrics" --export "$case_dir/profile" \
            python3 -B "$script" --allow-cuda-test --batch "$batch" --probe "$arm" \
                --output "$case_dir/check.json" > "$case_dir/ncu.log" 2>&1
        shopt -s nullglob
        reports=("$case_dir/"*.ncu-rep "$case_dir/"*.ncu-repz)
        test "${#reports[@]}" -eq 1
        ncu --import "${reports[0]}" --page raw --csv > "$case_dir/metrics.csv"
        sha256sum "${reports[0]}" "$case_dir/check.json" "$case_dir/metrics.csv" > "$case_dir/SHA256SUMS"
    done
done
