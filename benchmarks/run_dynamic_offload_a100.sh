#!/usr/bin/env bash
# Issue #11 A100 recipe, stages in separate processes: route kernel build receipt and CUDA gate; output identity of
# every arm and switch sequence against stock (Graph and eager decode, B=1 and a padded B=3 bucket, a switch landing
# under load, no Graph capture after a switch); paired CUDA Graph serving of every arm on a fresh runtime (AB/BA
# order across repeats, two pressure points, one HBM budget, decode log and GPU memory sampled) with a timed runtime
# switch for the dynamic arms; all-rank Nsight Systems and Nsight Compute of the route kernel.
#   MODEL_DIR=... WORKLOAD=fixed-ids.jsonl TTFT_SLO=... TPOT_SLO=... [CORRECTNESS_WORKLOAD=short.jsonl] \
#   [MEM_FRACTION=0.84] bash benchmarks/run_dynamic_offload_a100.sh NEW_OUTPUT_DIR [minimax_m3|deepseek_v4]
# Then: python -m benchmarks.dynamic_offload_report all --dir NEW_OUTPUT_DIR --output NEW_OUTPUT_DIR/report.json
set -Eeuo pipefail
umask 077
out=$1
model=${2:-minimax_m3}
mkdir "$out" "$out/correctness"
: "${MODEL_DIR:?}" "${WORKLOAD:?}" "${TTFT_SLO:?}" "${TPOT_SLO:?}"
port=${PORT:-31080}
repeats=${REPEATS:-3}
read -r -a pressures <<< "${CONCURRENCY:-16 48}"
last=${pressures[${#pressures[@]}-1]}
arms_dir=runtime/fixtures/dynamic_offload_arms
py=(python3 -B)
if [ "$model" = deepseek_v4 ]; then
    : "${MEM_FRACTION:?set the common --mem-fraction-static of every DeepSeek-V4 arm}"
    prefix=dsv4 launcher=()
    serve=(--model-path "$MODEL_DIR" --tp-size 4 --enable-hisparse
           --hisparse-config '{"top_k":512,"device_buffer_size":1024}')
else
    MEM_FRACTION=${MEM_FRACTION:-0.84}            # the issue #3 B32 fixed-work budget
    prefix=minimax launcher=(--sm80-bf16-projections)
    serve=(--model-path "$MODEL_DIR" --tp-size 8 --enable-hisparse
           --hisparse-config '{"top_k":2048,"device_buffer_size":4096}' --page-size 128 --attention-backend triton)
fi
# Every arm: one HBM budget, CUDA Graph decode, eager prefill, one scheduler stream, and a decode log line every 8
# steps (the real running batch of each window).
serve+=(--port "$port" --mem-fraction-static "$MEM_FRACTION" --decode-log-interval 8 --disable-overlap-schedule
        --disable-radix-cache --disable-prefill-cuda-graph)
{ nvcc --version; nsys --version; ncu --version; "${py[@]}" --version; nvidia-smi; } > "$out/toolchain.txt" 2>&1

# 1. Route kernel: JIT build receipt and the eager/Graph correctness gate (history, production geometry, consumer).
"${py[@]}" -m runtime.dynamic_offload_device build-receipt --allow-cuda-build --model "$model" \
    --config "$arms_dir/${prefix}_dynamic.json" --output "$out/build.json"
"${py[@]}" -m runtime.cuda_dynamic_offload_correctness --allow-cuda-test --output "$out/gate.json"

# Arms: full_resident is the pinned server without HiSparse; stock is native HiSparse on every eligible layer;
# frozen_a is fixed S = list A (MiniMax: the calibrated T40 of the issue #3 measurements; DeepSeek-V4: the low-miss C4
# group), frozen_b fixed S = list B; indirect is the slab path on list A with switching off; dynamic* start on list B
# and switch to list A after warm-up, so their steady state pairs with frozen_a and indirect.
arms=(full_resident stock frozen_a frozen_b indirect dynamic dynamic_shadow)
command_for() {   # cmd = the server command of arm $1
    local skip=0 a
    case $1 in
        full_resident) cmd=("${py[@]}" -m sglang.launch_server)
                       for a in "${serve[@]}"; do
                           if [ "$skip" = 1 ]; then skip=0; continue; fi
                           case $a in --enable-hisparse) ;; --hisparse-config) skip=1 ;; *) cmd+=("$a") ;; esac
                       done ;;
        stock) cmd=("${py[@]}" -m runtime.launch ${launcher[@]+"${launcher[@]}"} -- "${serve[@]}") ;;
        *) cmd=("${py[@]}" -m runtime.launch --hisparse-study-config "$arms_dir/${prefix}_$1.json"
                ${launcher[@]+"${launcher[@]}"} -- "${serve[@]}") ;;
    esac
}
start() {   # $1 run label, runs "${cmd[@]}": server log, GPU memory samples (per-card peak), /server_info (capacity)
    "${cmd[@]}" > "$out/$1.server.log" 2>&1 &
    server=$!
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv -lms 500 \
        > "$out/$1.gpu-memory.csv" 2>&1 &
    sampler=$!
    for _ in $(seq 900); do
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null; then
            curl -sf "http://127.0.0.1:$port/server_info" -o "$out/$1.server-info.json" || true
            return 0
        fi
        kill -0 "$server" 2> /dev/null || return 1
        sleep 2
    done
    return 1
}
stop() {
    for pid in "${server:-}" "${sampler:-}"; do
        if [ -n "$pid" ]; then kill "$pid" 2> /dev/null || true; wait "$pid" 2> /dev/null || true; fi
    done
    server= sampler=
}
trap stop EXIT
client() {   # $1 output, $2 concurrency, [$3 workload]
    "${py[@]}" -m benchmarks.sglang_stream_client --url "http://127.0.0.1:$port/generate" --workload "${3:-$WORKLOAD}" \
        --concurrency "$2" --ttft-slo-s "$TTFT_SLO" --tpot-slo-s "$TPOT_SLO" --output "$1"
}
switches=0
switch_to() {   # $1 run label, $2 list index: response and client wall time of one TP-agreed switch
    switches=$((switches + 1))
    curl -s -X POST "http://127.0.0.1:$port/set_internal_state" -H 'Content-Type: application/json' \
        -d "{\"server_args\": {\"hisparse_offload_list\": $2}}" -o "$out/$1.switch$switches-response.json" \
        -w '{"http_seconds": %{time_total}}\n' > "$out/$1.switch$switches-time.json" || true
}
receipts() { grep '"event": "hisparse_dynamic_offload' "$out/$1.server.log" > "$out/$1.receipts.jsonl" || true; }

# 2. Output identity. Every arm reads the same K/V bytes, so with greedy decoding each pass must reproduce stock
#    token for token: strictly at B=1, and at concurrency 3 (a padded, non-full Graph bucket) against the run-to-run
#    noise of a second stock server. Dynamic servers (list 0 = B, list 1 = A) replay the workload with no switch,
#    after B->A, after A->B, after three consecutive switches, and once while a switch lands between decoding steps.
cw=${CORRECTNESS_WORKLOAD:-$WORKLOAD}
check() {   # $1 label
    "${py[@]}" -m benchmarks.sglang_correctness_client --url "http://127.0.0.1:$port/generate" --workload "$cw" \
        --output "$out/correctness/$1.b1.json" || true
    client "$out/correctness/$1.b3.json" 3 "$cw" || true
}
live_switch() {   # $1 run label, $2 list index, $3 check label: retried until it lands while requests decode
    "${py[@]}" -m benchmarks.sglang_correctness_client --url "http://127.0.0.1:$port/generate" --workload "$cw" \
        --output "$out/correctness/$3.b1.json" &
    local client_pid=$!
    sleep "${LIVE_SWITCH_DELAY:-5}"
    while kill -0 "$client_pid" 2> /dev/null; do
        switch_to "$1" "$2"
        if grep -q '"updated": *true' "$out/$1.switch$switches-response.json"; then break; fi
        sleep 1
    done
    wait "$client_pid" || true
}
for arm in stock stock_repeat frozen_a frozen_b; do
    command_for "${arm%_repeat}"
    start "correctness-$arm"
    check "$arm"
    stop
done
for arm in dynamic dynamic_eager dynamic_shadow; do
    run="correctness-$arm"
    command_for "${arm%_eager}"
    if [ "$arm" = dynamic_eager ]; then cmd+=(--disable-cuda-graph); fi
    start "$run"
    check "$arm-none"
    switch_to "$run" 1
    check "$arm-b-to-a"
    switch_to "$run" 0
    check "$arm-a-to-b"
    for target in 1 0 1; do switch_to "$run" "$target"; done
    check "$arm-consecutive"
    live_switch "$run" 0 "$arm-live"
    stop
    receipts "$run"
    awk '/"event": "hisparse_dynamic_offload_switch"/ {s = 1} s && /Capture .*begin/ {n++} END {print n + 0}' \
        "$out/$run.server.log" > "$out/$run.captures-after-switch.txt"
done

# 3. Serving arms, fresh runtime each, AB/BA across repeats; dynamic arms switch to list A after warm-up.
for repeat in $(seq "$repeats"); do
    order=("${arms[@]}")
    if [ $((repeat % 2)) = 0 ]; then order=($(printf '%s\n' "${arms[@]}" | tac)); fi
    for arm in "${order[@]}"; do
        run="$arm-r$repeat"
        command_for "$arm"
        start "$run"
        client "$out/$run.warmup.json" "${pressures[0]}"
        case $arm in dynamic*) switch_to "$run" 1 ;; esac
        for concurrency in "${pressures[@]}"; do
            client "$out/$run-c$concurrency.json" "$concurrency"
        done
        stop
        receipts "$run"
    done
done

# 4. Nsight Systems on every rank (fork tracing, Graph nodes), one window after warm-up and the switch; per process
#    Graph node summaries, then per step and rank the layer breakdown (benchmarks.dynamic_offload_report nsys).
for arm in stock frozen_a indirect dynamic; do
    run="$arm-nsys"
    command_for "$arm"
    cmd=(nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --trace-fork-before-exec=true
         --delay "${NSYS_DELAY:-900}" --duration "${NSYS_DURATION:-60}" -o "$out/$run-%p" "${cmd[@]}")
    start "$run"
    client "$out/$run.warmup.json" "${pressures[0]}"
    case $arm in dynamic) switch_to "$run" 1 ;; esac
    client "$out/$run-c$last.json" "$last"
    stop
    for report in "$out/$run"-*.nsys-rep; do
        nsys export --type sqlite --output "${report%.nsys-rep}.sqlite" "$report"
        "${py[@]}" -m benchmarks.nsys_graph_summary "${report%.nsys-rep}.sqlite" \
            --output "${report%.nsys-rep}.graph-summary.json"
    done
done

# 5. Nsight Compute: route kernel launches at production geometry (resident and offloaded modes, before and after a
#    switch), cold caches, kernel replay.
ncu --kernel-name regex:load_cache_to_device_buffer_kernel --launch-count 16 --cache-control all \
    --replay-mode kernel --set full -o "$out/ncu-route" \
    "${py[@]}" -m runtime.cuda_dynamic_offload_correctness --allow-cuda-test --cases production
"${py[@]}" -m benchmarks.dynamic_offload_report all --dir "$out" --model "$model" --output "$out/report.json" || true
sha256sum "$out"/* "$out"/correctness/* > "$out/SHA256SUMS" 2> /dev/null || true
