#!/usr/bin/env bash
# Issue #11 A100 recipe: route kernel build receipt and CUDA gate, B=1 output identity of the switched dynamic arm
# (Graph and eager decode) against fixed S, then paired CUDA Graph serving of every arm on a fresh runtime (AB/BA
# order across repeats, two pressure points, GPU memory sampled), a timed runtime switch for the dynamic arms, and
# all-rank Nsight Systems plus Nsight Compute of the route kernel. Stages never share a process.
#   MODEL_DIR=... WORKLOAD=fixed-ids.jsonl TTFT_SLO=... TPOT_SLO=... [CORRECTNESS_WORKLOAD=short.jsonl] \
#       bash benchmarks/run_dynamic_offload_a100.sh NEW_OUTPUT_DIR [minimax_m3|deepseek_v4]
set -Eeuo pipefail
umask 077
out=$1
model=${2:-minimax_m3}
mkdir "$out"
: "${MODEL_DIR:?}" "${WORKLOAD:?}" "${TTFT_SLO:?}" "${TPOT_SLO:?}"
port=${PORT:-31080}
repeats=${REPEATS:-3}
read -r -a pressures <<< "${CONCURRENCY:-16 48}"
arms_dir=runtime/fixtures/dynamic_offload_arms
py=(python3 -B)
if [ "$model" = deepseek_v4 ]; then
    prefix=dsv4 launcher=()
    serve=(--model-path "$MODEL_DIR" --tp-size 4 --enable-hisparse
           --hisparse-config '{"top_k":512,"device_buffer_size":1024}')
else
    prefix=minimax launcher=(--sm80-bf16-projections)
    serve=(--model-path "$MODEL_DIR" --tp-size 8 --enable-hisparse
           --hisparse-config '{"top_k":2048,"device_buffer_size":4096}' --page-size 128 --attention-backend triton)
fi
# Every arm runs the same execution: CUDA Graph decode, eager prefill, one scheduler stream.
serve+=(--port "$port" --disable-overlap-schedule --disable-radix-cache --disable-prefill-cuda-graph)
{ nvcc --version; nsys --version; ncu --version; "${py[@]}" --version; nvidia-smi; } > "$out/toolchain.txt" 2>&1

# 1. Route kernel: JIT build receipt and the eager/Graph correctness gate (history and production geometry).
"${py[@]}" -m runtime.dynamic_offload_device build-receipt --allow-cuda-build --model "$model" \
    --config "$arms_dir/${prefix}_dynamic.json" --output "$out/build.json"
"${py[@]}" -m runtime.cuda_dynamic_offload_correctness --allow-cuda-test --output "$out/gate.json"

# Arms: full_resident is the pinned server without HiSparse; stock is native HiSparse on every eligible layer;
# frozen_* are fixed S; indirect is the slab path with switching off; dynamic* start on list 0 and switch to list 1
# after warm-up, so their steady state pairs with frozen_b.
arms=(full_resident stock frozen_a frozen_b indirect dynamic dynamic_shadow)
command_for() {
    case $1 in
        full_resident) local args=() skip=0
                       for a in "${serve[@]}"; do
                           if [ "$skip" = 1 ]; then skip=0; continue; fi
                           case $a in --enable-hisparse) ;; --hisparse-config) skip=1 ;; *) args+=("$a") ;; esac
                       done
                       echo "${py[*]} -m sglang.launch_server ${args[*]@Q}" ;;
        stock) echo "${py[*]} -m runtime.launch ${launcher[*]} -- ${serve[*]@Q}" ;;
        *) echo "${py[*]} -m runtime.launch --hisparse-study-config $arms_dir/${prefix}_$1.json ${launcher[*]} -- \
${serve[*]@Q}" ;;
    esac
}
start() {   # also samples every GPU's memory use while the server runs (per-card peak in the ledger)
    eval "$2" > "$out/$1.server.log" 2>&1 &
    server=$!
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv -lms 500 \
        > "$out/$1.gpu-memory.csv" 2>&1 &
    sampler=$!
    for _ in $(seq 900); do
        curl -sf "http://127.0.0.1:$port/health" > /dev/null && return 0
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
client() {
    "${py[@]}" -m benchmarks.sglang_stream_client --url "http://127.0.0.1:$port/generate" --workload "$WORKLOAD" \
        --concurrency "$2" --ttft-slo-s "$TTFT_SLO" --tpot-slo-s "$TPOT_SLO" --output "$1"
}
switch() {   # $1 run label; client wall time of the TP-agreed switch, then every rank's receipt from the server log
    curl -sf -X POST "http://127.0.0.1:$port/set_internal_state" -H 'Content-Type: application/json' \
        -d '{"server_args": {"hisparse_offload_list": 1}}' -o "$out/$1.switch-response.json" \
        -w '{"http_seconds": %{time_total}}\n' > "$out/$1.switch-time.json"
    grep '"event": "hisparse_dynamic_offload_switch"' "$out/$1.server.log" > "$out/$1.switch-receipts.jsonl"
}

# 2. Output identity at B=1 (sequential greedy requests): the dynamic arm after its switch to list 1, with CUDA
#    Graph and with eager decode, against fixed S = list 1. The verdict is recorded, not enforced.
for arm in frozen_b dynamic dynamic_eager; do
    run="correctness-$arm"
    case $arm in
        dynamic_eager) start "$run" "$(command_for dynamic) --disable-cuda-graph" ;;
        *) start "$run" "$(command_for "$arm")" ;;
    esac
    case $arm in dynamic*) switch "$run" ;; esac
    "${py[@]}" -m benchmarks.sglang_correctness_client --url "http://127.0.0.1:$port/generate" \
        --workload "${CORRECTNESS_WORKLOAD:-$WORKLOAD}" --output "$out/$run.json" || true
    stop
done
"${py[@]}" - "$out"/correctness-{frozen_b,dynamic,dynamic_eager}.json > "$out/correctness-verdict.json" <<'EOF'
import json, sys
runs = {}
for path in sys.argv[1:]:
    report = json.load(open(path))
    runs[path] = {r["request_alias"]: r.get("output_ids_sha256") for r in report.get("requests", [])}
base, *others = runs.values()
mismatch = {path: sorted(a for a in base.keys() | run.keys() if base.get(a) is None or base.get(a) != run.get(a))
            for path, run in runs.items() if run is not base}
print(json.dumps(dict(baseline=sys.argv[1], identical=not any(mismatch.values()), mismatched_requests=mismatch),
                 indent=2))
EOF

# 3. Serving arms, fresh runtime each, AB/BA across repeats.
for repeat in $(seq "$repeats"); do
    order=("${arms[@]}")
    if [ $((repeat % 2)) = 0 ]; then order=($(printf '%s\n' "${arms[@]}" | tac)); fi
    for arm in "${order[@]}"; do
        run="$arm-r$repeat"
        start "$run" "$(command_for "$arm")"
        grep '"event": "hisparse_dynamic_offload_bound"' "$out/$run.server.log" > "$out/$run.bound.jsonl" || true
        client "$out/$run.warmup.json" "${pressures[0]}"
        case $arm in dynamic*) switch "$run" ;; esac
        for concurrency in "${pressures[@]}"; do
            client "$out/$run-c$concurrency.json" "$concurrency"
        done
        stop
    done
done

# 4. Nsight Systems on every rank (fork tracing, Graph nodes), one window after warm-up and the switch.
for arm in stock frozen_b indirect dynamic; do
    run="$arm-nsys"
    start "$run" "nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --trace-fork-before-exec=true \
--delay ${NSYS_DELAY:-900} --duration ${NSYS_DURATION:-60} -o $out/$run-%p $(command_for "$arm")"
    client "$out/$run.warmup.json" "${pressures[0]}"
    case $arm in dynamic) switch "$run" ;; esac
    client "$out/$run-c${pressures[-1]}.json" "${pressures[-1]}"
    stop
    for report in "$out/$run"-*.nsys-rep; do      # per-process Graph nodes, streams and queue delay
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
sha256sum "$out"/* > "$out/SHA256SUMS" 2> /dev/null || true
