# 可以直接查和跑的参考代码

这些代码出自早期原型。CPU 可验证接口、页所有权和配对测量逻辑。真实引擎接线、按层/共享的物理回收和当前 GPU 收益以各 issue 的验收为准。

| 路径 | 可以借用的内容 |
| --- | --- |
| [runtime/layer_eviction_policy.py](../../runtime/layer_eviction_policy.py)、[runtime/layer_eviction.py](../../runtime/layer_eviction.py) | 按层决策、generation、错层/双回收防护和生命周期。 |
| [runtime/layer_eviction_scheduler.py](../../runtime/layer_eviction_scheduler.py)、[runtime/layer_eviction_tp.py](../../runtime/layer_eviction_tp.py)、[runtime/layer_forward_tp.py](../../runtime/layer_forward_tp.py) | 请求/层调度与 TP 一致性。CPU Gloo 夹具展示跨 rank 一致提交、撤销和物理页复用。 |
| [runtime/layer_hybrid_kv.py](../../runtime/layer_hybrid_kv.py)、[runtime/layer_hybrid_pool.py](../../runtime/layer_hybrid_pool.py) | hot/host/resident 状态、主 KV 的物理寻址、计划及峰值账本。当前作为 reference。真实 serving 需验证分配器和消费者的适配。 |
| [runtime/selection_oracle.py](../../runtime/selection_oracle.py)、[runtime/shadow.py](../../runtime/shadow.py) | 同选集下不同策略的选中位置/缓存验证。回放 miss 是策略效果线索。 |
| [runtime/split_copy.py](../../runtime/split_copy.py) | 拆分换入参考。应测相同算法的原生 fused、拆分串行及适用重叠。 |
| [benchmarks/controlled_layer_protocol.py](../../benchmarks/controlled_layer_protocol.py)、[benchmarks/paired_service_plan.py](../../benchmarks/paired_service_plan.py) | 配对配置与冻结 workload。有独立的离线计划验证；真实 runner 需要补齐 engine 安装、入口和输出验收。 |
| [benchmarks/sglang_arrival_client.py](../../benchmarks/sglang_arrival_client.py)、[benchmarks/sglang_stream_client.py](../../benchmarks/sglang_stream_client.py)、[benchmarks/aggregate_pairs.py](../../benchmarks/aggregate_pairs.py) | 固定到达、首 token/后续 token 计时、稳态和配对账本。客户端并发不代表 GPU 实际 B。 |
| [runtime/PROVENANCE.json](../../runtime/PROVENANCE.json)、[runtime/NOTICE](../../runtime/NOTICE) | 固定 SGLang 来源、含 Apache-2.0 通知的衍生 CUDA/接口材料及准确摘要。原始版权和[许可证](../../runtime/LICENSE)随相关模块保留。 |
| [runtime/study_config.py](../../runtime/study_config.py) | 三项工作共用的 `hisparse-study-config-v1` 严格解析、规范摘要和能力门控，launcher 参数 `--hisparse-study-config`。 |
| [runtime/shared_lru.py](../../runtime/shared_lru.py)、[runtime/shared_lru_replay.py](../../runtime/shared_lru_replay.py) | 全层共享热池的 CPU 参考（代次、租约、LRU/FIFO、静态配额、软保底、旁路）、原生逐层 LRU 对照、与 CUDA 核逐步一致的 `KernelMirror` 和固定夹具回放。 |
| [runtime/shared_lru_kernel.cuh](../../runtime/shared_lru_kernel.cuh)、[runtime/shared_lru_device.py](../../runtime/shared_lru_device.py) | 共享热池的 SM80 解析/回填核与 SGLang 接线：连续主 KV 平面、协调器生命周期、消费者视图、启动收据和不加载模型的 dry-run。 |
| [runtime/shared_lru_plan.py](../../runtime/shared_lru_plan.py)、[runtime/cuda_shared_lru_correctness.py](../../runtime/cuda_shared_lru_correctness.py) | 每臂配置与 AB/BA 配对计划；需显式开启的 A100 正确性门（逐次对照 `KernelMirror`，覆盖 eager 与 Graph）。 |
| [runtime/layer_placement.py](../../runtime/layer_placement.py)、[runtime/layer_placement_device.py](../../runtime/layer_placement_device.py) | 固定卸载层 S 与热池联用：CPU 参考（常驻层按逻辑位置直读，S 内复用共享/隔离 arena 或原生逐层 LRU）、SGLang 拆池接线（S 热平面 + 常驻完整平面、按层路由、容量计费、常驻层直达解码/预填充）和 dry-run。 |
| [runtime/combined_plan.py](../../runtime/combined_plan.py)、[runtime/cuda_combined_correctness.py](../../runtime/cuda_combined_correctness.py) | A–E 臂与放置×借页交互的配对计划；需显式开启的 A100 门，同时检查 S 上的 arena 和常驻层的固定 Triton consumer 读数。 |

下列 scoped 命令在只包含公开白名单的目录里运行通过。参考套件用 Python 3.12 + CPU PyTorch。命令不下载权重、不需要真实服务、CUDA 设备或任何服务器地址。Windows 如不支持 Linux 特定能力，精确说明跳过项。贡献者把新 CPU 测试加到自己的验收入口。

```bash
python -B -m unittest \
  runtime.test_layer_eviction_policy runtime.test_layer_hybrid_pool \
  runtime.test_layer_eviction runtime.test_layer_eviction_scheduler \
  runtime.test_layer_eviction_tp runtime.test_layer_hybrid_kv \
  runtime.test_selection_oracle runtime.test_shadow \
  runtime.test_layer_admission runtime.test_layer_admission_tp runtime.test_layer_forward_tp \
  runtime.test_split_copy benchmarks.test_compare_runs benchmarks.test_aggregate_pairs \
  benchmarks.test_controlled_layer_protocol benchmarks.test_sglang_stream_client \
  benchmarks.test_sglang_arrival_client benchmarks.test_sglang_correctness_client \
  benchmarks.test_paired_service_plan
python -B -m unittest runtime.test_request_policy runtime.test_contracts
python -B -m benchmarks.paired_service_plan --help
```

纯标准库 smoke 可用 `runtime.test_layer_eviction_policy`、`runtime.test_layer_hybrid_pool`、`runtime.test_request_policy`、`runtime.test_contracts` 加各 `benchmarks.test_*` 中上面列出的模块。安装 CPU Torch 后再做其余张量/Gloo 覆盖。

### 跨层共享热池（[shared-lru](shared-lru.md)）

[PR #4 的基础验证与夜间实验安排](shared-lru-pr4-validation.md)记录 CPU 检查结果及待验证的 GPU 条件；[元数据账本与优化顺序](shared-lru-metadata.md)列出按卡计算的显存占用、压缩候选和逐项消融要求。

[A100 全池扫描单核实测](shared-lru-scan-microbenchmark.md)包含真实 CUDA Graph 三臂计时、生产几何正确性收据和 Nsight Systems / Compute 数据。L60、B32 全命中时，共享 resolver 约 886 µs，隔离约 65 µs；这是当前扫描实现的性能问题，尚不是完整模型结果。

CPU 命令在新 clone 的仓库根目录运行。无 Torch 时张量测试跳过；装有带 CUDA 前端的 clang++ 时会额外对 `shared_lru_kernel.cuh` 的全部模板做 host/sm_80 语法检查。

```bash
python -B -m unittest discover -s runtime -t . -p "test_shared_lru*.py"
for arm in stock isolated_arena shared_arena; do
  python -B -m runtime.shared_lru_replay --fixture runtime/fixtures/shared_lru_fixture.json \
    --config runtime/fixtures/shared_lru_arms/$arm.json --output /tmp/shared-lru-$arm.json
done
python -B -m runtime.shared_lru_device dry-run --sglang-src "$SGLANG_SRC" \
  --config runtime/fixtures/shared_lru_arms/shared_arena.json
python -B -m runtime.shared_lru_plan --spec runtime/fixtures/shared_lru_plan_spec.example.json \
  --output-dir /tmp/shared-lru-plan --replay-results /tmp/shared-lru-stock.json /tmp/shared-lru-shared_arena.json
```

`SGLANG_SRC` 指向固定 commit 的 checkout；设置后 `runtime.test_shared_lru_device` 还会对它跑 dry-run。回放结果只比较合成轨迹上的 miss，不代表服务收益。示例计划的 split 摘要为全零占位，静态配额、软保底和旁路名单为空，对应臂会写明跳过原因，需用训练来源冻结后填入。

实机入口均需显式开关，输出 JSON，不改代码即可记录结果：

```bash
python -m runtime.shared_lru_device build-receipt --allow-cuda-build \
  --config runtime/fixtures/shared_lru_arms/shared_arena.json --output "$OUTPUT_DIR/build.json"
python -m runtime.cuda_shared_lru_correctness --allow-cuda-test --output "$OUTPUT_DIR/gate.json"
python -m runtime.launch --hisparse-study-config runtime/fixtures/shared_lru_arms/shared_arena.json \
  --sm80-bf16-projections -- --model-path "$MODEL_DIR" --tp-size 8 --enable-hisparse \
  --hisparse-config '{"top_k":2048,"device_buffer_size":4096}' --page-size 128 --attention-backend triton \
  --disable-overlap-schedule --disable-radix-cache
```

每个 worker 在绑定协调器时打印一行 `hisparse_shared_lru_bound` 收据（解析后配置、能力、几何、实际分配字节）；`diagnostics.mode=counters` 在进程退出时打印逐层计数。

### 联用整层卸载和跨层热池（[combined](combined.md)）

CPU 命令同样在新 clone 的仓库根目录运行。设置 `SGLANG_SRC` 后，`runtime.test_layer_placement_device` 还会在 CPU 上执行固定版本的 `MiniMaxSparseKVPool`、`HiSparseMHAMainPool`、换入、备份和后端方法源码，并对 checkout 跑 dry-run。

```bash
python -B -m unittest runtime.test_layer_placement runtime.test_layer_placement_device \
  runtime.test_combined_plan runtime.test_combined_gate
for arm in fixed_stock_layout fixed_isolated_arena fixed_shared_arena; do
  python -B -m runtime.shared_lru_replay --fixture runtime/fixtures/shared_lru_fixture.json \
    --config runtime/fixtures/combined_arms/$arm.json --output /tmp/combined-$arm.json
done
python -B -m runtime.layer_placement_device dry-run --sglang-src "$SGLANG_SRC" \
  --config runtime/fixtures/combined_arms/combined_example.json
python -B -m runtime.combined_plan --spec runtime/fixtures/combined_plan_spec.example.json \
  --output-dir /tmp/combined-plan
```

回放夹具的 S={1,3}，层 0、2 常驻；`combined_example.json` 即 combined.md 的 [5,15] 示例。dry-run 和计划中的容量都是按 stock 同一 KV 预算的估计：常驻层按逻辑行（`size × host_to_device_ratio`）计费，热池可用的设备 token 由剩余空间反推；未提供 `--index-bytes-per-token` 时估计不含索引 KV。示例计划只含三组随机 T40 名单，校准名单和静态配额需用训练来源冻结后补入。

实机入口：

```bash
python -m runtime.cuda_combined_correctness --allow-cuda-test --output "$OUTPUT_DIR/combined-gate.json"
python -m runtime.launch --hisparse-study-config runtime/fixtures/combined_arms/combined_example.json \
  --sm80-bf16-projections -- --model-path "$MODEL_DIR" --tp-size 8 --enable-hisparse \
  --hisparse-config '{"top_k":2048,"device_buffer_size":4096}' --page-size 128 --attention-backend triton \
  --disable-overlap-schedule --disable-radix-cache
```

固定卸载臂拒绝 `--max-total-tokens`，由 SGLang 自动定容。每个 worker 绑定协调器时打印收据：原生热池布局为 `hisparse_layer_placement_bound`，arena 布局为带 `placement` 段的 `hisparse_shared_lru_bound`。收据记录 S、常驻层、两类平面的实际字节、host pinned 字节和容量计费。

## 后续实机可复用的最小 CUDA 测试

[GLM Hybrid SM80 attention 参考移植](../../reference/glm-hybrid-sm80/README.md)附固定 vLLM 源码摘要、Triton attention、BF16 ABI、CPU 签名测试和隔离补丁生成器。它解决的范围是 attention 适配参考。完整模型索引器、extension 和 Graph 仍需按实验室环境核验。

`runtime.cuda_correctness`、`runtime.cuda_production_correctness`、`runtime.cuda_request_type_correctness`、`runtime.cuda_split_correctness` 可用于检查 SM80 上的共享元数据、相同选择和有效行拷贝。先读各入口的 `--help` 和源版本/能力校验。它们需要固定 SGLang、CUDA JIT、支持该 layout 的 A100 以及明确 GPU 开关，属于无模型张量测试。维护者执行真实权重、Graph 与 Nsight 的步骤另见各方案。对照相同路径、kernel 和内存预算，避免把张量测试当推理收益。

## 已有实测

[A100 原生 Graph](../../NATIVE_GRAPH_20261123.md)和[A100 B53 eager](../../NATIVE_EAGER_B53_20261123.md)是原生 60 层、8 rank 的时序与对照证据。[L20 记录](../../L20_20261120.md)包含同框架异硬件及按原生轨迹重算的选层结果。[跨模型记录](../../CROSS_MODEL_20261122.md)包含 GLM 共享索引和 GPU 驻留 Quest 访问分析。引用时保留实际模型、数据与执行模式。
