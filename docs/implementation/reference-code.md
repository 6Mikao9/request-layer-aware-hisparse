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

## 后续实机可复用的最小 CUDA 测试

[GLM Hybrid SM80 attention 参考移植](../../reference/glm-hybrid-sm80/README.md)附固定 vLLM 源码摘要、Triton attention、BF16 ABI、CPU 签名测试和隔离补丁生成器。它解决的范围是 attention 适配参考。完整模型索引器、extension 和 Graph 仍需按实验室环境核验。

`runtime.cuda_correctness`、`runtime.cuda_production_correctness`、`runtime.cuda_request_type_correctness`、`runtime.cuda_split_correctness` 可用于检查 SM80 上的共享元数据、相同选择和有效行拷贝。先读各入口的 `--help` 和源版本/能力校验。它们需要固定 SGLang、CUDA JIT、支持该 layout 的 A100 以及明确 GPU 开关，属于无模型张量测试。维护者执行真实权重、Graph 与 Nsight 的步骤另见各方案。对照相同路径、kernel 和内存预算，避免把张量测试当推理收益。

## 已有实测

[A100 原生 Graph](../../NATIVE_GRAPH_20261123.md)和[A100 B53 eager](../../NATIVE_EAGER_B53_20261123.md)是原生 60 层、8 rank 的时序与对照证据。[L20 记录](../../L20_20261120.md)包含同框架异硬件及按原生轨迹重算的选层结果。[跨模型记录](../../CROSS_MODEL_20261122.md)包含 GLM 共享索引和 GPU 驻留 Quest 访问分析。引用时保留实际模型、数据与执行模式。
