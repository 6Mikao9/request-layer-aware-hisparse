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

#### Resolver 开关：`features.owner_summary`、`features.touched_bitmap`

[2026-11-25 的 A100 profiling](../../results/a100-hisparse-profiling-20261125/README.md) 中，B48、60 层全命中时共享 arena 的换入核仍比隔离 arena 多约 724 µs，GPU DRAM 读取为 190.80 MB。两个开关对应[单核实测的优先消融](shared-lru-scan-microbenchmark.md#优先消融)，只改变 resolver 找行的方式，命中、victim、payload 和各张表都不变。两者默认关闭，可单独或同时打开，只对 `hot_pool.layout=arena` 合法：

- `owner_summary`（命中查找）：每 32 个 arena 行配一个 64 位字，记录哪些 plane 在该块拥有本代的行。阶段 3 按物理块扫描域，只在本 plane 位为 1 的块读取 owner 和反向 token，不再遍历 recency 列表。发布 owner 之后，收到 victim 的块按 owner 表重算。每卡增加 (slots + 1) × ⌈arena 行数 / 32⌉ × 8 B，MiniMax TP8、128 槽为 7,925,760 B，账本和计划的容量扣减已计入。
- `touched_bitmap`（稳定 LRU 更新）：每次调用在共享内存中为命中行和准入行置位，FIFO 只保留准入行。victim 扫描和阶段 9 改为测位，不再重复 owner/token/hash 查找。阶段 9 每线程一次读 8 项；某块没有触及行时只过一次屏障，原地保留或整体平移；触及行按位序追加，不再做 bitonic 排序。每个 CTA 的共享内存增加 ⌈域行数 / 32⌉ × 4 B，60 层共享域合计 131,632 B，低于 sm_80 的 166,912 B 上限；超出时绑定前报错。

另外修正了旁路臂的一个问题：recency 表只覆盖共享域，不含 scratch 行，原 host 检查却要求它等于 arena 行数，旁路臂在第一次长路径调用时会被拒绝。现在只要求表覆盖到域末尾，账本也按实际行数计费。

CPU 侧的证据：`runtime.test_shared_lru_opt` 让四组开关的 `KernelMirror` 在每种策略的随机轨迹上逐次比较输出、拷贝、计数和各张表，并检查维护后的摘要始终等于按 owner 重算的结果；其中长轨迹会让行在层间反复迁移。该模块还用逐线程的写法核对 CUDA 的分块压缩、按位追加和摘要遍历的下标计算。回放中，开关臂与关臂的 selection、输出和设备字节摘要相同。

```bash
python -B -m unittest runtime.test_shared_lru_opt runtime.test_benchmark_shared_lru_scan
python -B -m runtime.shared_lru_replay --fixture runtime/fixtures/shared_lru_fixture.json \
  --config runtime/fixtures/shared_lru_arms/shared_arena_optimized.json --output /tmp/shared-lru-optimized.json
```

实机顺序：先构建并过正确性门，两个门默认对全部开关组都与 mirror 逐次比较，生产几何门同时比较摘要；再跑受控单核探针。探针对 arena 臂先跑开关全关，其余开关组必须复现它的输出、owner 所属层、recency 顺序和占用（`logical_sha256`）才保留计时；`fragmented` 布局会把被测域的行打散到所有块，用来测摘要在碎片化所有权下的收益和维护成本。之后按计划里的配对（每个开关单独、两者合并、同工程水平的 isolated 与 shared，以及与 stock 的对比）跑同批次 Graph 服务。

```bash
python -m runtime.shared_lru_device build-receipt --allow-cuda-build \
  --config runtime/fixtures/shared_lru_arms/shared_arena_optimized.json --output "$OUTPUT_DIR/build-optimized.json"
python -m runtime.cuda_shared_lru_correctness --allow-cuda-test --output "$OUTPUT_DIR/gate.json"
python -m runtime.cuda_shared_lru_production_correctness --allow-cuda-test --all-planes \
  --output "$OUTPUT_DIR/production-gate.json"
python -m runtime.benchmark_shared_lru_scan --allow-cuda-test --arms stock isolated shared --batches 48 \
  --layers 60 --miss-blocks 0 2 4 --layouts contiguous fragmented --output "$OUTPUT_DIR/scan-switches.json"
```

#### 按键索引与增量 LRU：`features.hit_index`、`features.incremental_lru`、`features.compact_owner`

[Issue #6](https://github.com/6Mikao9/request-layer-aware-hisparse/issues/6)。上面两个开关只让整域扫描变快：阶段 3 仍要按块走完整个域，阶段 9 仍要读写整张 recency 表，60 层共享域 D = 245,760 行，每次调用的工作量随 D 增长。这三个开关把这两处换成只碰本次选中的键、受影响的条目和 victim 槽的结构，命中、victim 顺序、拷贝和计数仍与参考实现逐次相同。三者默认关闭，只对 arena 合法；`owner_summary` 与 `hit_index`、`touched_bitmap` 与 `incremental_lru` 分别是同一阶段的两种做法，配置解析拒绝同时打开。

- `hit_index`（命中查找）：每个请求槽一张两选桶索引，键是 (plane, token)，经 murmur3 fmix64 得到左半区和右半区各一个桶。每桶 16 个 int32，条目为 `指纹 << row_bits | arena 行号`，60 层时行号占 18 位、指纹 13 位。阶段 3 对每个选中键读两个 64 B 的桶，指纹相同的候选行再用 owner 标签和反向 token 确认，所以误判不会产生错误命中。发布时先删除 victim 的旧键，再把新键插入较空的桶（相同则取左桶），每个空位用 CAS 认领。两个桶都满时把该槽的标志置位，该槽在本代剩余时间内改走逐行扫描，结果仍然精确；`init_slot` 拷入模板索引并清零标志。容量按可承载键的行数的两倍分配：MiniMax 每槽 2 × 15,360 桶，模板最满的桶为 10/16，平均 8/16。
- `incremental_lru`（LRU 更新）：每个 recency 域一个环，容量 C = D + (使用该域的层数 + 1) × K，共享域为 370,688 项，隔离域为 8,192 项。环里 [head, tail) 中未标记死亡的条目按顺序就是原来的 recency 列表；另有每项一个死亡位、每行一个环内位置，以及每域的 head、tail 和重建次数。一次调用只对本次触及的行（LRU 为命中和准入，FIFO 为准入）标记旧条目死亡，再按行号排序追加到 tail；victim 扫描从 head 开始并跳过死条目，head 前移越过本次用掉的条目。两种情况会先做一次稳定重建，即把存活条目压到环首：本次追加可能溢出，或者本次是该域的第一个层且 tail 已超过 D + K。按层序执行时，后者每个 decode step 发生一次，开销与环长成正比，大约每层摊到 (D + L × K) / L 项，共享域约 6,144 项。
- `compact_owner`（元数据压缩）：owner 表从 int32 的 `generation << 16 | plane` 改为每行一个字节，值为层号，254 表示 taken，255 表示 free。这里不再存每行的 generation，正确性依赖两点：请求槽复用时 `init_slot` 在同一 stream 上重写整张表，旧请求的读写也都在这之前完成。跨代隔离由 `random_trace` 中的槽复用和生产门里第二代请求的检查覆盖。

元数据公式由 `shared_lru.arena_tables` 给出，S 为请求槽数，R 为 arena 行数，K 为选宽。每项与 CPU 上 `ArenaRuntime` 实际分配的张量逐项相等，由 `runtime.test_shared_lru_device` 检查。owner 表为 S·R·b 字节，b 在 `compact_owner` 时为 1，否则为 4。`hit_index` 为 S·2·⌈(R − scratch)/16⌉·16·4 字节，另有模板和每槽一个标志。`incremental_lru` 用环 S·ΣC·4、死亡位 S·Σ⌈C/32⌉·4、位置表 S·R·4 和元信息 S·域数·16 替换 `order` 的 S·ΣD·4。三个开关都不分配每次 launch 的 HBM 工作区，共享内存仍为 100,904 B/CTA。下表为 MiniMax TP8 每个 rank 的数值，S = 128，每请求的历史加 tail 载荷为 129,761,280 B，原生池每 rank 有 93 个 padded buffer：

| 臂 | arena 元数据 (B/rank) | 每槽 (B) | 扣减请求数 | 扣减后可准入 |
|---|---:|---:|---:|---:|
| shared_lru | 254,639,076 | 1,966,324 | 2 | 91 |
| + hit_index | 508,263,908 | 3,932,408 | 4 | 89 |
| + incremental_lru | 451,847,156 | 3,495,428 | 4 | 89 |
| + compact_owner | 158,546,916 | 1,229,044 | 2 | 91 |
| shared_lru_indexed（三者） | 609,379,828 | 4,724,232 | 5 | 88 |
| isolated_arena_indexed | 673,784,228 | 5,223,608 | 6 | 87 |

这些字节数是按公式与 CPU 分配核对过的，尚无 GPU 运行时数据。索引和环让每 rank 少准入 3 个请求；它们换来的是每次调用的表访问从约 20D 字节降到与 K 成正比，这个交换是否划算，要在同 HBM 的服务对照里实测。

CPU 侧的证据：

- `runtime.test_shared_lru_opt` 让全部 18 种合法开关组合的 `KernelMirror` 在每种策略的随机轨迹上逐次比较输出、拷贝、计数、owner 所属层和 recency 顺序。每次调用后还检查三点：索引恰好含每个存活键一次，环里存活条目与位置表一致，tail 之后没有死亡位。长轨迹会让环多次重建，并让 head 移动。
- 单独的用例覆盖三种情况：只重复调用非首层时的紧急重建，索引两个桶都满后退回逐行扫描，以及新一代请求拿到干净的索引。逐线程写法核对了重建压缩的下标。
- `runtime.test_shared_lru_kernel` 把 CUDA 头文件按主机 C++ 编译，用 412 组键核对 `index_key` 的桶号和指纹，并对 165 个模板变体做 host/sm_80 语法检查。
- 门禁自测（`runtime.test_shared_lru_gate`）和探针自测都经过 kernel ABI 模拟器，覆盖 8 组开关。
- 回放中，`shared_arena_indexed` 与 `shared_arena` 的 selection、输出、设备字节和计数摘要相同。

```bash
python -B -m unittest discover -s runtime -t . -p "test_*.py"
python -B -m unittest benchmarks.test_nsys_graph_summary
python -B -m runtime.shared_lru_replay --fixture runtime/fixtures/shared_lru_fixture.json \
  --config runtime/fixtures/shared_lru_arms/shared_arena_indexed.json --output /tmp/shared-lru-indexed.json
python -B -m runtime.shared_lru_device dry-run --sglang-src "$SGLANG_SRC" \
  --config runtime/fixtures/shared_lru_arms/shared_arena_indexed.json
```

dry-run 输出每个层的精确启动参数，包括模板参数、整型参数、grid、block 和动态共享内存。实机按 [`benchmarks/run_shared_lru_index_a100.sh`](../../benchmarks/run_shared_lru_index_a100.sh) 分阶段执行，各阶段在不同进程里运行：

1. JIT 构建与工具链收据。
2. 两个正确性门，覆盖全部开关组，包括 eager 和 Graph。
3. 不挂 profiler 的计时：原生、隔离、旧共享和优化共享四臂，miss 块数为 0、2、4（0%、12.5%、25%），并单独测每个开关和环重建调用。
4. Nsight Compute 采集核时、DRAM 读写字节和占用率。
5. Nsight Systems 带 `--cuda-graph-trace=node` 采集，由 `benchmarks.nsys_graph_summary` 汇总图节点、流内次序，以及图提交返回到首个 GPU 节点的排队时间。

整模服务按计划里新增的配对运行：每个开关单独、三者合并、isolated 与 shared 同为 indexed、以及 stock 对 shared_lru_indexed。这一步以及全部 GPU 数据都属于 `pending_hardware`。

## 后续实机可复用的最小 CUDA 测试

[GLM Hybrid SM80 attention 参考移植](../../reference/glm-hybrid-sm80/README.md)附固定 vLLM 源码摘要、Triton attention、BF16 ABI、CPU 签名测试和隔离补丁生成器。它解决的范围是 attention 适配参考。完整模型索引器、extension 和 Graph 仍需按实验室环境核验。

`runtime.cuda_correctness`、`runtime.cuda_production_correctness`、`runtime.cuda_request_type_correctness`、`runtime.cuda_split_correctness` 可用于检查 SM80 上的共享元数据、相同选择和有效行拷贝。先读各入口的 `--help` 和源版本/能力校验。它们需要固定 SGLang、CUDA JIT、支持该 layout 的 A100 以及明确 GPU 开关，属于无模型张量测试。维护者执行真实权重、Graph 与 Nsight 的步骤另见各方案。对照相同路径、kernel 和内存预算，避免把张量测试当推理收益。

## 已有实测

[A100 原生 Graph](../../NATIVE_GRAPH_20261123.md)和[A100 B53 eager](../../NATIVE_EAGER_B53_20261123.md)是原生 60 层、8 rank 的时序与对照证据。[L20 记录](../../L20_20261120.md)包含同框架异硬件及按原生轨迹重算的选层结果。[跨模型记录](../../CROSS_MODEL_20261122.md)包含 GLM 共享索引和 GPU 驻留 Quest 访问分析。引用时保留实际模型、数据与执行模式。
