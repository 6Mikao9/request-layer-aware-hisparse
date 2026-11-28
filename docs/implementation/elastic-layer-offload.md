# 按压力逐层弹性卸载的 HiSparse（issue #13）

[Issue #13](https://github.com/6Mikao9/request-layer-aware-hisparse/issues/13) 要的是用户体感接近 vLLM Hybrid HiSparse 的逐层动态卸载：低压力时 KV 全部常驻 GPU，TPOT 最低；压力上来后按启动时给定的顺序把一部分层卸到 CPU，换更高的并发、吞吐和 goodput；再往上卸载所有可卸载的层，除非 SLO 会被打破，那时让请求排队。decode 保持 CUDA Graph，支持 PD 分离的 decode 节点。

本实现以 issue #11 的固定 SGLang `e7399a46` 适配为基础，独立于共享 LRU。代码在 `runtime/elastic_*.py`：

| 模块 | 内容 |
| --- | --- |
| `elastic_config` | `hisparse-elastic-v1` 配置解析，只用标准库 |
| `elastic_memory` | CUDA VMM 弹性显存：物理 chunk 在层之间移动，不做释放再分配 |
| `elastic_policy` | 代价模型与压力控制器，纯 Python |
| `elastic_kv` | 设备侧状态与每种切换的拷贝，只依赖 torch |
| `elastic_route` | decode 热路径上的 CUDA 路由核（`elastic_route.cuh`，经固定 SGLang 的 JIT 编译）与绑定前的自检 |
| `elastic_device` | SGLang 适配：池、协调器、DSA/MiniMax/DeepSeek-V4 后端、scheduler、Graph、`/set_internal_state` |
| `elastic_sim` | CPU 上的服务模拟器，对比 full_resident、stock、fixed、hybrid 和 elastic |
| `cuda_elastic_gate` | A100 上第一件要跑的门禁：VMM、融合路由核、host 拷贝（含 C4 记录） |

现状：CPU 测试、对固定 checkout 的 dry-run 和模拟器都已跑通；A100 上的任何结果都还没有，见文末 `pending_hardware`。MiniMax-M3、DSA 系列（GLM-5、GLM-5.1、GLM-5.2-AWQ-INT4、DeepSeek-V3.2）和 DeepSeek-V4-Flash 的 C4 层都已接入。

## 与 vLLM Hybrid HiSparse 的取舍

vLLM Hybrid HiSparse 按请求管理页的驻留：放得下的页留在 GPU，放不下的整页溢出到 host；每层每步都跑融合解析器，命中驻留页就提前返回，P/D 导入直接进 host 层（[设计文档](https://docs.vllm.ai/en/latest/design/hisparse/)）。这样每层都要付一份解析器的固定开销，只要有页溢出，每层都可能触发传输。

本实现选择全局按层：同一时刻每层要么整层常驻（读逻辑行，不跑换入），要么整层卸载（原生 HiSparse 换入）。理由与 issue 评论一致：HiSparse 每层有固定开销，混合模式要付 60 层，按层卸载只对卸掉的 t 层付。全局按层还让 decode 的形状只和级别有关，steady 版可以每级一套 Graph，smooth 版可以用一张设备路由表。按请求的混合驻留没有实现；要做的话，可以在 smooth 版的路由表上再加一列每请求的驻留上界，`smooth_select` 的掩码已经是按位置算的。

## 配置

```json
{"schema": "hisparse-elastic-v1", "model": "minimax_m3", "mode": "steady",
 "layers": {"eviction_order": [[2, 1, 58, 15, 0, 51, 18, 7], [47, 13, 17, 6, 3, 9, 48, 43]],
            "always_offload": [], "pinned_resident": []},
 "slo": {"tpot_ms": 60.0, "tail_factor": 1.1, "margin_ms": 1.0},
 "profile": {"measurements": "data/a100-hisparse-graph-and-c4-20261124.json"},
 "controller": {"interval_steps": 8, "high_watermark": 0.9, "low_watermark": 0.6, "dwell_steps": 512},
 "memory": {"chunk_mib": 2, "precapture_levels": "all"},
 "features": {"write_through": true, "warm_start": true, "stale_hits": true}}
```

| 字段 | 含义 |
| --- | --- |
| `model` | `minimax_m3`（稀疏 MHA 层）、`dsa`（GLM-5/5.1/5.2、DeepSeek-V3.2 的 MLA 层）或 `deepseek_v4`（DeepSeek-V4-Flash 的 C4 层） |
| `mode` | `steady` 或 `smooth`，见“两个版本” |
| `layers.eviction_order` | 必填。压力上升时依次卸载的层组，`groups[0]` 最先离开 GPU；第 k 级卸载前 k 组。也可以给平铺的层号加 `granularity` 按 n 层分组，或写 `"by_miss_rate"`，按 profile 里的 miss 率从低到高排。没出现在任何列表里的层始终常驻 |
| `layers.always_offload` | 启动起就卸载、从不回填的层。DSA 可写 `"shared_index_followers"`，即复用上一层 top-k 的 follower 层，它们的拷贝由 anchor 的计划提前发出，卸载几乎不露出时间 |
| `layers.pinned_resident` | `by_miss_rate` 排序时排除的层 |
| `slo` | 可选。`tpot_ms`，以及模型预测乘的 `tail_factor` 和加的 `margin_ms`。给了 SLO 后，下一组卸载会把预测的 TPOT 推过 SLO 时不卸，新请求排队 |
| `profile` | 代价模型输入：每次 miss 的字节、每层选中行数、H2D 带宽、每层固定开销、每请求开销、按上下文分档的逐层 miss 率、`exposed`（每层换入未被重叠的比例）、`base_ms`（全常驻时的步长先验）。`measurements` 指向仓库内的实测数据文件，没写的字段从中拟合 |
| `controller` | 决策间隔（decode 步）、高低水位、回填前的驻留步数、预测增长的步数、回填带宽上限、级别范围与初始级别、`min_gain`（多卸一组要求的吞吐提升） |
| `memory` | `chunk_mib`（物理 chunk 大小，2 的幂且至少 2 MiB，即 A100 上的 VMM 粒度）、`precapture_levels`（steady 版启动时捕获哪些级别的 Graph，其余首次用到时捕获） |
| `features` | 见“开关与开销” |

`runtime/fixtures/elastic/` 有五份示例：`minimax_m3_steady.json`（6 组 × 8 层，按 11 月 24 日实测 miss 率从低到高，12 个最高 miss 层常驻）、`minimax_m3_smooth_slo.json`（同样 48 层按 2 层一组、TPOT SLO 60 ms）、`glm52_smooth.json`（`by_miss_rate`、3 层一组、follower 常驻卸载）、`dsv32_steady.json`（8 层一组、第 0 层常驻）和 `dsv4_flash_steady.json`（21 个 C4 层里 miss 率最低的 16 个按 11 月 24 日约 29 Ki 上下文档的实测排序，4 组 × 4 层，其余 5 个常驻）。

## 显存布局

HiSparse 设备池每层每个 KV 张量是一块 CUDA VMM 虚拟地址预留，前段是原生 hot 区（行号就是原生设备行号），后段是常驻区（第 L 行存逻辑位置 L）：

```
| hot 区：原生 size + page 行 | 常驻区：size × ratio + page 行，第 L 行 = 逻辑位置 L |
```

同一个指针加同一个行号，在两种模式下都能寻址，所以 steady 版给原生核传 hot 视图或常驻视图，smooth 版在一张 Graph 里按路由表选行。物理显存是一个 chunk 池，字节数恰好等于原生 HiSparse 设备池会分配的字节（`size + page` 行 × 每层每张量）。哪些区域需要物理页由当前状态决定：卸载层要 hot 区，常驻层要常驻区的前 tokens 行。逻辑容量是逻辑分配器上的一个上限（`Ceiling`），卸载一组后，空出来的 chunk 接到其余常驻层的常驻区后面，上限随之抬高；回填前上限先降下来。chunk 是唯一的物理单位，任何空闲 chunk 都能接到任何区域，不产生碎片。

全常驻时（第 0 级、无 `always_offload`）hot 区不占物理页，逻辑容量等于非 HiSparse SGLang 在同样预算下的容量。实现复用了固定 SGLang 的 VMM 工具（`srt/utils/cuda_vmm_utils.py`，即它给 post-capture KV backing 用的那套 `cuMemCreate`/`cuMemMap`/`cuMemSetAccess`），思路与 [vAttention](https://arxiv.org/abs/2405.04437)（ASPLOS'25）和 [kvcached](https://github.com/ovg-project/kvcached) 相同：虚拟地址不动，CUDA Graph 捕获过的指针始终有效。

卸载后，常驻区原有的 chunk 不立刻回收，按需从区域顶部往下拿（stale 行）。`ElasticMemory.stale_rows` 报告每层仍有物理页的前缀行数。smooth 版里卸载层对这些行仍能命中，回填时两版都跳过仍有效的记录。

DeepSeek-V4 的 C4 池以页为行：一行是 64 条记录（每条 576 字节值加 8 字节 scale，页补齐到 37440 字节）。hot 区就是原生 C4 池按 `_num_dsv4_physical_kv_pages` 算出的全部页，常驻区第 p 行存逻辑第 p 页，即 256 个 full token 的 64 条记录。逻辑容量按 full token 计，上限加在 SWA 分配器内部的 full 分配器上；hot 行按记录计。37440 字节的行与 2 MiB chunk 对不齐，常驻区起点落在 chunk 中间，每层约多占一个 chunk：按 dry-run 预设（21 层、每层 131072 条 hot 记录），第 0 级逻辑容量 506624 token，比行与 chunk 整除时少约 3.5%。

## 两个版本

| | steady（稳态性能） | smooth（平滑切换） |
| --- | --- | --- |
| decode Graph | 每级一套，键为 `("hisparse-elastic", level, shape)`；`precapture_levels` 启动时捕获，其余首次用到时捕获 | 一套 |
| 常驻层 | 原生非 HiSparse 路径，直接读逻辑行，不跑换入核（DSA 与 C4 的页表由一个路由核算出） | 每层先按路由表算出哪些选中位置读常驻区，余下交给原生换入核，再合并 |
| 卸载层 | 原生 HiSparse 路径 | 同上；常驻区里仍有效的 stale 行直接命中 |
| 切换 | 两个 batch 之间停下 decode，拷贝完再切 Graph | 只改路由表；回填拷贝按 `backfill_gbps` 分摊到多步，期间继续 decode |
| 额外开销 | 无每步开销；切换时有停顿，首次用到的级别要现场捕获 | 每个可切换层每步多一次路由核和一次合并核；回填期间两区都占物理页 |

decode 热路径上每层的下标计算由 `elastic_route.cuh` 的 CUDA 核完成（按仓库的[开发约定](agent-environment.md)，新的逐层 decode 热路径用 CUDA C++），经固定 SGLang 的 JIT（`load_jit`）编译，每个都替代 `ElasticKV` 里一串小 torch 算子：常驻层的 decode 页表（DSA、C4，两版都用）一次；smooth 版每层的选择、合并、写入行号各一次；shared-index 计划拷贝的请求数一次。torch 写法是参考。worker 绑定时先在设备上用随机表对比两者，不一致就拒绝启动并提示关掉 `fused_route`。

## decode、prefill 与 PD 准入

- 写穿（`write_through`）：协调器原来的 eager backup 换成逐层写穿，仍在 decode backup stream 上执行，下一次 forward 照旧等它的事件。上一步写出的记录从它实际所在的行拷到 host：卸载层取请求的设备 buffer 行，可切换的常驻层取逻辑行，始终常驻的层不写 host。于是卸载一组时 host 历史已经完整，不用整层 D2H。
- 存储：常驻层写常驻视图的逻辑行；卸载层走原生映射写 hot 行；smooth 版 decode 内按路由表算行号。
- prefill 必须 eager（launcher 和 worker 各检查一次），它的存取按当时的层模式选视图。MiniMax 读常驻视图时去掉稀疏 prefill 的 `loc_mapping`；DSA 常驻层的 prefill 页表不经 HiSparse 映射。staging 时卸载层从 hot 行备份，可切换的常驻层从逻辑行备份。
- PD decode 节点：KV 照原生直连准入进 host。卸载层照原生处理（短请求预载到 buffer，长请求全 miss），常驻层的逻辑行在 staging stream 上从 host 填充。填充没完成的请求暂时移出 waiting queue，各 rank 用一次 Gloo all-reduce 对齐完成的个数后再放回，所以 decode 步不会等一次长的 H2D。
- DeepSeek-V4：C4 层的压缩器、indexer 和 attention 调用都不带模型层号，适配在 `MQALayer.forward` 上设层上下文。压缩器经 `_translate_loc_to_hisparse_device` 取写入位置、经 `get_extra_key_buffer` 取 buffer，常驻层拿到逻辑位置和常驻视图，卸载层照原生。decode 的 indexer 按关键字调用 `swap_in_selected_pages(..., layer_id=compress_layer_id)`，hook 沿用原生参数名。每 4 个 token 才闭合一条记录，最新记录要在保留槽 H 里再被读最多 3 步，所以卸载时的 warm 除了按换入表拷，还把每个长请求的最新记录拷进槽 H。

## 切换

controller 在 TP rank 0 上每 `interval_steps` 个 decode batch 决策一次，结果在协调器的 Gloo 组上广播，各 rank 在同一个 batch 边界执行。每次决策打印一行 `hisparse_elastic_transition` 收据（状态、级别、逻辑 token、hot 行、耗时、拷贝与回收的 chunk 数）。

- 卸载（升级）：先把要离开的层最新一条记录刷到 host（写穿落后一步），再给它们接上 hot 区，按各自的换入表把常驻行拷进 hot 行，长请求的最新记录拷进保留槽 H（D2D；空闲 chunk 不够、只能回收它们自己的常驻行时改从 host 拷），然后切模式，空出的 chunk 用来抬高逻辑上限。表保持原样：新请求的表是位置 0..H−1，以前卸载过的层保留当时的 LRU 状态，两种都与拷过去的行一致。
- 回填（降级）：逻辑上限先降到回填状态放得下的值，必要时压缩（compaction）把高处的活页搬下来；然后从 host 拷回历史，跳过 stale 行里仍有效的记录。steady 版一次拷完再切模式；smooth 版每步按带宽配额拷一点，最后一步用 D2D 补上最新记录再切模式。
- 同级调整：只改逻辑上限和 hot 行数。
- 放不下的决策（比如用户钉住的级别装不下当前的 KV）在改动任何状态之前就判定为 `deferred`，下次决策再试；真正中途失败的切换会让整个 worker 组报错停止。
- 钉住级别：`POST /set_internal_state {"server_args": {"hisparse_elastic_level": 3}}` 钉在第 3 级，`null` 交回 controller。基准测试的固定级别臂就用它。

## 控制器与 SLO

步长模型沿用 [11 月 24 日实测汇总](../../A100_HISPARSE_GRAPH_AND_SLO_20261124.md)：`T(S, B) = base(B) + Σ_{l∈S} t_l(B)`，其中

```
t_l(B, ctx) = exposed_l × (fixed_us + per_request_us × B + B × min(selected_rows, ctx) × miss_l(ctx) × bytes_per_miss / h2d_bandwidth)
```

`miss_l` 按上下文分档取自实测；MiniMax 的固定开销与带宽来自 `data/a100-hisparse-graph-and-c4-20261124.json` 中换入核耗时对 miss 数的拟合（固定项约 0，约 10.9 GB/s）。`base(B)` 在服务运行时在线拟合（实测步长减去模型给出的卸载开销，指数遗忘的加权最小二乘），所以模型只需要把两个卸载集合的差值算对。SLO 检查为 `T × tail_factor + margin_ms ≤ tpot_ms`。

每次决策：

1. 观测逻辑 token 的用量和最高活页、hot 行、运行和排队的请求、平均上下文、最近的步长。
2. 有压力（排队的请求放不下，或 hot 行不够下一个 prefill 块，或发生了 retract）时，从当前级别往上，对每一级求在 chunk 池里放得下的最大 batch（运行请求的活行加 `horizon_steps` 步增长，每接纳一个新请求加它的 prompt），第一个能装下全部需求的级别即目标；SLO 不允许的级别不再往上看，请求排队。
3. 吞吐保护（`throughput_guard`）：多卸一组时，按模型预测的“batch / 步长”要比少卸一组至少高 `min_gain`，否则停在少卸的那级，并把准入上限设为那一级的 batch。
4. SLO 准入（`slo_admission`）：准入上限取模型认为不破 SLO 的最大 batch。
5. 逻辑上限只给到所选 batch 需要的量（加水位余量），多出的 chunk 留给刚卸掉的层当 stale 行，之后需要时再扩。
6. 需求低于低水位并持续 `dwell_steps` 步后回填一组；smooth 版还要求回填状态（两区都占页）放得下。

controller 只用 CPU 上的标量，rank 0 每次决策只有一次 GPU 归约（活页的最高位置）。

## 模型适配

| 模型 | 可卸载层 | 状态 |
| --- | --- | --- |
| MiniMax-M3.1-Flash（BF16 KV，TP8） | HiSparse 主池的全部稀疏层（K、V 各一块组合 buffer）；dense 层与 index 池不变 | 已接入 |
| GLM-5、GLM-5.1（含量化版）、GLM-5.2-AWQ-INT4（W4A16，BF16 KV） | `GlmMoeDsaForCausalLM` 走 `deepseek_v2` 与 DSA 后端，每层一块 MLA 组合 buffer；indexer 的 K 缓存不变。权重量化不经过 KV 路径；KV 须为 BF16，FP8 KV 会被 launcher 和建池 hook 拒绝 | 已接入 |
| DeepSeek-V3.2（含 QuantTrio AWQ，61 层） | 同 DSA | 已接入 |
| DeepSeek-V4-Flash（TP4/TP8） | 21 个 C4 层（第 2、4、…、42 层），每层一块页组合 buffer；SWA、C128、indexer K 与压缩状态池不变 | 已接入 |

DSA 的 IndexShare（GLM-5.2 有 78 层，其中 57 层复用上一层的 top-k）：follower 层不跑自己的换入核，用 anchor 的槽表，拷贝由 anchor 记录的 miss 计划在 prefetch stream 上提前发出。为此：

- steady 版里，常驻的 anchor 只要还有卸载的 follower，就照常跑换入核，但带 `skip_io`，只更新槽表和计划，不拷自己的字节；常驻的 follower 不重放计划，直接读逻辑行。
- smooth 版里，anchor 的核始终带 `skip_io`，它自己的字节和每个 follower 的字节都用 `copy_cache_planned_mla` 重放计划，请求数取 `num_real_reqs × (该层路由模式 == 卸载)`，在设备上算出，所以同一张 Graph 在两种模式下都对。
- 卸载一个 follower 或 anchor 时，按 anchor 的表把行拷进 hot 区（warm start 按表拷，正是为了这种共用表的情况）。

DeepSeek-V4 的接入：`DeepSeekV4TokenToKVPool._make_kv_pool` 构造 `HiSparseC4DevicePool` 时换上组合 buffer，`create_buffer` 逐层返回 hot 视图，所以池的 `data_ptrs` 和 host 池的设备指针都指向组合 buffer 的起点；`get_key_buffer` 按层模式返回视图；两种位置翻译只在 C4 层的 forward 内按模式改写，分配器和 staging 调用时仍走原生映射；`set_key_buffer(_fused)` 写常驻层时临时把该层 buffer 换成常驻视图。协调器侧与 DSA 共用页选择的 hook（`layer_id` 是 C4 池内的序号）。`compaction` 对 DeepSeek-V4 强制关闭：搬动一页还要同时搬它的 SWA 映射、C128 和 indexer K 条目，这部分没做，逻辑上限因此只能等高处的请求结束再降。

## 开关与开销

| 开关（默认） | 作用 | 打开的代价 / 关掉的代价 |
| --- | --- | --- |
| `write_through`（开） | 可切换的常驻层每步把上一条记录写 host | 每步每层多一个 token 的 D2H；关掉后每次卸载要把离开层的整段历史 D2H |
| `warm_start`（开） | 卸载时把常驻行按表拷进 hot 行 | 卸载时一次 D2D；关掉后长请求从全 miss 开始（短请求仍然要拷，原生快路径直接读 buffer） |
| `stale_hits`（smooth 开，steady 强制关） | 卸载层继续命中仍有物理页的常驻行 | 每层选择时多两次比较；省下的是换入字节 |
| `backfill`（开） | 压力下降后回填 | 无 |
| `compaction`（开；DeepSeek-V4 强制关） | 降上限时把高处的活页搬下来 | 降上限那一刻的 D2D 和一次 `req_to_token` 改写；关掉后只能等高处的请求结束 |
| `slo_admission`（有 SLO 时开） | 按模型限制准入 | 排队变长，换 TPOT |
| `throughput_guard`（开） | 多卸一组要换来至少 `min_gain` 的吞吐 | 无；关掉后只要装得下就卸 |
| `fused_route`（开） | CUDA 路由核 | 首次绑定时一次 JIT 编译；关掉后常驻层每层每步多十几个小 Graph 节点，smooth 版每层再多二十多个 |

模拟器里 `min_gain` 是吞吐与 TPOT 的主要旋钮：MiniMax、8–32 Ki 输入、到达率 2 和 4 时，0.02 比 0.1 吞吐高 0.4–2.3%，TPOT p50 高 21–28%。默认值偏吞吐；更看重 TPOT 时调大，或者直接给 SLO。

## 复用的实现与相关工作

- 固定 SGLang 的 HiSparse 协调器、换入核、host 池和直连准入保持原样调用；本实现只换掉 eager backup、staging、预载和 DSA 与 DeepSeek-V4 的页选择，补上常驻层的读写。C4 记录的拷贝用固定 SGLang 的 `transfer_cache_dsv4_mla`。
- 固定 SGLang 的 VMM 工具 `srt/utils/cuda_vmm_utils.py` 与 `srt/mem_cache/kv_vmm_backing.py`（post-capture KV backing）。
- issue #11 的适配框架：`_patch` 的防叠加检查、AST 符号与调用点检查、`PROVENANCE.json` 摘要（本次增加 `elastic_dependency_sha256`）、`/set_internal_state` 钩子、Gloo 上的 TP 一致性。
- [vLLM Hybrid HiSparse](https://docs.vllm.ai/en/latest/design/hisparse/)：按请求的页驻留与溢出，是本 issue 的对照对象。
- [vAttention](https://arxiv.org/abs/2405.04437)（ASPLOS'25）与 [kvcached](https://github.com/ovg-project/kvcached)：虚拟地址连续、物理页按需映射的 KV 管理。
- [LayerKV](https://arxiv.org/abs/2410.00428)：按层分配和卸载 KV、SLO 感知调度，目标是 TTFT。
- [InfiniGen](https://www.usenix.org/conference/osdi24/presentation/lee)（OSDI'24）：卸载系统中按下一层的预测只预取关键 KV。
- [ECHO](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf)（OSDI'26）：原生稀疏注意力的 KV 卸载，GPU 池按层管理、整条 decode 路径在一张 GPU graph 内，DeepSeek-V3.2 上相对 SGLang 生成吞吐最高 2.1×。

## 正确性证据（CPU）

Python 3.12 + CPU PyTorch，共 49 项：

- `runtime.test_elastic_config`（7）、`runtime.test_elastic_memory`（6）、`runtime.test_elastic_policy`（9）、`runtime.test_elastic_sim`（4）：配置规则、chunk 账本与 stale 前缀、代价模型和 controller 的决策（按需卸载、SLO 拦截与准入上限、吞吐保护、回填驻留）、模拟器。
- `runtime.test_elastic_device`（21）：在 CPU 版 `ElasticMemory`（未映射字节填 0xA5，读到 stale 数据会暴露）上运行真实的 `Runtime` 与 `ElasticKV`，页选择的协调器 hook 装在一个复述原生协调器的类上（槽表、LRU 换入核、`skip_io`、shared-index 的 miss 计划与重放）。每次 attention 读取都与该（请求，层，位置）写入的值比较。DSA 行上覆盖 steady 的 0→1→2→1→0、smooth 的分摊回填与 stale 命中、回填跳过 stale 行、关闭写穿、follower 常驻卸载而 anchor 切换（两版）、PD 直连准入、warm 的两种来源、compaction、`deferred`、钉住级别后交回 controller 自动回填、排队压力触发卸载、steady 预捕获每级 Graph 并恢复显存状态、`Ceiling`、MHA 行拷贝与 MiniMax 按块选择。DeepSeek-V4 用同一套装置换成 C4 页行（第 2、4、6、8 层，每 4 个 token 一条记录，H 为一页 64 条，逻辑分配器包在 SWA 分配器里）：steady 0→1→2→1→0、smooth 分摊回填与 stale 命中、PD 直连准入，每条读取逐字节比较 584 字节的记录；去掉 warm 对保留槽 H 的拷贝时 steady 与 PD 两项会失败。C4 池的 hook 另在复述的固定池类上测：组合 buffer 的构造与原生页数、`data_ptrs`、按层模式的视图和位置翻译、常驻与卸载两种写入、`MQALayer.forward` 的层上下文。另有 launcher 的执行规则与 worker 安装；设置 `SGLANG_SRC` 后，五份示例配置对固定 checkout 的 dry-run 全部通过（摘要、符号签名、调用点）。
- `runtime.test_elastic_route`（2）：`elastic_route.cuh` 用 clang 的 CUDA 前端按 sm_80 编主机侧和设备侧，`-Wall -Wextra` 无告警（复用 `test_shared_lru_kernel` 里按固定签名写的 SGLang/tvm-ffi 桩）；再把这些核当主机 C++ 编译运行（块内同步处每个 CUDA 线程一个系统线程），在 ratio 1 和 4 的随机表上与 torch 写法逐项比较，覆盖 strided top-k、stale 上界与有物理页的行界、越界与负位置。提交前手工做过一次变异检查：把比较符、块内同步、负位置、闭合 token、合并取值、路由基址等十处分别改坏，这项测试每处都失败。

全量 `python -m unittest discover -s runtime -t .` 为 297 项，在 macOS 上只有 `test_shared_lru_kernel.test_index_key_matches_the_python_statement` 失败：它用主机 C++ 编译器编 `.cuh`，clang 不认 PTX 内联汇编约束，与本次改动无关。

## 模拟结果（模型，不是测量）

`runtime.elastic_sim`：MiniMax-M3 TP8 预设，每 rank 16 GiB KV，300 条请求，输入 16–32 Ki、输出 256–1024，泊松到达，`minimax_m3_steady.json`。`hybrid` 是按 vLLM 设计文档写的模型（每层每步 15 µs 固定开销，溢出页按各层 miss 率换入，容量同 stock），不是 vLLM 本身。

| 到达率（条/秒） | full_resident | stock | fixed:3 | hybrid | elastic（min_gain 0.02） | elastic（0.1） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 617 tok/s，TPOT p50 23.7 ms | 507，56.3 | 605，28.4 | 617，23.8 | 613，27.8 | 615，25.1 |
| 2 | 755，25.1 | 520，56.5 | 646，28.8 | 750，38.7 | 794，33.9 | 785，33.9 |
| 4 | 765，25.1 | 525，56.4 | 653，28.8 | 760，38.6 | 802，34.0 | 797，33.9 |

过载时（2、4）模型给出 elastic 的吞吐比 hybrid 高约 5–6%，TPOT p50 低约 12%。到达率 1 时 elastic 的 TPOT 比 hybrid 高 6–17%：突发排队时 controller 卸到第 2、3 级换 TTFT，再等 `dwell_steps` 才回填。这些数只说明控制逻辑按预期工作，结论要看 A100。

## 运行

CPU，在仓库根目录：

```bash
SGLANG_SRC=/path/to/pinned/sglang python -B -m unittest runtime.test_elastic_config runtime.test_elastic_memory \
  runtime.test_elastic_policy runtime.test_elastic_sim runtime.test_elastic_device runtime.test_elastic_route
python -B -m runtime.elastic_device --config runtime/fixtures/elastic/minimax_m3_steady.json --sglang-src "$SGLANG_SRC"
python -B -m runtime.elastic_sim --config runtime/fixtures/elastic/minimax_m3_steady.json --rate 2 --requests 300
```

dry-run 不导入 torch 和 SGLang，输出摘要校验、符号检查和每一级的逻辑容量（规划预设，worker 从真实池重新推导）。GLM 的 follower 层号用 `--followers` 传入，例如 GLM-5.2 的 57 个 follower。

服务（以 MiniMax 为例；DSA 模型把配置换成 `glm52_smooth.json` 或 `dsv32_steady.json`；DeepSeek-V4-Flash 换成 `dsv4_flash_steady.json`，`--hisparse-config` 用 C4 记录计的 `top_k` 和 `device_buffer_size`，例如 11 月 24 日实测用的 512 和 1024，后者须是 64 的倍数）：

```bash
python -m runtime.launch --hisparse-elastic-config runtime/fixtures/elastic/minimax_m3_steady.json -- \
  --model-path "$MODEL_DIR" --tp-size 8 --enable-hisparse \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 4096}' \
  --disable-overlap-schedule --disable-radix-cache --disable-prefill-cuda-graph --mem-fraction-static 0.84
```

launcher 拒绝 overlap、Radix、投机解码、PD 的 prefill 侧、`--enable-mixed-chunk`、MiniMax 与 DSA 的非 BF16 KV 和 prefill CUDA Graph；PD decode 节点加 `--disaggregation-mode decode`。

## A100 上的测试与 vLLM Hybrid 对照

按顺序，每一步用新进程：

1. 门禁：`python -m runtime.cuda_elastic_gate --output elastic_gate.json`，单卡。VMM 在卸载、扩容、回填、收缩状态间移动 chunk，保留的行字节不变、新映射的行读到 0；CUDA 路由核与 torch 写法在 token、C4（ratio 4）、块选择、合并、写入行号、计划拷贝请求数和常驻页表上逐位一致；`transfer_kv_per_layer(_mla)` 与 torch 下标一致；C4 记录经 `transfer_cache_dsv4_mla`（host 侧走核指针表）的 H2D、D2D、D2H 与 torch 写的页布局逐字节一致。
2. dry-run：对镜像里的 SGLang 跑上面的 dry-run，`status` 必须为 `pass`。
3. 输出一致性：同一份短请求文件，B=1 贪心解码，elastic server 依次用 `hisparse_elastic_level` 钉在 0、中间级、最高级，与原生 HiSparse server 逐 token 比较；常驻层读的是同一份 KV，预期完全相同。再在 decode 进行中改钉住级别（升和降各一次），输出仍应相同。steady 与 smooth、Graph 与 eager decode 各跑一遍。
4. 服务性能，对照臂：`full_resident`（不开 HiSparse）、`stock`（原生 HiSparse）、`fixed:k`（elastic 钉在第 k 级，等价于固定名单）、`elastic-steady`、`elastic-smooth`、`vllm-hybrid`。所有臂同模型、同 TP、同 `--mem-fraction-static`（或 vLLM 的同等显存预算）、同 `max_running_requests`、同 HiSparse `top_k` 与 `device_buffer_size`，decode 都开 CUDA Graph。
   - 负载：固定种子的请求文件；泊松到达率扫低、中、过载三档；输入分 8–16 Ki、16–32 Ki、32–64 Ki 三档，输出 256–1024；再加一组开关式突发（30 s 高、30 s 低循环），专门考察切换。
   - PD：按 issue 评论的虚拟 PD，单机 8 × A100 先以 prefill 实例生成 KV 存 SSD，再把同一台机器切成 decode 实例导入，decode 侧用 `--disaggregation-mode decode`。
   - 指标：TPOT p50/p95/p99，TTFT p50/p95，输出 token/s，TTFT 与 TPOT 双 SLO 下的 goodput（同时满足两项 SLO 的请求速率，即 [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)（OSDI'24）的口径；另报在 SLO 达成率 90% 时可承受的最大到达率），decode 日志里的实际 `#running-req`，retract 次数，级别随时间的曲线和每次切换的耗时（收据），PCIe H2D 带宽（`nvidia-smi dmon` 或 Nsight），每卡峰值显存。
   - 重复：每个配置 AB/BA 交错三次，报均值与 95% 区间。
5. Nsight Systems：steady 与 smooth 各一段包含一次卸载和一次回填的全 rank 时间线，看停顿、拷贝与 decode 的重叠，以及 smooth 版每层路由核的耗时。

需要带回的材料：`elastic_gate.json`；每个 rank 的 `hisparse_elastic_bound` 收据和全部 `hisparse_elastic_transition` 收据；每臂每档的指标表和原始请求级输出；decode 日志；Nsight 导出；vLLM 的版本、commit 和启动参数；`nvidia-smi` 采样。只有这些到齐，才能判断“比 vLLM Hybrid 更好”是否成立。

## 已知限制

- MiniMax 与 DSA 只支持 BF16、未量化的 KV（GLM-5.2-AWQ 的权重量化不受影响）；DeepSeek-V4 只支持原生 HiSparse 也只支持的 V4 页布局（不支持 unified KV、trtllm 的 uniform FP8）。不支持投机解码、overlap 调度、Radix、mixed chunk、PP、DP attention；prefill 必须 eager。
- dense 层、MiniMax 的 index 池、DSA indexer 的 K 缓存，以及 DeepSeek-V4 的 SWA、C128、indexer K 与压缩状态池，按全部逻辑容量静态分配，不参与弹性。
- DeepSeek-V4 的 `compaction` 强制关闭（见“模型适配”）。
- DeepSeek-V4 PD 直连准入的长请求（记录数大于 H）：在它闭合第一条新记录之前（最多 3 步），卸载层按原生规则从保留槽 H 读最新一条 prompt 记录，而原生直连准入没有往槽 H 装数据。这是原生 DeepSeek-V4 HiSparse 的行为，本实现没有改；常驻层不受影响（逻辑行由 host 填满）。这一点只是读代码得出的，没在 GPU 上确认。
- 每层每张量的区域按 2 MiB chunk 取整，最多浪费约一个 chunk；DeepSeek-V4 的页行与 chunk 对不齐，每层约多占一个 chunk（见“显存布局”）。
- steady 版首次用到的级别要现场捕获 Graph，那一次切换会慢；`precapture_levels: "all"` 在启动时全部捕获，启动时间随级别数增加。
- smooth 版回填期间返回的层两区都占物理页，所以回填要等需求降到这个状态放得下。
- CUDA 路由核在 CPU 上只做过两件事：clang 的 CUDA 前端按 sm_80 做语法与类型检查，以及把核当主机 C++ 编译运行、与 torch 写法逐项对照；nvcc 构建和 GPU 上的运行都待实机。
- smooth 版每层仍是三次 launch（选择、原生换入、合并）。把选择与合并并进原生换入核可以降到一次（issue #11 的路由核就是这样改原生核的），这里没有做；TileLang 版本也没有做。
- 按请求的混合驻留没有实现。

## pending_hardware

启动收据的 `pending_hardware` 列出 `sm80_vmm_map_unmap`、`fused_route_self_check`、`c4_page_transfer`、`steady_graph_sets_capture`、`tp8_level_broadcast`、`pd_decode_admission_fill`、`serving_tpot_throughput_goodput`、`vllm_hybrid_hisparse_comparison` 和 `nsys_transition_timeline`，均未在 GPU 上运行。
