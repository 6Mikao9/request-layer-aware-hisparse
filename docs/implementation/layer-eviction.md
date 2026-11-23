# 固定层 KV 卸载的实现与稳态验收

第一笔改动在同一份 vLLM 中增加一种物理布局。让 GPU 上总是会等主机回填的层尽量保留完整历史，把回填便宜的层放在主机。原生 vLLM Hybrid 继续作为默认入口和主要对照。只要 CPU 合同、运行时接线、构建配方和可复现实验齐全，就可以提交代码 PR。研究结论等第二阶段用相同解码批量在真机测出。

## 从一行 KV 开始

请求是一路独立生成的对话。prefill 把提示词处理成各层历史 KV；decode 每步为仍活跃的请求生成新 token。一个“逻辑位置”是请求内的 token 序号。“物理行”是某一层 KV 张量中可由注意力核直接读取的下标。页由连续的物理行组成。模型 indexer 给出 top-k 逻辑位置；HiSparse 先查驻留页，再查 GPU 热缓存，最后从锁页主机内存取缺失行。LRU 为后续需要腾出热槽。常驻层也必须把逻辑位置换成正确的物理行。若错把请求 A 的物理行交给 B，形状完全正确但输出已经损坏。

TP8 把一次推理分给八个 GPU 进程，每个进程是一个 rank，并在层间进行 collective 通信。主目标设备是 8×A100 SXM4 80 GB，计算架构 SM80。GPU 本地 KV 用 HBM。此平台 GPU 间可用 NVLink；主机 DDR 中锁页缓存的 GPU 读取经过主机路径（PCIe）。记录实机 NUMA 与 GPU 拓扑，分别测有效主机载荷和 NCCL 等待。NVLink 标称带宽不能换算为 KV 读主机的带宽。

## 已知版本及证据

| 用途 | 公开入口 | 落地条件 |
|---|---|---|
| GLM 主对照 | `neuralmagic/vllm` 固定 `e8ef1e07bd2f174bebfe34c3a3e35e952931efb1`。GLM-5.2-AWQ-INT4 以模型配置选择 compressed-tensors 加载，BF16 主 KV，TP8 | 固定分支 FlashMLA Sparse 对架构有门槛，注册表里没有第三方 Triton sparse 后端。本项目曾用 SM80 适配完成小规模 eager 诊断。代码交付中列出实际可取得的适配源、版本和许可，再验 graph。单独 checkout 上游不等于装好该适配器。[分支源码](https://github.com/neuralmagic/vllm/tree/e8ef1e07bd2f174bebfe34c3a3e35e952931efb1) · [SM80 参考移植](https://github.com/zzjchen/GLM-5.2_8A100_Serving/tree/e5e32381cb05e8fe3bac6ded7bd009b0e4ced2d4) · [检查点](https://huggingface.co/cyankiwi/GLM-5.2-AWQ-INT4) |
| MiniMax 机制观测 | SGLang `e7399a46f420d5903c3f2eddee74f0f9cfe68e11`，MiniMax-M3.1-Flash。60 层 MSA，top-k 2048，参考热槽 H=4096。主行、索引 K 分开计量 | 参照 `MiniMaxM3Attention`、`HiSparseCoordinator.swap_in_selected_pages`、`_run_swap_in_kernel`。B32 A100 原生 Graph 上高低名单各十层，逐步最大 rank 的核时间差 787.62 µs/层；T40 冻结名单相对三组随机少 3.16–5.25 ms/步原生核小计。旧 8×L20 PCIe、16K/B17 上 T=10/20/30 的平均 rank 核小计差仅约 0.79/1.22/1.26 ms，原生整步约 61 ms。两组都是同一原生轨迹中重新分组的描述量。[模型实现](https://github.com/sgl-project/sglang/blob/e7399a46f420d5903c3f2eddee74f0f9cfe68e11/python/sglang/srt/models/minimax_m3.py) · [换入实现](https://github.com/sgl-project/sglang/blob/e7399a46f420d5903c3f2eddee74f0f9cfe68e11/python/sglang/srt/managers/hisparse_coordinator.py) · [公开 A100 结果](../../NATIVE_GRAPH_20261123.md) |
| 最近主线 | vLLM #53781 合入了 sparse-MLA 的 HiSparse 支持；团队报告 GLM-5.3 的按压力逐页释放。#33398 已提出部分层常驻、部分层卸载及 sparse top-k 换入 | 将新的 mainline 作为单列的可移植性/强基线复核。每个表写 SHA、consumer 核和设备；不同后端或硬件的差值不能归给布局。研究点是等预算下如何选层以及暴露等待能否减少。[合入 PR](https://github.com/vllm-project/vllm/pull/53781) · [按页设计](https://vllm-project.github.io/2026/09/08/glm53-part1-hybrid-sparse-offloading.html) · [分层 RFC](https://github.com/vllm-project/vllm/issues/33398) · [HiSparse 论文](https://arxiv.org/abs/2608.07009) |

## 配置与模块边界

在本仓库的研究启动器接入 `--hisparse-study-config PATH`。这是项目接口。启动器在初始化分配器、绑定 worker 和捕图前校验一次，记录解析后的配置和摘要，全部 rank 必须一致。示例中按零起始层号固定卸载 GLM 2–5 层，仅作接线演示。

```json
{
  "schema": "hisparse-study-config-v1",
  "placement": {"mode": "fixed_layers", "offload_layers": [2,3,4,5], "selection": "fixed"},
  "hot_pool": {"mode": "isolated", "scope": "request_rank"},
  "diagnostics": {"mode": "none"},
  "features": {}
}
```

未启用项目配置时与 `placement.mode=native` 走相同上游路径。`full_resident` 走原生全驻留；`fixed_layers` 按 S 执行，空名单即全驻留、全集即全层主机历史+热缓存。全集与主动按页释放的 native 是不同基线。`selection=random|calibrated` 的最终输入仍是一份显式层名单；将源数据划分、种子、算法与最终名单摘要随配置冻结。越界、重复 ID、模型架构/层数不符、无效组映射、超过资源预算均在捕图前报错。此实验固定 `hot_pool.mode=isolated`，多个请求/rank 独占各自热缓存；共享池属于其他实验因子。

| 代码责任 | 实际挂接点 | 交付物和不变量 |
|---|---|---|
| 静态计划及页池 | `HiSparseCoordinator.reclaim_resident_blocks`、`_plan_spill`、`build_row_mirrors`、`update_spills`、`free`；同时追到 HMA 建池及绑定 | 由 S 和真实每 rank 层行宽、页长、对齐、热槽容量与尾页，计算驻留和卸载两类层的实际分配。若一个 HMA block id 同时预留所有层，需要拆分可独立分配的 backing/池，并让调度器看到可重新分配的页。只改层标签不会减少已分配 tensor 大小。非 S 常驻页引用不随 S 归还；记录物理唯一分配，防止重复记账。[coordinator](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/v1/core/hisparse_coordinator.py) · [池构造](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/v1/core/kv_cache_utils.py) |
| 主机镜像与请求生命周期 | `HiSparseConnectorWorker.register_kv_caches`、`start_step`、`stage_row_mirror_mapping`、`_enqueue_layer_mirror`、`take_transfer_updates`，以及 `HiSparseCacheHandle.write_target` / `finish_kv_update` | 给 S 内的层取得完整正确的主机历史。尾页可写；封页后异步复制并在传输与所有旧读者的依赖满足后复用。跨请求共享/取消/异常时遵守原有引用计数和事件。pool 容量不足时以原生机制延后或拒绝准入。S 外无需主机行服务时关闭对应镜像是一项独立消融，双方仍受相同 pinned-host 上限约束。[worker](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py) · [写入 handle](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/v1/hisparse/runtime.py#L1083-L1144) |
| 索引、换入、attention | `SparseMLAIndexGroupBuilder.register_layer`、`HiSparseMLAIndexGroup.convert_decode_logical_to_physical_topk`、`physical_kv_cache`；`HiSparseRuntime.swap_in`、`_resolve_and_stage_group`、`_swap_rows`、`wait_for_staged_swap` | 输入为每 query 的请求映射、逻辑 top-k 及 block table；输出是与真实 consumer 的 cache view/stride 配套的物理行及可选有效数。padding 必须维持 -1，不跨请求读。正常混合路径使用 `hisparse_resolve_residency`/`hisparse_gather_compact`；全驻留直接转换也必须保持行序、重复及有效数语义。使用参数已有的输出 buffer。[索引组](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/v1/attention/backends/mla/index_group.py) · [原生换入](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/v1/hisparse/runtime.py#L847-L1031) · [CUDA 算子](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/csrc/libtorch_stable/hisparse_kernels.cu) |

**先实现同一索引组内布局一致的版本。** GLM 的 78 层按模型定义组成 21 个索引组。L0、L1 各一组，L2 起每四层共用本步逻辑选集。先验证整组常驻/整组卸载，再开放组内混合。混合组共享的是逻辑位置：常驻成员拿自己的 resident block table；卸载成员共享 miss 列表/目标行的前提是热槽编号、LRU 状态、stride、源行映射全部兼容。不同 view 必须分别生成对应的物理行。即使组首常驻，它仍发布本步选集，触发组内卸载成员的 resolve/gather。检查 follower 能否与前层重叠，直到其 consumer 使用前才按本层事件等待。新步、取消请求与槽位重新分配时重置版本。对无共享 indexer 的组，仅在本步真实选集可用后换入。[模型分组逻辑](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/models/deepseek_v32/attention.py) · [等待与 gather](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/v1/hisparse/runtime.py#L929-L986)

静态名单、设备端掩码、工作区和事件在建池/捕图前准备。生产 decode 热路径严禁为做选层而执行 `.item()`、逐层 D2H 读数、设备全局同步、host 逐行拼装、临时大张量或为常驻层额外复制 KV。需要取一行主机 KV 的工作交给原生设备算子；减少多余复制和同步不代表消除真实 miss 数据传输。诊断计数使用预分配设备 buffer，窗后批量取回。保留同布局的“走通用映射”和“直接驻留映射”两臂作因果对照。

## CPU 实现顺序和可验收样例

1. 在研究仓库提交固定 SHA、补丁及单命令准备入口，入口校验基础树和所需符号、运行 `git apply --check` 后才应用。固定 Hybrid `runtime.py` 的 SHA-256 为 `25914537a0532fa36b69d712bd3304a5704aa2c84f077f552d7163e745b7aae4`。依赖升级时重新核对算子 schema、index group 输出和 worker 持有的物理 view。明确产出运行时补丁、CPU 测试和稍后由维护者执行的 GPU runner。CLI 必须真的传到初始化/服务入口。
2. 构建纯 CPU 参考表。设层 0–3、两请求 A/B，页长 2，历史长度 6/5，每层热槽至少容纳同时访问的有效选集。用 `(请求,层,位置)` 生成唯一行值。A 选 `[0,3,3,5,-1]`，B 选 `[0,4,1,-1,-1]`；另备全 padding query。固定 S={1,3}，预测每个有效输出的值，检查按调用约定的 valid counts 和 padding。记录实际 host 源行、写入热槽、物理 page id 与被覆盖位置；第二步重复相同期望零新增数据读，再追加 token 触发尾页封存、重用和首行 miss。驻留臂给 A/B 分配不同 block id，故意交错映射。
3. 对比未启用与 native 模式的调用轨迹、计划和返回值。S 空集与全驻留参考逐元素比；S 全集与 CPU 全卸载 oracle 比；混合与逻辑原始值逐位置比。测共享选集的 4 层组、首层常驻、仅首层卸载、全组常驻、全组卸载和独立 indexer。随机层名单用固定种子；八 rank 可用同进程八份不同映射模拟，验证配置一致、映射独立与最满 rank 拒绝准入。
4. 容量测试先填 A，再将 S 中历史搬到 CPU，记录可复用的实体物理页数，立即分配给 B 并覆写哨兵，逐层验证 A 输出。模拟异步完成前/后的读、取消、页迟到、重复完成和 epoch 变化，确保旧操作不能写回新请求。测试首 token、短于 top-k、末尾半页、多步 LRU、重复 slot、写请求失败回滚和热槽不足。不可支持的前缀共享、投机验证或混合 prefill/decode 模式须在启用候选策略时显式拒绝；原生模式仍能使用上游支持的功能。
5. PR 提供 CPU 安装命令、pytest 精确命令、实际通过项和跳过项。Linux CI 使用空 CUDA 设备列表跑 CPU 测试；另一份 build-only CI 从版本锁指定的 CUDA **devel** 镜像、匹配 CUDA 的 PyTorch 开发包、编译器、CMake/Ninja 构建，设置 `TORCH_CUDA_ARCH_LIST=8.0`（CMake 自身控制的 CUDA 目标再传 `CMAKE_CUDA_ARCHITECTURES=80`），检查实际 nvcc 命令与 `cuobjdump --list-elf` 输出含 SM80，保存依赖版本和产物 SHA。Triton JIT 的目标核另在真实 SM80 首次编译验收。纯 CPU runner 没有 nvcc 时如实记为“构建待验”。编译通过不能确认算子已运行、Graph 可回放或数值正确。

## GPU 执行步骤和开关

1. 在每个臂都用的 SM80 后端先复验小张量逻辑行→物理行→attention 的数值，随后用真实 GLM 权重启 TP8。研究 runner 负责消费项目 JSON 并把扩展注册进 vLLM 子进程；不能把本项目参数直接传给 stock `vllm serve`。公共 PR 必须提供一条从配置到实际启动的命令、`--dry-run` 输出、镜像摘要和补丁 SHA。参数骨架为 `python -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --tensor-parallel-size 8 --dtype bfloat16 --kv-cache-dtype bfloat16 --block-size 64 --kv-cache-memory-bytes "$KV_BYTES_PER_RANK"`，再由 runner 注入经注册的 SM80 backend、`hisparse_config.host_pool_gib`、`device_buffer_size=4096` 与项目配置。预算由维护者按探测结果设定。先验证 eager 两臂，再以可捕获的同一 consumer 比 Graph 两臂；曾有 eager-only 适配，移除参数本身不会带来 Graph 支持。记录八 rank 的 `cudaGraphLaunch`、真实回放步、图形状及是否发生 eager 回退。
2. 用模型上下文和自动 KV 池实测可容纳的负载，不设硬编码的全服务 max-total-tokens。测 stride、真实块数、尾部/热槽、索引、元数据、图工作区、prefill 峰值与每 rank 峰值，另记锁页 DRAM（按节点去重）、有效使用量和瞬时余量。两臂匹配实际物理容量上限，保留页对齐误差和实际 B；使用相同冻结输入、token 预算、已消费前缀、采样设置、TP、图形状和 host NUMA。独立留出来源与已用于选层的输入隔离。匹配不到的点列明 OOM/调度原因。
3. 开始计量前完成预填、既定布局、缓存及图预热。在同长度桶、同实际 B 的首段匹配稳定 decode 区间，用调度收据和客户端逐 token 交付对齐起止。先在 16–32 Ki 段，再在一个实测可达的长上下文/高压桶重复。原生 Hybrid 要确认出现驻留释放/实际回填；弱压力另外报告。每请求 TPOT 等于窗内首末 token 到达间隔除以其间生成的 token 数；报请求 p50/p95。窗口输出 token/s 为同边界内全部新增 token 除以墙钟跨度。请求变化、首 token 与过渡等待另列。主性能来自 `diagnostics.mode=none`；计数和 Nsight 各用配对诊断波。运行顺序交错，多次独立波及来源，按波/来源聚类给 95% 区间。

| 改动因子 | 开关及配对 | 读数 |
|---|---|---|
| 分层布局 | native Hybrid（允许调其页压力水位）、full_resident（装得下时）、fixed 全卸载、fixed 部分卸载 | 同预算、相同并发下的基线；先核实物理容量，再看稳态指标 |
| 选层名单 | 同卸载字节量/层数，低 miss、随机 13/37/71、高 miss；另测按校准集中本层暴露等待/可释放 GiB 从小到大选。GLM 随机基线匹配组首/后续位置 | 冻结每层八 rank 的加权 miss 分母、逐步最慢 rank 和名单。MiniMax 首选 T40；GLM 按实际容量重新算 T |
| 常驻旁路 | 同一 fixed 布局，通用 resolver 与直接 resident 映射各跑一遍 | 节点数、映射/等待、物理行值；只多占 HBM 的结果不合并 |
| 精确预取 | 主对照两臂都用原生预取；消融逐臂把 follower gather 延至本层使用前。保持逻辑 top-k、LRU 和实际回填字节相同 | 报组首/后续层、等待/整步变化和名单×预取交互。copy stream 有核不等于 consumer 正在等待 |
| 镜像与观测 | 同布局切换仅 S 镜像/全层额外镜像，须有相同主机预算；显式 features 布尔开关分别控制旁路、镜像优化、预取消融。计数与 Nsight 分别独立开启 | 对比 D2H 字节、队列和热路径开销，记录每次实际生效的开关。与共享热池组合留待独立因子实验 |

只有同一份逻辑输入与输出语义验证后才保留对照。设备诊断按 `arm,wave,step,rank,group,layer` 记录历史有效位置、resident/hot/miss、有效回填行/字节、resolver/gather 次数、KV 就绪、consumer 其他依赖满足时刻和 NCCL 等待。总线计数可用时与有效字节分栏。层内分解先选同一步同层完整 decoder 最慢 rank，再取其余区间；整步另报八 rank 的 forward 包络。跨 rank、并行 stream 不能简单相加。控制缺失字节、地址及 stride 后对比零 miss/首行/多行、单层集中/多层分散、同地址 HBM/主机源，用于解释固定开销。单算子或轨迹重算不提供端到端收益。

**性能验收建议。** 解盲前固定两个目标压力点、随机名单、最小有用改进和 p95 回退容限。在相同 HBM/host 上限和实际 B 下，以原生 Hybrid、原生预取开启的稳态结果为分母，目标是 decode 输出 token/s 提升约 5%，配对区间排除零，并且 TPOT p95 满足预先写下的容限。若低 miss 与随机相当，报告名单效应未检出；若所有固定名单都胜过 Hybrid，继续检查旁路节点及其暴露等待。若收益只出现在 eager 或单核，按对应范围记录。代码 PR 可先按正确性合入；结果表为每个工作点保留实测、未测、超预算、语义失配或区间未决状态。


## 可选的主 KV 传输与计算重叠

**GLM 的组间位置。** 增加一份启动前冻结的 S，优先保留产生索引的组首的主 KV，从同组后续层中选取卸载成员。组首的 indexer 仍须计算本步真实 top-k；组首常驻仅改变主 KV 布局。在按层数 T、释放字节、组长度和组内位置匹配的随机名单内，再比较低 miss 与随机名单。另用同 T/字节但交换组首与后续层角色的名单单独测角色效应。L0、L1 是单成员组，单列。不同名单都与原生 Hybrid、预取开启的强基线在同预算下比较。[固定模型分组](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/models/deepseek_v32/attention.py)

**按依赖调度回填。** 组首 indexer 结束后发布一次本步逻辑 top-k 及就绪事件。copy stream 等此事件，根据本请求、本组实际布局生成紧凑的缺失源行和目标行，再为各卸载成员发起 gather 并分别记录就绪事件。计算流在同一期间可执行常驻组首 attention 或中间层计算；只在读取某个成员的主 KV 前等待该成员事件。下一步、另一请求或另一个索引组不得使用尚未发布的选集。MiniMax 没有验证同一步共享选集的层按本层 indexer 完成后取行。[原生事件与逐层换入](https://github.com/neuralmagic/vllm/blob/e8ef1e07bd/vllm/v1/hisparse/runtime.py#L929-L986) · [模型适配说明](model-adapters.md)

**可选的层内精确分块。** 给这一项独立的布尔开关，默认关闭。当前层真实 top-k 已知后，先按可读行与待回填行将有效选集分成互不重叠的块，保留每个逻辑条目的出现次数、mask、位置及分数缩放。计算流先处理就绪块，copy stream 同时载入其他块。每块保存逐 query/head 的最大 logit、移位后的指数和及未归一化加权 V 和；按全局最大 logit 重标定各块再累加，最后统一归一化。空块贡献零，所有行无效时沿用原生 mask/output 约定。异步写入完成前禁止消费该块，消费结束前热槽禁止复用。此处“精确”指相同选集与数学结果，浮点求和顺序改变需要逐输出核对 dtype 对应容差。先以 CPU 密集参考覆盖极大/极小 logits、重复项、全 padding 和乱序完成，再在目标 consumer 做配对。

在同一固定名单和预算下跑跨层原生预取开/关 × 层内分块开/关四臂；再与名单层数、角色和组内位置均匹配的对照重复。关闭跨层预取时仍使用相同的本步选集与紧凑 miss，只把传输推迟至本层消费前。额外分块临时缓冲计入 HBM，记录实测有效字节、事件、重叠区间、消费者暴露等待及无插桩稳态 TPOT/输出 token/s。跨层预取主基线保持原生开启；层内分块收益须对比仅开启跨层预取的一臂。
