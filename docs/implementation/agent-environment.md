# 给编码助手的开发和验收入口

直接从你收到的 issue 开始工作，在本仓库建立分支并提交 PR。目标是能接入真实服务的加速代码、逐项可切换的消融和可重复的实验入口。只用 CPU 环境即可完成首轮 PR。实验室维护者在有卡时执行其中的 GPU 验收，不需要发任务的人配置软件、操作服务器、转交权重或解释源码。

## 开始前先选对任务

| 方向 | 本轮改什么 | 主对照 |
| --- | --- | --- |
| [逐层驱逐](layer-eviction.md) | 固定一组整层 CPU 历史和 GPU 热缓存，其他层完整驻留 | 同一 GLM、同一 Hybrid vLLM 后端的原生按页策略，保留可用的精确预取 |
| [全层共享 LRU](shared-lru.md) | SGLang 原生 HiSparse 的所有合格卸载层共享热槽 | 原生均分、同寻址路径均分、独立校准的静态分层、相关改进 |
| [两者联用](combined.md) | 只把已卸载层接入共池，保持常驻层和原始选集 | 固定同一卸载名单的独立热池、强静态配额和共享池，再做交叉 |

三条路线共用[配置和消融合同](ablation-contract.md)。分别做小而可审查的 PR。只有依赖代码已可用时才接入另一个 PR，优先用共同接口和 CPU 测试约束依赖。不要提交两份完整框架副本。修改上游源码时交付明确作用在哪个 commit 的补丁或独立适配模块，附应用、构建和撤销说明。

## 可用环境和证据

| 对象 | 固定信息 | 如何使用 |
| --- | --- | --- |
| 投稿方的开发环境 | Linux x86-64、Python 3.12 为主要目标，普通 CPU、Git 和联网取公开源码。macOS/Windows 只需能运行可移植测试 | 以 `python -B -m unittest` 跑指定模块。在新 clone 或临时目录里完成。深度依赖 Torch 的测试使用 CPU 版 Torch，可在无 Torch 时独立运行纯 Python 测试 |
| SGLang 运行时 | `sgl-project/sglang@e7399a46f420d5903c3f2eddee74f0f9cfe68e11`。[公开固定树](https://github.com/sgl-project/sglang/tree/e7399a46f420d5903c3f2eddee74f0f9cfe68e11) | 以该 commit 读协调器、分配器、模型及 Graph 路径。安装树的关键文件有运行时摘要验收。优先修真实 serving 入口，脱离服务的模拟只用于先行验证 |
| Hybrid 基线 | `neuralmagic/vllm@e8ef1e07bd2f174bebfe34c3a3e35e952931efb1`。[公开固定树](https://github.com/neuralmagic/vllm/tree/e8ef1e07bd2f174bebfe34c3a3e35e952931efb1) | 固定本实验的 resolver、gather、共享选集依赖。另列上游 [vLLM 合入 PR](https://github.com/vllm-project/vllm/pull/53781) 和 [Hybrid 设计](https://docs.vllm.ai/en/latest/design/hisparse/)。分支有 API/布局变更时分别测 |
| A100 主目标 | 8×NVIDIA A100 SXM4 80 GB，SM80，TP8、PP1、DP1，Linux 容器，GPU 间 NVLink/NVSwitch 拓扑经运行时确认 | 确认 rank 到本地 KV head、device 和实际互联。CPU 内存读取使用本机实测链路与 NUMA 测量，不能把 GPU 间 NVLink 额定带宽代入主机搬运。TP4 可作为功能和小规模性能点，报告单列 |
| MiniMax 压力目标 | 实验室固定 `MiniMax-M3.1-Flash`，MXFP8 权重兼容路径，SM80 BF16 投影适配，BF16 activation/KV；60 个参与诊断的 MSA 层，Top-K 2048，原生热池 H4096，例行 128 行页 | 以实际模型和张量 layout 自省现有几何。选择顺序、重复索引、padding、短请求、最新 KV、CPU 历史均沿用原生语义。实际 KV head 数会随 TP 与实现变化，不能拿每 rank 的理论 512 B 替代运行时发现值 |
| GLM 同后端对照 | `cyankiwi/GLM-5.2-AWQ-INT4`，实际为 compressed-tensors 权重配置，BF16 主 KV。A100 需要支持稀疏 attention 和 indexer 的 SM80 路径；已用固定分支的移植版做了 eager 诊断 | 使用相同适配的 Hybrid control/treatment。不要直接用 A100 不支持的 FlashMLA kernel 或把较新上游架构当成同一条实现。验证 GPU 路径和 Graph 的责任在实验室。 [模型配置](https://huggingface.co/cyankiwi/GLM-5.2-AWQ-INT4/blob/main/config.json) |
| L20 历史点 | 8×L20 TP8 的 PCIe 条件。主要用于理解 host 回填和不同压力的现象 | 历史显存、并发和带宽不同。将其视为单独硬件结果；第一版以 A100 为必须验证平台。 [公开记录](../../L20_20261120.md) |

固定实验镜像在 A100 上用内容 ID `sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87` 记录。这个 ID 用于身份验收。贡献者用公开的固定源码和自己的 CPU 环境。维护者把已经存在的实机镜像、校验过的只读权重和输入清单挂进运行容器，测试命令用 `$MODEL_DIR`、`$DATA_MANIFEST`、`$OUTPUT_DIR`、`$SGLANG_SRC`、`$VLLM_SRC` 等参数。代码和示例不得假定有能登陆的主机、云账号、访问密钥或固定宿主路径。

### 先搞清几个名词

- **Prefill** 处理已有上下文并为每层写出 KV。**Decode** 每步生成一个 token，通常需要每层读取一部分历史。压力实验统计预热后的解码窗口。
- **KV 行** 是物理 KV 层、请求序列和逻辑 token 上的一份 K/V 数据。TP 把计算或头划到多个 rank，每张卡取本地行。共享热池让不同层使用同一份**空间**，独立存放各层内容。
- **HiSparse** 将完整历史留在 CPU，GPU 维护热槽。模型的 Top-K 逻辑选择完成后，命中从热池读，缺失从 host 回填。换入结果要以物理槽顺序送给 attention。
- **Hybrid HiSparse** 让全量历史驻留页与热槽共享 GPU 资源，压力变化时调整页。GLM 的同组层可消费已知的共享逻辑选集，后续层有原生精确预取。跨层 KV 内容仍不同。
- **CUDA Graph** 把形状和地址稳定的执行捕获一次后回放。运行时 batch 可能有 padding。验收同时检查实际 B、图档位和真实回放，区分 eager 和 Graph。异步副本与读写生命周期依赖实际 stream/event。
- **TPOT** 看每个请求后续 token 间隔。**decode throughput** 在规定稳态墙钟窗口内统计最终交付的输出 token/s。**goodput** 用冻结的 SLO 计数。核时间、等待和端到端指标分别报告。

## 代码接入和本地动作

1. 读 issue 与方案，再从[消融合同](ablation-contract.md)推导你的默认路由和每个开关的对照。先写一个很小的 CPU oracle，模拟两层、两个请求、真实选择顺序和已完成/未完成事件。写明显不同的 bytes，使错层、错代次、错 rank 一定被测试抓到。
2. 对照固定的公开源码。MiniMax 的热路径从 `MiniMaxSparseAttnBackend._hisparse_swap_in_blocks` 到 `HiSparseCoordinator.swap_in_selected_blocks`，再调用 `load_blocks_to_device_buffer_mha`。检查实际消费者使用哪一个 K/V 基址和索引，不只验证策略表。查看 `alloc_device_buffer`、`map_last_loc_to_buffer`、`request_finished`、prefill staging/备份、短请求增长和 Graph replay 构造。适配层输出地址必须和真实 attention 输入对应。 [固定协调器](https://github.com/sgl-project/sglang/blob/e7399a46f420d5903c3f2eddee74f0f9cfe68e11/python/sglang/srt/managers/hisparse_coordinator.py)
3. 实现解析、配额和 lifecycle 检查后，把新路径接到真实服务器的初始化和每个 TP worker。在固定 Graph shape 使用加载前已分配的 arena、映射、临时区和计数缓冲。对 GPU ABI/tensor layout、TP 一致性、权重/KV dtype、unsupported 模式做启动验收。全部开关关闭时保持原生调用、Graph 和分配行为。
4. 运行[已发布的参考代码和 CPU 测试](reference-code.md)。新增测试用公开合成数据，纯 CPU 完成。CUDA 开发头可离线编译时，用 SM80 目标构建，保存编译命令、CUDA/Python/Torch/NCCL 版本、基础源码和产物摘要。跳过时写明原因。
5. 交一份能随 PR 审查的运行脚本。它要提供 baseline、candidate、单开关和必要组合的配置或生成器，写出解析后的 CLI、真实设备/池/批量及阶段证据。实验室接手只需给运行脚本实际路径、设备和授权窗口，不需补业务逻辑。

有固定公开源码可用时，用一份新 checkout 校验补丁。下面仅是准备命令，补丁文件名由 PR 冻结。

```bash
git clone https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
git -C "$SGLANG_SRC" checkout --detach e7399a46f420d5903c3f2eddee74f0f9cfe68e11
git -C "$SGLANG_SRC" apply --check /absolute/path/to/your-sglang.patch

git clone https://github.com/neuralmagic/vllm.git "$VLLM_SRC"
git -C "$VLLM_SRC" checkout --detach e8ef1e07bd2f174bebfe34c3a3e35e952931efb1
git -C "$VLLM_SRC" apply --check /absolute/path/to/your-vllm.patch
```

测试级别分别写入 PR。CPU 单测、SM80 离线编译、单卡正确性、TP 功能、真实服务 Graph 和干净性能各有自己的结果字段。只有实际执行的阶段写 `pass`。GPU 阶段可写 `pending_hardware`；保留触发条件、具体命令和应该生成的文件。

## GPU 维护者接手的执行顺序

1. 核对镜像与源码、A100 卡数、当前设备的 GPU UUID/拓扑、只读权重摘要、host pinned 内存、memlock/cgroup 和输出磁盘。分配自有容器，遇外部使用让出设备。运行脚本接收维护者选定的设备，禁止把所有设备写死为抢占目标。
2. 先用真实模型的小请求验证 token 顺序、Top-K、K 和 V 字节、latest、padding 及多个同时请求。单列跨 rank 的状态一致性。随后验证全部开关关闭的版本与原生输出。同初态操作用精确比对，跨独立服务把 stock 自身重复性和 logits 容差写清。
3. 从真实 `/server_info`、pool 初始化和 Graph 收据确定容量。**省略 `--max-total-tokens`**。用模型自动定池、显存实际余量、host 账本、Graph 工作区和封存输入共同决定最大的可行实际 B。OOM 保留失败与显存记录，根据证据微调显存比例或批量。保持配对臂都能实际达到同一 B 档。粒度取整、不同测量臂分到的物理容量与空闲量如实记录。
4. 统一先 prefill、建池、布局设置/预热和捕图。以同一批原始请求、实际输出前缀和相同 Graph 形状采稳态连续 decode，结束前保留自然 EOS 与真实长度。另启计数和 Nsight 诊断进程。Graph replay 的 kernel 节点和实际依赖另取，端到端性能使用无插桩进程。 [数据字段与验收](ablation-contract.md)
5. 先报告功能与容量，再判定可归因的性能。参考 A100 B32、B53 和旧 L20 结果，用它们做压力点选择与工具校准。公开现有历史只含原生 profile 与离线重算，真策略是否能加速由独立留出数据决定。 [A100 Graph](../../NATIVE_GRAPH_20261123.md) · [A100 B53 eager](../../NATIVE_EAGER_B53_20261123.md) · [跨模型](../../CROSS_MODEL_20261122.md)

## PR 中逐项填的证据

- 固定仓库和 commit、补丁或新模块路径、许可证及上游来源，实际运行的版本、命令、摘要与改动范围。
- 总开关和每个独立技术的关/开配置、兼容限制、关闭原生等价、证据文件、误差或失败验收条件。
- CPU 回归结果、SM80 产物或待实机说明，真实 GPU 入口、一次成功/失败应该产生的报告格式。
- 性能目标、原生与最强对照、资源和 workload 对齐方法、持有的反事实及可重复的实验矩阵。工程优化两边各自有明确的开关。
- 检查提交只包含代码、说明、合成夹具和已授权的匿名汇总。日志、账户与宿主信息、原始对话、真实 token IDs、profile 原始二进制、权重及私有清单均留在维护者的实验工作区。
