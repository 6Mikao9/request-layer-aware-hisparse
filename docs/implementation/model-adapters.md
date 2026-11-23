# 主 KV 重叠与模型适配

主 KV 是 attention 最终读取的历史 K/V。索引缓存用于 Top-K 选择，必须保留模型原有位置、精度和驻留语义。优化目标是把已经确定要读的主 KV 搬运与能独立执行的计算重叠，减少消费前真正等数据的时间。全层共享与逐层卸载均保留不增加新重叠的对照。以下是待验收设计，GPU 行为和收益以实跑为准。

## GLM 与其他共享逻辑索引的模型

GLM 的某些层沿用索引计算层的逻辑选集。对每个 selector group 先区分 leader（真的计算 indices）与 follower（复用）。实验中可以优先卸载 follower，把组首的主 KV 在 HBM 保持完整。复用的是逻辑 token ID，各层实际 K/V 仍独立。从模型/后端的运行时 group metadata 得到关系，不能硬编码每四层或套用另一模型的列表。 [Hybrid 作者的 IndexShare 说明](https://vllm-project.github.io/2026/09/08/glm53-part1-hybrid-sparse-offloading.html) · [HiSparse 的精确层间预取](https://arxiv.org/html/2608.07009v1)

1. leader 的实际 GPU indexer 发布当前 step 的有序选择，记录生产依赖。不能在选择得出前假设本步 Top-K，也不能把上一 token 的选集直接复用。
2. 读到选集后，为 S 内的 follower 做其本层 miss resolve，在搬运流提交需要的行，实现精确预取。若 leader 主 KV 完整驻留，leader attention 和中间能独立执行的算子可以继续，让 follower 的回填有时间重叠。
3. 每个 follower 进入 attention 前只依赖它自己的 ready event 或等价的 device-side 依赖。选择准备好、索引映射可见、K/V 两份都到位、输出顺序均需核对。复制未完和读者还在用的槽均受保护。
4. 排序策略读训练期的实际有效 HBM 释放、copy ready 到消费的 slack、最慢 rank 的暴露等待。相同 T 做 follower 优先、一般低代价、随机或 leader-heavy 对照。每种布局分别跑原生预取保留与禁用，解释哪部分收益来自选层、哪部分来自预取或二者交互。用 `features.prefer_index_followers` 或封存相同层数的独立策略配置控制选层。原生预取禁用另用 `features.disable_native_prefetch`。如果实现优化原生预取的提交和分块方案，再增加独立 flag。
5. 保留两个边界夹具。组首常驻/后续层卸载时仍有真实共享选择生产与该 follower 的回填。组首卸载/后续层常驻时只有组首自己的搬运，不能重复复制 follower。若多个消费者同时在途，总 pin 和 staging 计入预算并有上限。

不要预设 follower 一定更快。选择发布后距 follower 的可用计算可能过短，或瓶颈在 PCIe 争用、拷贝 kernel 和跨 rank 到达。用 Nsight 记录每 rank 的选择完成、提交/搬运、ready、其他依赖就绪、attention 和 collective。只报告真实关键路径节省。

## 同一层主 KV 边搬边算

实现一个独立可关闭的候选。把同一选择分成已经驻留且可读的子集与一到多个待搬的子集。当前 Q 可用且第一段可读时，先算它的部分 attention；其他段准备好后算对应部分。全部段消费完再合并输出。若 native kernel 支持精确的分片状态，复用它；否则用明确的在线 softmax 状态，避免每个分块分别做 softmax 再简单相加。

对于一个查询头，用 FP32 累积每段最大 logit m、按 m 平移后的指数和 z、加权值和 a。合并两段取 m*=max(m1,m2)，对两段 z 与 a 按 exp(m_i-m*) 重标再相加，最后输出 a/z。空段、全 mask、logit scale、RoPE、attention bias/sink、重复索引和最新 token 均需在参考模型中验证。合并不同浮点顺序时给出正常 stock 重复差和明确容差。模型 Top-K 永远一致；每个条目的逻辑选中次数和语义保持不变。 [FlashAttention 的精确分块](https://arxiv.org/abs/2205.14135) · [FlashInfer attention state 合并 API](https://docs.flashinfer.ai/api/cascade.html)

- `features.overlap_copy_consume` 开关闭时，保持同样的分块、预算、kernel 和结果，在第一个计算前等待所有段就绪。比较打开时的真实重叠。
- `features.chunked_attention_merge` 单独衡量切分/合并增加的 kernel、额外 IO、FP32 scratch 和占用率。可排对照序列「原 fused」「同分片但串行」「分片重叠」。
- Graph 预先分配固定形状的阶段 buffer 与 event。实际 lengths 和计数在设备端，固定有效 lane mask。单/双缓冲的大小、host pinned 区、回填粒度和最大在途数统一入账。没有可保护的 slot 时使用明确的安全准入/回退 arm。
- 某些 GPU kernel 会从 host pinned 地址直接读，搬运可能本来就在 fused kernel 内。诊断先辨认实际 CUDA node、内存流量与 consumer 依赖，再决定能否拆成真实可重叠阶段。避免把 kernel launch 数、主机 CPU enqueue 或等待 API 的耗时等同于链路流量与 GPU 等待。
- 如借用已有 code path、CUDA kernel 或缓存实现，PR 链接到作者论文、项目和固定文件/版本，保留许可证和差异说明。不要复制一整套外部框架。记录复用范围及单独开关，引用清单应能让另一个 AI 复算。

## 各模型执行规则

| 模型/后端 | 第一阶段的专门适配 | 独立实机验证 |
| --- | --- | --- |
| MiniMax MSA + SGLang | 每层 native Top-K 已有独立读取路径。仅当运行时明确证明跨层共享选集，才复用同组预取。在本配置先尝试同层 hit-first / miss 后算、真实行合并及 zero-miss 小开销路径。保持 BF16 KV、SM80 投影、短请求和实际 tail。 | 60 层与全部 TP rank。相同原生融合路径、同实际 B、原生 Graph/优化对照。 |
| GLM DSA/IndexShare + 固定 Hybrid | 组首驻留、follower 优先候选。只在 producer 原选择发布后预取 miss。适配 MLA packed/量化 layout 和一层一条消费 ready 依赖。 | 以固定 GLM 权重、同 SM80 适配和 Graph/eager 分开为条件。验证 leader/follower 两类布置与 prefetch 开关。 |
| Qwen3-30B-A3B + Quest | 已有 Quest 数据来自 GPU 驻留 KV 的选择。用它观察页选择、层间局部性和真实 Graph 时序。默认没有 HiSparse 的 host 主 KV 搬运。 | 单列 TP4，按实际 block/page gather 计时。若以后增加 host-backed Quest，这是一条新路径，先逐行验证正确性再给出搬运对照。 |
| DeepSeek-V4-Flash C4 | C4 压缩记录、C128 与滑窗布局独立。先验证实际 selector 与 A100 路由，只在同类行型和已知逻辑选择内实施。 | 先验证 21 个 C4 层 miss。正式全模型性能、C128 对照和重叠另行测，不借用 MiniMax 槽宽或 GLM group 假设。 |

模型适配以发行时真实源码、量化布局、可运行 GPU kernel 和公开[实验环境](agent-environment.md)为准。合并 PR 前每个已实现分支必须有 CPU oracle、明确兼容检测、功能受限时的真实结果及合法关闭路径。模型无此优化前提时在报告里写不可比较原因。
