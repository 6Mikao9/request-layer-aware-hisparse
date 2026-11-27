# Request- and Layer-Aware HiSparse Hot Buffers

本仓库研究稀疏注意力解码时，哪些请求、哪些层值得多留一些 KV 在 GPU，以及省下的换入时间能否改善整步性能。

## 实现任务

编码助手可在普通 CPU 环境完成 PR，从[开发和实机环境](docs/implementation/agent-environment.md)、[配置与逐项消融](docs/implementation/ablation-contract.md)、[参考代码和测试](docs/implementation/reference-code.md)开始。三个实现规格为[逐层驱逐](docs/implementation/layer-eviction.md)、[全层共享 LRU](docs/implementation/shared-lru.md)和[两者联用](docs/implementation/combined.md)。[逐层动态卸载](docs/implementation/dynamic-offload.md)（issue #11）在预注册的卸载名单之间运行期切换，与共享 LRU 无关。[按压力逐层弹性卸载](docs/implementation/elastic-layer-offload.md)（issue #13）按显存和排队压力逐组卸载、回填层，分 steady 与 smooth 两版，支持 TPOT SLO 和 PD decode 节点。[主 KV 重叠与模型适配](docs/implementation/model-adapters.md)给出各模型的路径和独立开关。

## 11 月 24 日实测更新

MiniMax 原生 HiSparse 在 A100 TP8 上完成 8–16K/B86、16–32K/B48、32–64K/B44 三档 CUDA Graph 诊断，三档均观察到高 miss 层的换入和完整 decoder 层用时更长。DeepSeek‑V4‑Flash 用较短请求冻结层组后，在 32 条独立长请求上复现了层间 miss 差异。附整步耗时拟合对照、误差和 SLO 反解条件。[实测汇总与模型](A100_HISPARSE_GRAPH_AND_SLO_20261124.md)

## 11 月 23 日实测更新

8 × A100 上，MiniMax 原生 HiSparse 在 B32、16–32 Ki token 输入下完成了 CUDA Graph 的逐层 Nsight 采集。事前分组的高 miss 层与低 miss 层，按每步取八卡最慢换入核、再跨八步取中位数的口径，分别为 **1,065.88** 和 **278.26 µs/层**。同 rank 计时显示，换入耗时差到 attention merge 结束仍保留 **99.65%**。固定层策略的 TPOT 与吞吐仍待配对实测。 [实测口径与逐层图](NATIVE_GRAPH_20261123.md)

同一台 A100 上的独立留出 eager B53 测量覆盖 60 层和全部 8 个 rank。按每层最慢 decoder rank 对齐，高低 miss 组的换入事件差为 **693.86 µs/层**，完整 decoder 层差为 **285.77 µs/层**。 [B53 逐层图与分段计时](NATIVE_EAGER_B53_20261123.md)

## 11 月 22 日实验更新

| 平台与模型 | 观测 | 报告 |
|---|---|---|
| 8 × A100，MiniMax，原生 HiSparse | 16–32 Ki 输入、实际 B32 的校准窗口中，历史高 miss 组比低组每层多用 595 µs 换入。完整 decoder 层的中位差缩至 194 µs，八个测量步中出现过反向差异。 | [逐层耗时与 335 个独立留出来源的压力基线](REPORT_20261122.md) |
| 8 × L20，MiniMax | 同一组 16 Ki、B17 请求三轮测得单层换入差为 202–212 µs。报告用原生 CUDA Graph 核计时估算固定 40 或 50 层的选层收益。 | [L20 实测与估算假设](L20_20261120.md) |
| A100，Qwen3 + Quest、GLM-5.2 | Qwen3 长输入的逐头回放有可重复的层间差异。GLM 的共享选集也有差异，但固定总容量后，学习名单在部分设置下输给均分。 | [跨模型结果与 DeepSeek-V4-Flash 的验证安排](CROSS_MODEL_20261122.md) |

固定低 miss 层相对随机层的实际 decode 加速仍待同显存、同并发的 CUDA Graph 对照。此前的请求类型、容量回放和拆分换入实验见 [11 月 12 日报告](REPORT_20261112.md)。

## 研究问题

在输入长度和运行条件相近时，聊天、工具调用型 agent 与长文档请求是否有可重复的跨步 KV 复用差异？同一请求中，不同层或物理 KV 源增加一页 GPU hot buffer 的边际收益是否不同？实验固定模型和稀疏 Top-K 选择规则，分别考察驻留容量对 miss 的影响，以及换入路径对真实服务延迟和吞吐的影响。

## 分配策略

- `request_type` 是拟议的可选 API 字段。调用方已知类型时可将其作为容量先验，缺失或未知时回退到统一定额。是否值得使用类型信息，以训练后冻结的同预算留出对照检验，并与只使用输入长度的策略比较。
- 容量按层或实际物理 KV 源设定。共享索引的层仍可能存储独立 KV payload。多个层读取同一份物理 KV 时只计一次，按真实页和字节核算，并核对 TP 各 rank 的映射。
- 分配价值取决于增减一页能避免多少换入字节及关键路径延迟，也需考虑查找、重映射和暖机成本，并保留当前 Top-K 所需的最小容量。当前的静态容量表以训练来源的模拟 miss 为拟合目标，选定后冻结。

## 11 月 12 日实验

截至 2026-11-12，SGLang、MiniMax M3.1 Flash 与 A100 上已采集 150 条真实轨迹，三类各 50 条。144 条自然 EOS，6 条到达配置容量边界后停止。原生 miss 的类型差异受输入、生成长度及实际上下文分布影响。

在相同的六层逻辑槽预算下，共用逐层表在冻结留出中将模拟 miss 次数减少 **9.41%**。该结果来自六个采样层、两个采样 rank 的同轨迹 LRU 回放。按类型分别拟合的额外改善为 0.0188 个百分点，点态配对 95% 区间为 [−0.0447, 0.0809] 个百分点，跨过零。

报告分别列出固定统一活跃容量下的融合与拆分换入、真实缓存语义检查，以及留出集 C1/C2 的无插桩结果，并给出各自的分母和执行配置。

## 实现框架

本轮实测采用 SGLang。后续可选择 SGLang 或 vLLM，依据目标模型与硬件适配、容量和 KV 映射的修改成本、CUDA Graph 支持及同框架内的实测结果。项目独立于 VMM 显存管理。跨框架比较需要分别建立匹配基线。

## 后续工作

- 将容量曲线对应的 miss 减少换算为实测时间收益，进一步验证逐层分配能否改善延迟，并通过真实物理页分配检查显存和可接纳并发的变化。
- 扩展代码、文档处理与 rollout 等场景，比较调用方类型标签、轻量分类器与长度或近期 miss 基线。在线调整还需验证迁移成本、最小驻留容量和迟滞机制，目前未实现。

## 相关研究与实现

- SPIN 动态调整每个请求的 GPU KV 容量，并以支持局部性的替换策略减少跨设备换入。 [SPIN §5.3](https://arxiv.org/html/2604.26837v1#S5.SS3)
- Adaptive DHiSparse 保留固定宽度的 hot cache，并用池中可回收的页保存额外历史。 [Adaptive DHiSparse §3.1](https://arxiv.org/html/2610.13708v1#S3.SS1)
- SGLang 协调器包含 HiSparse 请求缓冲区的分配和换入路径。vLLM 提供本地 KV offload 的架构说明。以下是上游参考。 [SGLang HiSparse 协调器](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/hisparse_coordinator.py) · [vLLM HiSparse 设计](https://docs.vllm.ai/en/latest/design/hisparse/)
