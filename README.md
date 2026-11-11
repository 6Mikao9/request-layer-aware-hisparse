# Request- and Layer-Aware HiSparse Hot Buffers

This research proposal asks whether allocating GPU hot-buffer capacity jointly by request type and physical KV source reduces decode transfer or latency under a fixed memory budget. The model and its exact sparse Top-K selections stay fixed. Only the residency of selected KV entries changes.

## 研究问题

相同输入长度的聊天、工具调用型 agent 和长文档任务，是否有可重复的跨步 KV 复用差异？在同类请求内，各层或共享 KV 源是否也有不同的容量收益曲线？若两种差异能在独立测试集上复现，就在相同的物理 GPU 显存预算下，比较联合分配与统一定额。

## 分配策略

- 以请求到达时的类型作为容量先验，优先接收调用方提供的可选 API 字段 `request_type`。缺失或未知时使用统一定额。后续评估轻量分类器，特征限于分配时已经可用的信息。对阶段变化明显的请求，可进一步测试基于近期 miss 的容量反馈，并计入调整成本。
- 每类请求按层或真实物理 KV 源分配容量。共享索引的层可能仍有独立 KV payload。多个消费者若共享同一份物理 KV，只计一次。容量按真实页和字节核算，计入页对齐，并确保 TP 各 rank 对应的分配与映射一致。
- 根据增减一页能避免的 miss 字节、传输量及关键路径上暴露的延迟调整容量。高 miss 率本身无法说明扩容有收益。纳入预取重叠、重映射及暖机成本，保留执行当前 Top-K 所需的最小容量。

## 实验计划

1. 目标规模为约 200 个独立聊天来源、200 个独立 agent 来源和约 100 个 LongBench-v2 样本。使用完整输入，记录自然 EOS 结束的生成。按独立会话或任务来源分组划分训练和测试，同一来源的前缀、续写或中间 checkpoint 不跨集合。先冻结分类器、容量曲线拟合和策略选择，再运行测试。
2. 在类型间做长度匹配或分桶，对齐负载与并发条件，同时保留仅使用长度和负载的基线。固定模型、Top-K、物理 KV 表示与替换策略，扫描容量并按独立来源报告置信区间。
3. 在同一物理显存预算下比较统一容量、仅按类型、仅按层或物理源、类型与源联合分配，以及离线 oracle。轨迹回放测 miss 和搬运字节。单独跑无轨迹插桩的端到端实验，测 TPOT、TTFT、吞吐与分类型尾延迟，核查真实物理页占用，并验证 Top-K 与基线一致。

## 相关工作

- SPIN 的调度器根据运行请求的序列长度和资源需求调节每请求缓冲空间。本提案测试请求类型是否提供额外预测信息，并据此对物理 KV 源分别定额。 [SPIN §5.3](https://arxiv.org/html/2604.26837v1#S5.SS3)
- Adaptive DHiSparse 保留固定宽度的 hot cache，并弹性管理可回收的历史驻留页。本提案的控制量是按请求类型与物理源设定的 hot buffer 宽度。 [Adaptive DHiSparse](https://arxiv.org/html/2610.13708v1)
- SGLang 和 vLLM 提供统一的 hot buffer 容量配置。SGLang 的短请求路径可按长度少分配，vLLM 的历史驻留也可随显存压力变化。实验将区分这些机制和请求类型与物理源联合定额的效果。 [SGLang 协调器](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/hisparse_coordinator.py) · [vLLM HiSparse 设计](https://docs.vllm.ai/en/latest/design/hisparse/)

## 当前状态

六条试跑仅用于检查采集与分析管线。扩大样本验证正在进行。请求类型和层间差异、联合分配的效果以及端到端收益均待验证。
