# 配置、消融与性能验收合同

任何要归因为研究机制或工程优化的结果，都必须能把对应技术关掉。开、关两臂读相同的输入、使用相同权重和实现（该技术本身修改的部分除外），在相同计时边界下比较。全部关闭是原生基线。只有实现使用了开关且实际走到了对应分支，才有证据讨论效果。

## 启动配置的共同 ABI

新增项目 launcher 参数 `--hisparse-study-config PATH`。适配器负责消费它；上游未修改的二进制仍接收原有运行参数。JSON 为严格 UTF-8，拒绝重复键、未知值、NaN、冲突字段和不存在的物理层号。服务启动前解析，父进程把规范化配置与 SHA-256 交给每个 TP worker，worker 重验摘要、权重几何和版本后才能分配或捕获。某个 arm 因硬件或 ABI 不支持无法运行时，明确失败原因。

```json
{
  "schema": "hisparse-study-config-v1",
  "placement": {"mode":"native","selection":"fixed","offload_layers":[]},
  "hot_pool": {"mode":"isolated","scope":"request_rank"},
  "features": {},
  "diagnostics": {"mode":"none"}
}
```

| 字段 | 合同 |
| --- | --- |
| `placement.mode` | `native` 保持后端的原生模式。 `full_resident` 为真正的全驻留对照。 `fixed_layers` 使用显式的物理 KV 层集合，集合成员把历史放在 host，以热池服务，其余层完整 GPU 驻留。固定布局在预热和捕图前生效。 |
| `placement.offload_layers` | 物理 KV 层的绝对零基层号，不能重复，校验与实际模型的索引和存储映射一致。区分 selector 组成员、decoder 层和物理 KV 层。 |
| `placement.selection` | `fixed`、`random` 或 `calibrated` 均使用预先冻结的名单。随机组附受控 seed，校准组附训练来源和名单摘要。服务只消费名单。 |
| `hot_pool.mode` | `isolated` 保持各层独立配额或原生路径。 `shared` 从合格的 host-backed 层列表派生共享域。[池开关](shared-lru.md) |
| `hot_pool.scope` | 本阶段 `request_rank`，单个请求的一次生命周期内、同一 TP rank。行键包含层号，物理槽真实存放对应层的 K/V。 |
| `features` | JSON 对象。每项新增工程技术为独立的命名布尔开关，默认 `false`。原生预取维持 stock 设置，另外提供禁用或新实现的对照。 |
| `diagnostics.mode` | `none` 用于干净性能。 `counters` 用于选集、缓存和物理账本计数。 `nsys` 用于 GPU 核和依赖。注明应用事件与 Nsight 的真实来源。 |

无配置和全默认模式都应走原生路径。这个分支不安装实验 hook，不分配实验设备内存，不在逐层热路径创建计时事件或执行额外的布尔分支。将开关分派绑定在初始化和 Graph 构造，重复 replay 使用稳定地址。引擎自己的 `--enable-hisparse` 保持原生语义。新开关是否生效由启动收据和分支观测核对。

### 每个技术点的消融

| 机制或假设 | 开关或控制因子 | 关与开的比较 |
| --- | --- | --- |
| 整层卸载及驻留 | `placement.mode` | 原生页级 Hybrid 与固定层使用同模型、同实际 HBM 和 host 上限。MiniMax 的选层比较另做，避免把跨框架结果混成模块加速。 |
| 校准选择 | `placement.selection` 和冻结名单 | 同 T、同 H/预算、同工作量。至少 3 个预定随机 seed。校准只读取训练。保留高代价反向名单作机制检查。 |
| 驻留层直读 | `features.resident_direct` | 同固定布局和原始 Top-K，比较正确的驻留 resolver 与直达路径。记录省下的 kernel、等待和总时间。 |
| 共池寻址与借页 | `hot_pool.layout`、`hot_pool.mode` | 比较 stock、同寻址独立配额、等物理预算借页。统计实际可复用的 backing、空置页和元数据。 |
| LRU、保底和旁路 | `hot_pool.eviction`、`hot_pool.quota`、`hot_pool.admission` | 同实现及预算下比较 FIFO、独立静态分区、LRU 无保底、软保底。旁路单独测试，计入临时区、实际缺失与搬运。 |
| 精确预取 | `features.disable_native_prefetch` 或独立新预取开关 | 同布局、同原始选集。原生可用预取保留在强基线，解释等待和消费差异。 |
| 每项工程优化 | 独立命名，例如 `zero_miss_fast_path`、`fuse_resolve_update`、`coalesced_gather`、`overlap_copy_consume` | 先在同一算法上逐项关与开，再在相同工程水平上比较算法。以优化过的原生实现作为强对照。 |
| 压力与 rank 负担 | workload 和硬件条件作为独立实验轴 | 长度和并发分档配对，记录各 rank 及整步设备包络。用实际 PCIe/NUMA 等待诊断何时影响关键路径。 |

有依赖关系的开关可以明确拒绝非法组合，在支持的最小公共实现上做消融。例如保持同一 arena 寻址再切换借页，保持正确同步再切换重叠。禁止以非法映射或缺失同步的 kernel 充当更弱的对照。组合包仍必须能关闭任一顶层机制，并记录实际执行路径。

## 重复运行与判断

1. 预先冻结模型、量化/投影、KV dtype、框架与镜像摘要、TP/DP/PP、选择、输入、到达/采样种子、实际 batch 与长度、Graph 档、预取状态、记录窗口、训练和验证划分、硬件频率和互联条件。每对只改变待验证开关，留出结果不能回流选层或调参。
2. 各 arm 使用新 runtime。prefill、建池、布局和捕图完成后同等预热。用同一批输入、可校验的真实输出前缀和同一 Graph 档配对连续满批稳态 decode。容量改动先证明物理收支及配对准入。记录实际 B、replay、尾部和自然 EOS。
3. 无插桩轮作为实际收益主证据。AB/BA 交错执行。在至少两个预定压力档各收三对或更多独立配对，再按 workload 来源或独立轮次聚类给出 95% 区间。层、rank 和连续 step 通常不是独立样本。
4. 同一物理 HBM/host 预算下，默认性能目标为相对声明的最强基线至少约 5% 的稳态 decode 吞吐提升，区间支持正收益，并通过预注册的尾部 TPOT 容差和质量检查。固定实际 B 时同时报 TPOT 降幅。以容量提升为主时，在同一延迟 SLO 下补 goodput。未达目标标记为未通过性能验收，保留负结果及 OOM 或预算差异。
5. 按 rank 监控实际 HBM，记有效 payload、tail、元数据、host 镜像、索引、Graph 池和 scratch。账本包含 CUDA/NCCL 等运行时开销，标明已分配、可重新分配和峰值。若声称释放空间，实际交给另一层或请求，并验证第二个使用者读到正确内容。
6. 另开诊断轮观察原始选择、去重逻辑 miss、请求搬运字节、实测链路流量及 consumer 暴露等待。分别记 GPU stream 依赖、有效拷贝、NCCL 到达和完整 forward。每步先找完整 forward 的瓶颈 rank，再结合该 rank 的本层时间线解释阻塞。各层独立最大值不能加成整步时间。

## 结果文件与 PR 模板

建议生成 `run_config.json`（请求配置、实际版本/开关和摘要）、`admission.json`（设备、自动池容量、实际 B、物理字节账本）、`correctness.json`（选集、内容、stream、生命周期）、`metrics.json`（稳态性能）和独立的 `diagnostic_summary.csv`。PR 写明 schema、字段、单位和拒绝一轮的条件。未知项写 null 和原因。

| 指标 | 口径 |
| --- | --- |
| 稳态窗口 | 后端实际 step、实际 B、上下文和输出长度、预热规则、参与/失败/缺测数 |
| TPOT 和吞吐 | 每个请求取窗内相邻 token 间隔平均，报告 p50/p95（ms/token）。同窗有效交付 token 数除实际墙钟秒数。另报 SLO 与 goodput |
| 物理账本 | bytes per rank，主机共享数据物理去重。区分 allocated/reserved/live 和重新分配证明 |
| 回填与读路 | 历史 unique 选择、miss 行、tail/padding、有效 K/V 字节、核时间、consumer 暴露等待、完整层与整步（µs） |
| 证据绑定 | 对照 run ID/输入摘要、配置摘要、真实开关、重复数、配对差值与不确定性 |

在 PR 描述加入「claim → switch → off run → on run → matching → evidence」。每条 claim 连到具体测试、配置或验收命令。CPU 未执行的 CUDA/Nsight 阶段标 `pending_hardware`。维护者签发 GPU 收据后关闭性能检查项。
