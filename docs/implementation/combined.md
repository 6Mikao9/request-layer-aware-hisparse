# 联用整层卸载和跨层热池

这是给编码助手的实现规格。先读[开发和实机环境](agent-environment.md)、[逐项消融合同](ablation-contract.md)和[模型适配及重叠路线](model-adapters.md)。本任务复用[逐层驱逐](layer-eviction.md)与[跨层共享 LRU](shared-lru.md)的接口。贡献者用 CPU 完成 PR，提供维护者随后可直接执行的真实 GPU 命令。

## 功能边界

给定初始化时冻结的物理 KV 层集合 S：

- S 外层保留完整 GPU 主 KV，原始 Top-K 只查对应的驻留行。
- S 内层在 host 有完整历史。每个请求的每个 rank 在兼容的 S 层之间共享一组 GPU 历史热槽，层号是键的一部分。每层保留原始选集、顺序、重复和 padding。
- 共享关/开使用相同 S 及相同物理总预算。S 为空相当于全驻留，只有一层时共享退化为独享，覆盖所有合格层时与全层共享 LRU 一致。初版在开始 prefill 前选布局。建池、布局、预热和 Graph 捕获完成后才进入测窗。
- 两层各有 h 槽时，若行型一致，可用 2h 槽替代 h+h。例如 h=4096，两层工作集分别 3072、5120，8192 槽可全部容纳。静态配额也可以做到。需要额外测两个工作集在生成阶段发生交换，以及一层持续扫描、污染另一层热数据的情况。

MiniMax TP8 的示例行宽为 512 B，每请求、每 rank 8192 行等于 4 MiB 历史 payload。若两层各有独立的 128 行尾页，另占 128 KiB。真实预算将常驻 KV、K/V 对齐、尾、正反映射、保留但不可用页、Graph 池与 scratch 一起统计。不同层行型的比较按可用物理 bytes，每组兼容形状单独成 arena。这些容量数字是算例。

## 公共接口与接线

使用项目启动器的 `--hisparse-study-config PATH`。下面层号只用于构造验证样例。

```json
{
  "schema": "hisparse-study-config-v1",
  "placement": {"mode":"fixed_layers","selection":"fixed","offload_layers":[5,15]},
  "hot_pool": {"mode":"shared","scope":"request_rank","layout":"arena","eviction":"lru","quota":"none","admission":"all"},
  "features": {},
  "diagnostics": {"mode":"none"}
}
```

同一配置分别切 `hot_pool.mode=isolated` 和 `placement.mode=native`。后者清除 S，按引擎的合格 host-backed 层列表接入共池。两项默认且 `features` 为空时走原生。改动必须连到实际初始化、alloc、writeback、lookup、attention 消费和 finish。如果另两条 PR 未合入，先用公共接口在 CPU 完成衔接及生命周期测试，PR 写明依赖，最终提交实际 serving patch。

1. **初始化**。每个 worker 验证权重和版本，创建冻结 placement 快照。绝对层号、索引组、行型、容量、arena 域和摘要在 TP rank 间一致。生产步骤使用原生 layer→KV 映射；不要把绝对物理层和组索引混用。
2. **路由**。从原 allocator 拿真实 request pool ID 和 allocation generation。返回给 attention 的 descriptor 包含绝对层、有效长度、K/V 基址或兼容 view、原生顺序的物理行、就绪 dependency、generation 和读 pin。S 外直达完整 KV，任何 S 内借页均不能影响它。
3. **安全共享**。在同一层的完整有序选集可用后锁住所有本步命中，再挑自由槽或合法 victim。victim 来自同请求、同 rank、S 内。等待前一写入、host 镜像和 attention reader 完成再撤销旧映射。K/V 都到位后才发布新 tag，消费结束释放 pin。重复逻辑位置共享有效物理数据，但保留原生有序输出。padding 无副作用。
4. **尾部和预取**。分别保护最新实际写入、上一 token 的 backup、预取在途行、已选未消费行。前驱至下一冻结层的逻辑选择未准备好时不能精确预取。GLM 组首保留驻留、follower 可优先选入 S，使用其共享索引，限制同一时刻在途热槽和 stream event。逐层通路和同层分块 merge 各有独立开关。
5. **Graph**。捕获前预分配稳定地址、device-side tag、临时区、有效 lane 和其他 meta。支持实际 B 小于档位时 padding。禁止新增逐层 `.item()`/`.cpu()`、host 轮询、同步、设备临时分配或 TP collective。无法承载有效 K 和在途保护的资源配置明确报错或用已验证的安全降级 arm，收据注明。
6. **清理**。取消、完成、错误回滚和复用逐个验证。若申请物理页实现了回收，用实际第二个 owner 重用并核对值。所有静态、共享与原生 arm 对齐峰值账本。

## 无 GPU 交付测试

| 夹具 | 必须观察到的现象 |
| --- | --- |
| 两层、两请求、两个 rank，payload 每层/代次不同 | 请求 A 的层 0 可以复用 A 的层 1 的已归还物理槽。A 的另一 rank、请求 B 和常驻层映射不动。新读者从真实槽读回正确 K 和 V。 |
| h=2 的 h+h 与 2h | 同历史字节预算。A 循环 3 个 token，B 重读 1 个，预热后共池可全命中。第二例 B 扫描连续新 token，触发跨层污染。 |
| 配额、单层和边界 | 软保底独立开关，用相同总驻留观察保底效果。S 为空、单层及全合格层方向的结果清楚。 |
| 乱序 copy、reader 延迟、重用 | 在途回填/attention/backup/预取不可覆盖，完整 K/V 后才可见，代次复用不串读。 |
| 路由、短转长和尾部 | 新旧路径产生相同选集与 padding/out-of-range 结果。只有 S 内持有 host backing，短序列容量按真实状态处理。 |
| 用户开关 | 无配置与全关同 stock。两项分别开，联用。非法层号、核心布局、K 下限、未知字段、rank 摘要变化等明确报错。 |

## 固定物理预算的稳态比较

| Arm | 主 KV 布局 | 热池 | 能说明什么 |
| --- | --- | --- | --- |
| A | 原生全层 HiSparse | 原生均分 | 同后端主基线 |
| B | 原生全层 HiSparse | 同寻址独立配额，另测共享 | 新寻址开销、全层共享效应 |
| C | 固定 S，随机/训练校准两类 | 等量独立 `T*h` | 单独逐层驱逐和共享关 |
| D | 与 C 相同 S | 训练集冻结的强静态配额 | 共享池的强对照，允许保持均分 |
| E | 与 C 相同 S | 共享 `T*h` 或等物理 bytes | E–C、E–D 分离共享收益；同 T 的随机/校准与共享开关做交互 |

先在同一 MiniMax/SGLang 检查 A–E。与 GLM Hybrid 的比较在同一 GLM、同一原生 Hybrid 适配上做，保留其已有精确预取。只有两个机制都接好了 GLM 路径，才加 GLM 组合验证。改变 T 时常驻层完整 KV 也纳入同一物理 HBM 上限，由剩余空间反推可用热预算。真实 B、graph replay、host budget 都要对齐。省略 `--max-total-tokens`，根据运行时实际定容选择高压工作点并保留 OOM/准入证据。

首轮从固定布局建立并预热完成开始。同窗记录 TPOT p50/p95、稳态交付 tokens/s、整步与瓶颈 rank，以及暴露等待。强基线包含性能可用的优化，所有新优化分别关/开。目标是相同实际 B 与物理预算下，独立留出上相对最强比较臂约 5% 的稳态 decode 提升。报告是否达标、尾延迟、正确性和冲击压力。模拟、单核或 CPU 成功各有自己的结果。

## 发 PR

提供固定上游版本、应用补丁/模块、CPU oracle 和测试、GPU 运行脚本、每臂配置或生成器、输出 schema、需求到实际证据映射。PR 的每个技术组合按 [消融合同](ablation-contract.md)贴「claim → switch → off → on → evidence」。GPU、真实模型、NSYS 和性能阶段由维护者执行，未跑标 `pending_hardware`。保留上游引用和许可证。只有公开合成夹具与匿名汇总随 PR 提交。
