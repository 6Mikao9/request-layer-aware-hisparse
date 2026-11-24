# 共享 LRU 的元数据占用与优化顺序

2026-11-24，针对 [PR #4](https://github.com/6Mikao9/request-layer-aware-hisparse/pull/4) 的源码审查。以下字节数是张量形状计算与 CPU 分配收据核对，尚无该 CUDA 核的真实性能结果。

## 当前占用

每个 TP rank 独立管理请求热池。设请求槽数为 S，层数 L，每层历史热槽 H，总共享行数 R=L×H。这里 S=128、L=60、H=4096，因此 R=245760。S 是预留请求槽数，不是该步实际 batch。

| 表 | 用途 | 形状与类型 | 每卡占用 |
| --- | --- | --- | ---: |
| `row_owner` | 每个物理热行当前属于哪个层，以及请求槽 generation | S×R，int32 | 120 MiB |
| `order` | LRU 顺序中的物理行号 | S×R，int32 | 120 MiB |
| 模板、occupancy、generation、floors、status | 初始化、配额与生命周期 | 多张小表 | 约 2.843 MiB |
| 原生 `lru_slots` | arena 路径不用于换入，但原生请求结束仍会重置它 | L×S×H，int16 | 60 MiB |

新增 arena 表准确为 **254,639,076 B/rank（242.843 MiB）**，不含诊断计数器。原生 LRU 的 60 MiB 不在这个新增数中；原生 token/loc 表、主 KV、索引和 Graph workspace 也不能漏算。见 [字节账本](../../runtime/shared_lru.py) 与 [实际张量收据](../../runtime/shared_lru_device.py)。本次修正将此前遗漏的四张持久初始化模板纳入账本，新增了账本与实际 CPU 张量逐项求和的回归检查。

## 候选优化与独立开关

以下均为待实现和验证的设计；当前配置解析器尚未注册这些开关，不应把表内名称直接传给服务。

| 优先级 / 建议开关 | 可减少的显存，每卡 | 实现边界与代价 |
| --- | ---: | --- |
| 1 / `elide_native_lru` | 60 MiB | arena 构造时跳过原生 LRU 分配，保留原生请求结束的等待和清理，只跳过 LRU 重置，并拒绝 arena 下进入依赖此表的原生换入入口。构造后删除张量不能证明降低初始化峰值或 allocator reserved。 |
| 2 / `compact_owner` | 90 MiB | owner 改成 uint8，0–63 表示层、254/255 表示临时占用/空闲。当前 int32 还保存逐行 generation；必须证明旧请求写入已经完成、新请求全量重置且 Graph replay/异常回滚不留旧 owner。保留 slot generation。若仍保留每行 uint16 generation，只能省 30 MiB。 |
| 3 / `compact_order` | 30 MiB | 真正的共享域使用 low16+high8 精确表示行号。R=245760，至少需要 18 位，不能直接改成 uint16。两部分采用独立单元写入，避免多个线程修改同一个压缩字。额外寻址和访存指令需测量。 |

前两项合计可减少 **150 MiB/rank**；三项合计可减少 **180 MiB/rank**，均是相对于当前包含原生冗余表的总分配量，不能直接从“新增 240 MiB”中减去 180。上述数字只计 live 表，模板变化另列。隔离域内不超过 65536 行时，`order` 可存 uint16 域内偏移，省 60 MiB；这个结论不能用于整个 60 层共享域。18 位紧密打包虽理论上更省，原地压缩时存在多线程对同一个字的读改写竞争，先不采用。

每项保留独立开关，默认维持本 PR 实现。先用固定选择序列验证返回地址、K/V 字节、LRU 顺序、占用数、generation、错误状态和计数分母；覆盖 eager、小型 Graph 与生产几何。性能比较包含开/关两臂、相同实际 batch 以及同显存预算下最大可行 batch。不能把减少显存自动等同于提升吞吐。

## 更大的风险是扫描成本

长序列路径每层扫描整个域查命中，随后再次扫描并重写 LRU 顺序。按 int32 实现，两个 `order+owner` 读取遍历和一个 `order` 写遍历为 20R 字节逻辑访问，即每个长请求每层约 **4.6875 MiB**。60 层、B32 且全部走长路径时为 **8.79 GiB/步**，另有 token 检查、淘汰搜索和 KV 搬运。短序列会走专用分支。这个数是源码访问量估算，L2 命中、编译优化和硬件事务会影响真实 HBM 流量。

因此先用 Nsight Systems 确定 resolver 是否在关键路径，再用 Nsight Compute 看实际 L2/DRAM 流量、带宽、寄存器、占用率与 barrier stall。若全池扫描占主导，压缩 dtype 仍可能不足，后续需要按 token 定位的目录或分组管理来减少扫描。目录自身显存和更新成本必须计入，并保持同一缓存策略作单因素对照。

固定上游依据：[原生 LRU 构造](https://github.com/sgl-project/sglang/blob/e7399a46f420d5903c3f2eddee74f0f9cfe68e11/python/sglang/srt/managers/hisparse_coordinator.py#L690-L696)、[请求结束清理](https://github.com/sgl-project/sglang/blob/e7399a46f420d5903c3f2eddee74f0f9cfe68e11/python/sglang/srt/managers/hisparse_coordinator.py#L1395-L1458)。[夜间验证方案](shared-lru-pr4-validation.md)保留当前实现作为首次实机基准。
