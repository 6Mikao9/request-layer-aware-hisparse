# 逐层动态卸载与显存管理

[Issue #11](https://github.com/6Mikao9/request-layer-aware-hisparse/issues/11) 要求服务运行期间在预注册的卸载名单之间切换，decode 继续使用只捕获一次的 CUDA Graph。本实现不依赖跨层共享 LRU。卸载层各自使用原生 hot buffer，驻留层保留完整主 KV，不使用 arena，也不跨层借槽。MiniMax-M3 的稀疏层和 DeepSeek-V4-Flash 的 C4 层共用一套适配接口。

目前 CPU 测试和对固定 SGLang `e7399a46` 的 dry-run 已通过。A100 上的 JIT 编译、路由核正确性、Graph 切换、TP8 收据、物理显存和服务性能都还没有运行，见文末 `pending_hardware`。vLLM Hybrid HiSparse 对照之后单独做。

## 配置

```json
{"schema": "hisparse-study-config-v1",
 "placement": {"mode": "fixed_layers", "offload_layers": [0, 1, 2, 3, 9, 11],
               "alternate_offload_layers": [[17, 19, 20, 21, 22, 23]]},
 "features": {"route_indirect": true, "dynamic_offload": true, "host_shadow": false}}
```

| 字段 | 含义 |
| --- | --- |
| `placement.offload_layers` | 名单 0，即启动时卸载的层集合 S。层号是模型的绝对层号，只能取可 host 备份的层，即 MiniMax 的稀疏注意力层或 DeepSeek-V4 的 C4 层。 |
| `placement.alternate_offload_layers` | 名单 1、2……，每份与名单 0 层数相同（同一个 K），互不重复。需要 `route_indirect`。 |
| `features.route_indirect` | 全部名单的并集 U 放进 slab，经 GPU 路由表寻址。关闭时是冻结 S 的直接路径。 |
| `features.dynamic_offload` | 允许运行期切换名单。需要 `route_indirect` 和至少一份备用名单。 |
| `features.host_shadow` | U 中驻留的层也持续写 host 历史，切换时省掉一次 D2H。需要 `dynamic_offload`。 |

三个开关默认关闭。`runtime.study_config` 拒绝不一致的组合，worker 的启动收据记录解析后的配置和能力。

## 两种布局

**冻结 S（`route_indirect` 关）。** HiSparse 设备池只为 S 建平面；其余可卸载层放进一个普通池（MiniMax 为 MHA 池，DeepSeek-V4 为 C4 池），按逻辑行保存完整主 KV。S 走原生换入核；驻留层的换入返回 None，消费者按 `req_to_token` 直接读。这是“冻结被驱逐层”的对照版本。

**路由间接（`route_indirect` 开）。** U 为全部名单的并集，K 为每份名单的层数。U 的主 KV 放在 K、V 各一块 slab 中（DeepSeek-V4 只有一块 K-only 的页 slab），行号是绝对行：

```
| K 个 hot 平面 | |U|−K 个 resident 平面 | |U| 个 tail 视图 |
hot 平面：原生设备池的行数，含 padding 页
resident 平面：逻辑行数加 padding 页
tail 视图：请求槽数向上取整到整页；第 slot 行存该请求在本层最新的一条记录
```

GPU 上有一张 `[|U|, 4]` int32 路由表，每行是（模式，平面首行，tail 视图首行，0），模式 0 为卸载，1 为驻留。U 之外的可卸载层留在直接驻留池里，与冻结臂相同。HiSparse 协调器的原生表和 host 历史覆盖 U 的全部层。切换只改写路由表、相关平面的数据和原生表，不改任何指针或 launch 参数，所以已捕获的 Graph 继续有效。

## decode 一步

1. 存储：新 token 写进请求在本层的 tail 行，即 `req_pool_indices + tail_view`。Graph 的 padding 行槽号为 0，写到 0 号 tail 行；请求池的 0 号槽是 padding 槽，不属于任何请求。
2. 回刷：协调器原来的 eager backup 换成 tail 回刷，仍在 decode backup stream 上执行，下一次 forward 等它的事件。上一步写进 tail 的记录复制到它的 home：驻留层的逻辑行，卸载层位置小于 H 时的 hot 槽，以及卸载层的 host 行（`host_shadow` 开时驻留层也写 host）。每步工作量固定，host 侧不同步，见下文“回刷”。
3. 换入：U 的每层调用一次 `rlh_dynamic_offload_route` 核。核由 `runtime/build_route_overlay.py` 从固定版本的 `kvcacheio/hisparse.cuh` 机械派生，产物摘要记录在 `runtime/PROVENANCE.json`。驻留模式为每个选中位置输出 `平面首行 + req_to_token[(p+1)·R−1] / R`（MiniMax R=1，DeepSeek-V4 R=4）后返回；卸载模式保留原生的 LRU 与 host 换入逻辑，只是最新记录从 tail 行读。
4. 读取：decode 期间 KV getter 返回整块 slab，换入核给出的是绝对行号。U 之外的驻留层不经过换入核，与冻结臂一样直接读。

## prefill、staging 与请求结束

`route_indirect` 要求 prefill eager。prefill 的存储和读取在 Python 层按当前路由选平面：卸载层经 HiSparse 映射写 hot 行，驻留层写逻辑行；读驻留平面时按逻辑行寻址（MiniMax 去掉稀疏 prefill 的 `loc_mapping`，DeepSeek-V4 跳过位置映射）。staging 按路由把卸载层备份到 host，`host_shadow` 开时也备份驻留层。

请求的设备 buffer 分配后，卸载层原生表里的位置改为 slab 绝对行，buffer 增长时新位置同样处理。请求结束前改回平面内行号，原生 `request_finished` 照常按页释放分配器。PD 分离和直连准入在 launcher 与 worker 绑定时都会被拒绝。

## 切换

```bash
curl -X POST "http://127.0.0.1:$PORT/set_internal_state" -H 'Content-Type: application/json' \
  -d '{"server_args": {"hisparse_offload_list": 1}}'
```

`hisparse_offload_list` 只在 `features.dynamic_offload` 开时注册，取值是名单序号。请单独发送这个键：固定版本先逐键 validate 再逐键 apply，切换在 validate 中完成，同一请求里排在后面的键被拒绝时，切换已经生效。

scheduler 在两次 batch 之间处理这条请求。每个 TP rank 运行 `dynamic_offload.SwitchProtocol`，在专用 Gloo 组上依次投票：

1. propose：各 rank 的 epoch、路由摘要和配置摘要必须一致，否则判定组失效；目标名单未注册或各 rank 不同时拒绝。
2. prepare：有 chunked prefill 或 staging 拷贝在途时拒绝。否则先 `cuda.synchronize`，计算在场请求的摘要，再把将要卸载的层的 home 历史从 resident 平面拷到 host（`host_shadow` 开时跳过）。任一 rank 失败，或各 rank 的请求摘要不同（rank skew），整组拒绝，路由不变。
3. apply：只从 host 重建。将要卸载的层把前 min(home, H, cap) 条记录写进 hot 平面，并重置原生 token、位置和 LRU 表；将要驻留的层从 host 填满全部 home 历史；最后写路由表并同步。任一 rank 失败时，所有 rank 执行逆计划，从 host 恢复源路由。恢复失败时抛出 `SwitchProtocolError`，异常从 scheduler 抛出，整个 worker 组必须停止。
4. commit：各 rank 切换 host 侧的 staging、回刷和 prefill 路由，再交换收据摘要；摘要不一致同样判定组失效。

进入协议的每次尝试，每个 rank 都在 server 日志里打印一行 `hisparse_dynamic_offload_switch` 收据。`status` 为 `committed`、`unchanged`、`refused` 或 `rolled_back`，并带进出层、拷贝行数、各阶段秒数、投票和错误信息。HTTP 返回 `updated=false` 表示拒绝或回滚。CPU 冒烟测试（单 rank、4 个可切换层的玩具几何）中的两条收据，省略了摘要和计时字段：

```json
{"event": "hisparse_dynamic_offload_switch", "status": "refused", "source": "0", "target": "1",
 "error": "switch to 1 refused: ['SwitchError: a chunked prefill is in flight']"}
{"event": "hisparse_dynamic_offload_switch", "status": "committed", "source": "0", "target": "1", "epoch": 1,
 "entering": [[1, 0, 0], [4, 1, 1]], "leaving": [[2, 0, 0], [3, 1, 1]], "rows": {"prepare": 36, "apply": 66}}
```

`entering` 每项是（层，原 resident 平面，新 hot 平面），`leaving` 每项是（层，原 hot 平面，新 resident 平面）。

## 对照臂与开关

| 臂 | 设置 | 用来比较 |
| --- | --- | --- |
| `full_resident` | 不开 HiSparse 的固定版本 server | 全量驻留 |
| `stock` | 原生 HiSparse，覆盖全部可卸载层 | 原生基线 |
| `frozen_a`、`frozen_b` | 名单 A 或 B，`route_indirect` 关 | 固定 S；`frozen_a` 是已实测的固定 T40 |
| `indirect` | 名单 A，`route_indirect` 开，不切换 | 相对 `frozen_a` 的路由路径代价 |
| `dynamic` | 再开 `dynamic_offload`，从名单 B 启动，预热后切到 A | 稳定态对 `frozen_a` 和 `indirect`；切换耗时单独报告 |
| `dynamic_shadow` | 再开 `host_shadow` | 相对 `dynamic`，切换省下的 D2H 与每步多出的 host 写入 |

后五个臂的配置在 `runtime/fixtures/dynamic_offload_arms/{minimax,dsv4}_*.json`。MiniMax 的名单 A 是 issue 3 实测过的 calibrated T40（0–19、34、36–48、50、51、55、57–59，`placement.calibration` 记录训练集与名单摘要），名单 B 是 issue 3 的随机对照 `random_t40_s13`，U 为 51 层，U 之外 9 层。DeepSeek-V4 的名单 A 是 11 月 24 日实测的低 miss C4 组（L4、L28、L30、L36、L42），名单 B 是高 miss 组（L14、L16、L24、L34、L40），U 为 10 层。动态臂最后停在名单 A，所以它的稳定态直接和固定 T40 比。

decode 可以是 eager，也可以是整步 CUDA Graph，launcher 和 worker 各检查一次。使用本适配的臂都要求单 stream decode，即 `--disable-overlap-schedule`、`--disable-radix-cache`，不开 TBO/SBO、投机解码和 PD 分离，PP=1。`route_indirect` 另外要求 prefill eager（`--disable-prefill-cuda-graph`），拒绝 `--enable-mixed-chunk`，路由核只为 sm_80 构建。issue 中“择机再加”的 transition overlap 开关没有实现。

以下是预期，尚未测量。卸载层运行与原生相同的换入逻辑，每次 launch 多读一行路由表；U 之外的驻留层与冻结臂走同一路径；U 中当前驻留的层每步多一次驻留模式的换入核，冻结臂对这些层直接读。`indirect` 对 `frozen_a` 的配对会量出这部分代价，所以 U 只应包含确实要切换的层。

## 已有 profiling 的复核

issue 要求先用已有数据复核换入、resident read、attention、collective、层间等待与关键 rank，再优化证据支持的热点。数据是 issue 里链接的四组 issue 3 结果。

- 换入：B8 那一对（32–64 Ki，TP8）里，瓶颈 rank 上 40 个卸载层的换入核合计，原生 5,451 µs，固定 T40 5,443 µs；改为驻留的 20 层从 4,473 µs 降到 0。换入期间几乎没有别的核在跑（暴露比例中位 0.999），整段都在关键路径上。
- resident read：attention reader 的 NCU 中，驻留直读比读 hot 槽每层多 1.02 µs（B8 为 15.36→16.38 µs，B32 为 37.38→38.40 µs），两种读法输出相同，对 FP32 的误差都约 5×10⁻⁴。B8 时间线里驻留层的 attention 加 merge 中位多 1.4 µs（13.7→15.1 µs）。
- attention 与层间等待：merge 到 merge 的周期合计，20 个驻留层从 11,810 µs 降到 6,216 µs，比去掉的换入多降约 1.1 ms；40 个卸载层从 20,249 µs 降到 19,721 µs。多降的部分是等待变少。
- collective 与关键 rank：每 rank 的 collective 中位从 6.0–6.6 ms 降到 4.2–4.8 ms，这段时间包含等最慢 rank。原生的瓶颈 rank 集中在 rank 4（128 步中 73 步），T40 分散到 rank 0（50 步）、rank 4（30 步）和 rank 6（20 步）。整步中位从 33.43 ms 降到 26.75 ms。
- Graph 之外：干净配对的 TPOT p50 比同臂 Graph 包络 p50 多 1.99 ms（原生）和 1.67 ms（T40），约占 TPOT 的 6%。两组数取自不同窗口，只能作估计。这段是 scheduler 在两步之间的 host 工作，非重叠调度下直接计入 TPOT。
- B32 固定工作量（16–32 Ki，显存比例 0.84）：TPOT p50 78.13→58.64 ms，吞吐提高 33.2%；逻辑 KV 容量 1,037,824 对 889,344 行，0.7828 的预算下固定臂没能准入 B32。
- GLM-5.2 的 vLLM Hybrid HiSparse：精确预取让 follower 层的 KV 在 attention 前约 5.73 ms 就绪，换入核本身用时相近（12.3 对 13.8 µs）；三条请求的输出不一致，没有端到端结论。MiniMax 的稀疏层和 DeepSeek-V4 的 C4 层各自计算 Top-K，没有能提前拿到选集的 follower 层，这种预取用不上。

由此得出三点。卸载层的换入核是剩下最大的单层开销，但它是各臂共用的原生逻辑，改它属于搬运工程优化，要加到所有臂上，也不算逐层卸载的收益，这次不动。驻留读多出的 1 µs 左右可以忽略。证据支持的热点在 Graph 之外：tail 回刷在 `map_last_loc_to_buffer` 里、forward 之前执行，正落在这段 host 时间里，而第一版回刷逐层做下标计算，还用布尔掩码挑出有 home 的记录，在 CUDA 上每次掩码选择都要等 GPU 返回个数，是一次 host 同步。

## 回刷

现在的回刷从设备上的路由表一次算出所有层的 tail 行和 home 行；没有 home 的记录（卸载层上超出 hot 槽的记录）把 tail 行拷回自己，因此不需要掩码。槽号和记录号合成一次 pinned、非阻塞的上传，C4 的指针表在绑定和切换时建好。host 拷贝保持固定版本 backup 的结构：MiniMax 逐层调用 `transfer_kv_per_layer`（固定版本的 MHA backup 也逐层），C4 一次调用 `transfer_cache_dsv4_mla`（固定版本的 C4 backup 也是一次）。下表是每步工作量，按 CPU 上的 aten 调用计数（不含视图），几何为合成的 B=48：

| 每步 | MiniMax U=50、K=40，旧 | 新 | C4 U=10、K=5，旧 | 新 |
| --- | ---: | ---: | ---: | ---: |
| 下标计算调用（CUDA 上各一次 launch） | 398 | 24 | 88 | 23 |
| 布尔掩码选择（CUDA 上各一次 host 同步） | 100 | 0 | 20 | 0 |
| 由 Python 数据建设备张量（上传） | 2，阻塞 | 1，pinned 非阻塞 | 7，阻塞 | 1，pinned 非阻塞 |
| host 拷贝 launch | 40 | 40 | 5 | 1 |

新回刷的下标调用数与层数无关，`test_flush_work_is_fixed_and_sync_free` 检查 5 层和 40 层下相同且没有掩码选择。GPU 上每步省下多少时间要等 A100 测量。

## DeepSeek-V4-Flash

- 只接管 C4 层。hook 在 `DeepSeekV4TokenToKVPool._make_kv_pool` 中替换 C4 HiSparse 池，并在 `_init_compressed_layer_mapping` 之后把各层的 `layer_mapping` 指向 slab 层的门面对象或直接驻留池。
- slab 采用 64 行一页的 V4 页布局，每行是 576 B 值加 8 B scale，页按 576 B 对齐为 37,440 B。不支持 trtllm uniform FP8 布局。
- `swap_in_selected_pages` 收到的是池内序号，层号由 `MQALayer.forward` 的上下文提供。U 之外的驻留 C4 层用 `req_to_token[rid, 4p+3] // 4` 直接取行，写进原生输出缓冲，padding 和无效位置为 −1。
- decode 中压缩器写入时，闭合一组的 token 写到请求的 tail 行，不闭合的写到 0 号 tail 行。分配设备 buffer 前，prompt 最后一条 C4 记录复制到各层的 tail。回刷发生在 (seq−1) % 4 == 0 的步，回刷记录 (seq−1)//4 − 1。
- 配方默认 TP4、`top_k` 512、`device_buffer_size` 1024。

## 显存账本

建池时按逻辑行给驻留层计费，所以冻结、间接和动态各臂在同一个 `mem_fraction_static` 下都容纳 K 个 hot 平面和 L−K 个 resident 平面，L 是本 stage 的可卸载层数：

- MiniMax：`DefaultPoolConfigurator._compute_cell_size` 每 token 加 (L−K)·(ratio−1)·row，row 是一层一个 token 的 K+V 字节数。
- DeepSeek-V4：`DSV4PoolConfigurator._get_bytes_per_full_token` 为每个驻留 C4 层每 token 补足 1/4 − 1/(4·shrink) 条记录的页字节。

tail、多出的 |U|−K 层原生表和路由表不在 cell size 里，占用 KV 预算之外的显存；host 历史也比冻结臂多 |U|−K 层。dry-run 按规划预设给出每 rank 的数值，只计 U 的层，U 之外的层在两种臂中相同：

| 每 rank | MiniMax-M3 TP8 | DeepSeek-V4 C4 |
| --- | ---: | ---: |
| 预设 | 396,544 设备 token，ratio 2，H 4096，129 槽，页 128，行 256 B | 131,072 设备 token，ratio 2，H 1024，129 槽，页 64，记录 584 B |
| U / K / U 外驻留层 | 51 / 40 / 9 | 10 / 5 / 11 |
| slab 行数 | 24,605,312 | 1,970,560 |
| hot 与 resident 平面 | 12,591,235,072 B | 1,151,654,400 B |
| tail | 6,684,672 B | 1,123,200 B |
| 原生表，动态 / 冻结 | 276,212,736 / 216,637,440 B | 13,870,080 / 6,935,040 B |
| 路由表 | 408 B | 80 B |
| 设备合计，动态 / 冻结 | 12,874,132,888 / 12,807,872,512 B | 1,166,647,760 / 1,158,589,440 B |
| pinned host，动态 / 冻结 | 20,712,456,192 / 16,245,063,680 B | 1,535,040,000 / 767,520,000 B |

按这组预设，动态臂每 rank 比冻结臂多用 66,260,376 B 设备显存（MiniMax，约 63.2 MiB）和 8,058,320 B（DeepSeek-V4，约 7.7 MiB），pinned host 分别多 4,467,392,512 B 和 767,520,000 B。预设只用于规划，worker 从真实池重新推导所有值。

每个 rank 绑定协调器时打印一行 `hisparse_dynamic_offload_bound` 收据，内容包括解析后的配置和能力、层划分、cell size 计费、逻辑 KV 容量（分配器的逻辑 token 数和每平面 hot 行数）、账本、实际分配的字节（slab 或 S 平面、直接驻留池、路由表、原生表、pinned host），以及绑定时的 CUDA allocator 状态。权重、Graph 工作区和 index KV 列在 `unavailable` 中，因为它们在各臂相同，或要到捕获后才存在。配方在每次运行中采样 `nvidia-smi` 并保存 `/server_info`，`benchmarks.dynamic_offload_report memory` 由此给出各卡峰值和服务容量。

## 正确性证据

CPU 测试（Python 3.12 + CPU PyTorch）：

- `runtime.test_dynamic_offload`，15 项，除两进程 Gloo 一项外不需要 torch：rank oracle 与冻结语义逐步对照；TP8 模拟下的提交、拒绝、拷贝失败与中断回滚、组失效；两进程 Gloo 上的同一协议。
- `runtime.test_dynamic_offload_device`，13 项：torch 版 `SlabState` 在 20 步、6 次切换的历史上读到的行，与纯模型及每份名单的冻结路径逐字节相同，`host_shadow` 开关两种都测；切换中重建失败后从 host 回滚；C4 页布局、C4 指针表和直接驻留 C4 行；回刷的固定工作量。设置 `SGLANG_SRC` 后，10 个臂对固定 checkout 的 dry-run 全部通过，重新派生的路由核与仓库文件逐字节相同。A100 门的驱动在 CPU 上以路由语句代替核、以 consumer 语句代替 Triton 核运行：history 共 37 个事件、2,214 次读取；生产几何下 MiniMax 比较 29,896 行和 16 次 attention 输出（对 FP32 的最大误差 4.7×10⁻⁴），DeepSeek-V4 比较 8,192 行。
- `runtime.test_dynamic_offload_install`，4 项，在仿造的固定版本 SGLang 模块上执行 `install`。MiniMax 和 DeepSeek-V4 各有一项动态臂测试，依次走 sizing、建池、层映射、协调器绑定、prefill 存取、staging、decode（含 host 换入、tail 回刷、C4 每 4 个 token 闭合一条记录、buffer 增长）、经 `/set_internal_state` 的拒绝和两次提交，以及请求结束与槽复用；每次读取检查拿到的（请求，层，token 或记录），C4 还检查 scale 字节，并检查 `request_finished` 释放的页正好是该请求的平面内页。两个模型各有一项冻结臂测试确认原生路径和层序号映射。DeepSeek-V4 这组测试发现并修正了一个缺陷：绑定收据按 `k_buffer` 统计字节，C4 池只有 `kv_buffer`，DeepSeek-V4 worker 会在绑定时报错。
- `benchmarks.test_nsys_layer_timeline`、`benchmarks.test_dynamic_offload_report`：Nsight 归约、配对区间和报告的计算，用合成的导出和输出目录。
- `runtime.test_shared_lru_config`、`runtime.test_shared_lru_launch`：配置规则和 launcher 执行规则。

A100 上的门禁尚未运行。`runtime.cuda_dynamic_offload_correctness` 的 history 分别以 eager 和每个（层，bucket）只捕获一次的 Graph 运行路由核，每个事件后与语句逐位比较输出、原生表、slab 和 host，`host_shadow` 两种都跑。production 在两种模型的生产几何下，切换前后都比较核与语句；MiniMax 还用固定版本的 decode consumer `flash_decode_with_gqa_share_sparse` 读路由出的 slab 行，输出须在 FP32 参考的容差内（atol 0.002、rtol 0.02，与 issue 3 的 reader probe 相同），驻留模式的层还须与冻结臂的直读结果逐位相同。DeepSeek-V4 没有做 FP32 对照：它在 sm_80 上的 C4 decode consumer 还没确认，门禁对 C4 只逐字节比较记录。

配方第 2 阶段在服务级覆盖 issue 列出的场景。每个臂和每种切换序列都应与 `stock` 逐 token 相同：B=1 顺序请求严格比较，B=3（Graph bucket 有 padding、不满）对照第二个 stock server 的运行间差异。动态 server（名单 0 为 B，名单 1 为 A）依次跑：不切换、B→A、A→B、连续三次切换之后，以及一次在请求 decode 期间生效的切换（被拒就重试），Graph 与 eager decode 各一个 server，另加 `dynamic_shadow`。每个动态 server 结束后统计第一次切换之后日志里的 Graph capture 行数，应为 0。`benchmarks.dynamic_offload_report correctness` 汇总这些结论，只记录不拦截。

## 运行

CPU，在仓库根目录：

```bash
python -B -m unittest runtime.test_dynamic_offload runtime.test_dynamic_offload_device \
  runtime.test_dynamic_offload_install runtime.test_shared_lru_config runtime.test_shared_lru_launch \
  benchmarks.test_nsys_layer_timeline benchmarks.test_dynamic_offload_report
python -B -m runtime.dynamic_offload_device dry-run --sglang-src "$SGLANG_SRC" --model minimax_m3 \
  --config runtime/fixtures/dynamic_offload_arms/minimax_dynamic.json
python -B -m runtime.dynamic_offload_device dry-run --sglang-src "$SGLANG_SRC" --model deepseek_v4 \
  --config runtime/fixtures/dynamic_offload_arms/dsv4_dynamic.json
python -B -m runtime.build_route_overlay --source-root "$SGLANG_SRC/python/sglang"
```

`SGLANG_SRC` 指向固定 commit 的 checkout。dry-run 不导入 torch 和 SGLang，输出摘要校验、符号与调用点检查、层划分、slab 几何、账本和路由核的模板参数。`build_route_overlay` 重新派生路由核，结果与仓库中的文件不同就报错。

A100：

```bash
MODEL_DIR=... WORKLOAD=fixed-ids.jsonl TTFT_SLO=... TPOT_SLO=... [CORRECTNESS_WORKLOAD=short.jsonl] \
  [MEM_FRACTION=0.84] bash benchmarks/run_dynamic_offload_a100.sh "$OUTPUT_DIR" minimax_m3    # 或 deepseek_v4
python -m benchmarks.dynamic_offload_report all --dir "$OUTPUT_DIR" --model minimax_m3 --output report.json
```

所有臂用同一个 `--mem-fraction-static`（MiniMax 默认 0.84，即 issue 3 B32 的预算；DeepSeek-V4 必须指定），并每 8 步写一行 decode 日志。各阶段使用独立进程：

1. 路由核的 JIT 构建收据和正确性门。
2. 服务级输出一致性，见“正确性证据”。`CORRECTNESS_WORKLOAD` 未设置时使用 `WORKLOAD`。
3. 七个臂各用新 runtime，AB/BA 交错重复三次，两个并发压力点（默认 16 和 48）。动态臂预热后切到名单 A，单独记录 HTTP 耗时和各 rank 的收据。每次运行保存 server 日志（含 decode 日志和收据）、`/server_info` 和 GPU 显存采样。
4. `stock`、`frozen_a`、`indirect` 和 `dynamic` 的全 rank Nsight Systems（追踪 Graph 节点），导出 SQLite 后按进程汇总 Graph 节点。
5. 路由核在生产几何下的 Nsight Compute。B8 数据里换入核是读路径上的热点，attention reader 已在 issue 3 测过 NCU。

`benchmarks.dynamic_offload_report` 汇总这些输出：
- 每次运行的 TTFT、TPOT p50/p95、输出 token/s、自然 EOS 与 SLO goodput。
- 实际 batch：decode 日志中 `#running-req` 等于客户端并发、且在 CUDA Graph 上的行数，至少 8 行（64 步）才算合格。
- 按比较对和压力点计算每个合格配对的相对变化、均值和 95% t 区间。比较对为 `frozen_a/stock`、`frozen_b/stock`、`frozen_a/full_resident`、`indirect/frozen_a`、`dynamic/frozen_a`、`dynamic/indirect` 和 `dynamic_shadow/dynamic`。
- 各卡峰值、服务容量、绑定收据，以及切换耗时与收据。
- `benchmarks.nsys_layer_timeline` 给出的每步瓶颈 rank、关键 rank 和每层的 swap / attention / collective / TP 等待 / 其他 / 空闲拆分。装有 matplotlib 时由 `benchmarks.plot_layer_timeline` 画出 `stock/frozen_a`、`frozen_a/indirect` 和 `frozen_a/dynamic` 的成对图。

这两个 Nsight 工具移植自 issue 3 的 PR #10。DeepSeek-V4 在 sm_80 上的 attention 与层末 kernel 名还没确认，要从第一份 trace 读出后用 `--attention-kernel`、`--layer-end-kernel` 和 `--nsys-layers` 传入。

## pending_hardware

启动收据的 `pending_hardware` 字段列出 `sm80_jit_build`、`route_kernel_bit_equality`、`cuda_graph_switch_without_recapture`、`tp8_switch_receipts`、`physical_memory_receipt`、`serving_tpot_throughput`、`nsys_all_rank_timelines` 和 `ncu_route_kernel`。服务级输出一致性、回刷优化在 GPU 上的效果、DeepSeek-V4 的全部 GPU 结果和 vLLM Hybrid HiSparse 对照也还没有数据。
