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
2. 回刷：协调器原来的 eager backup 换成 tail 回刷，仍在 decode backup stream 上执行，下一次 forward 等它的事件。上一步写进 tail 的记录复制到它的 home：驻留层的逻辑行，卸载层位置小于 H 时的 hot 槽，以及卸载层的 host 行（`host_shadow` 开时驻留层也写 host）。
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
| `frozen_a`、`frozen_b` | 名单 A 或 B，`route_indirect` 关 | 固定 S |
| `indirect` | `route_indirect` 开，不切换 | 相对 `frozen_a` 的路由路径代价 |
| `dynamic` | 再开 `dynamic_offload`，预热后从 A 切到 B | 稳定态对 `frozen_b`；切换耗时单独报告 |
| `dynamic_shadow` | 再开 `host_shadow` | 相对 `dynamic`，切换省下的 D2H 与每步多出的 host 写入 |

后五个臂的配置在 `runtime/fixtures/dynamic_offload_arms/{minimax,dsv4}_*.json`。MiniMax 的名单为 40 层，U 为 50 层；DeepSeek-V4 为 5 个 C4 层，U 为 10 层。这些名单用于接线和配对，选层方法不属于本 issue。

decode 可以是 eager，也可以是整步 CUDA Graph，launcher 和 worker 各检查一次。使用本适配的臂都要求单 stream decode，即 `--disable-overlap-schedule`、`--disable-radix-cache`，不开 TBO/SBO、投机解码和 PD 分离，PP=1。`route_indirect` 另外要求 prefill eager（`--disable-prefill-cuda-graph`），拒绝 `--enable-mixed-chunk`，路由核只为 sm_80 构建。issue 中“择机再加”的 transition overlap 开关没有实现。

以下是预期，尚未测量。卸载层运行与原生相同的换入逻辑，每次 launch 多读一行路由表；U 之外的驻留层与冻结臂走同一路径；U 中当前驻留的层每步多一次驻留模式的换入核，冻结臂对这些层直接读。`indirect` 对 `frozen_a` 的配对会量出这部分代价，所以 U 只应包含确实要切换的层。

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
| U / K / U 外驻留层 | 50 / 40 / 10 | 10 / 5 / 11 |
| slab 行数 | 23,811,840 | 1,970,560 |
| hot 与 resident 平面 | 12,185,108,480 B | 1,151,654,400 B |
| tail | 6,553,600 B | 1,123,200 B |
| 原生表，动态 / 冻结 | 270,796,800 / 216,637,440 B | 13,870,080 / 6,935,040 B |
| 路由表 | 400 B | 80 B |
| 设备合计，动态 / 冻结 | 12,462,459,280 / 12,401,745,920 B | 1,166,647,760 / 1,158,589,440 B |
| pinned host，动态 / 冻结 | 20,306,329,600 / 16,245,063,680 B | 1,535,040,000 / 767,520,000 B |

按这组预设，动态臂每 rank 比冻结臂多用 60,713,360 B 设备显存（MiniMax，约 57.9 MiB）和 8,058,320 B（DeepSeek-V4，约 7.7 MiB），pinned host 分别多 4,061,265,920 B 和 767,520,000 B。预设只用于规划，worker 从真实池重新推导所有值。

每个 rank 绑定协调器时打印一行 `hisparse_dynamic_offload_bound` 收据，内容包括解析后的配置和能力、层划分、cell size 计费、账本、实际分配的字节（slab 或 S 平面、直接驻留池、路由表、原生表、pinned host），以及绑定时的 CUDA allocator 状态。权重、Graph 工作区和 index KV 列在 `unavailable` 中，因为它们在各臂相同，或要到捕获后才存在。各卡峰值来自配方在每次运行中对 `nvidia-smi` 的采样（`*.gpu-memory.csv`）。

## 正确性证据

CPU 测试（Python 3.12 + CPU PyTorch）：

- `runtime.test_dynamic_offload`，15 项，除两进程 Gloo 一项外不需要 torch：rank oracle 与冻结语义逐步对照；TP8 模拟下的提交、拒绝、拷贝失败与中断回滚、组失效；两进程 Gloo 上的同一协议。
- `runtime.test_dynamic_offload_device`，12 项：torch 版 `SlabState` 在 20 步、6 次切换的历史上读到的行，与纯模型及每份名单的冻结路径逐字节相同，`host_shadow` 开关两种都测；切换中重建失败后从 host 回滚；C4 页布局和直接驻留 C4 行。设置 `SGLANG_SRC` 后，10 个臂对固定 checkout 的 dry-run 全部通过，重新派生的路由核与仓库文件逐字节相同。A100 门的驱动在 CPU 上以路由语句代替核运行：history 共 37 个事件、2,214 次读取；生产几何下 MiniMax 比较 29,896 行，DeepSeek-V4 比较 8,192 行。
- `runtime.test_dynamic_offload_install`，2 项：在仿造的固定版本 SGLang 模块上执行 `install`，依次走 sizing、建池、协调器绑定、prefill 存取、staging、decode（含 host 换入、tail 回刷和 buffer 增长）、经 `/set_internal_state` 的拒绝、不变和两次提交，以及请求结束与槽复用。每次读取检查拿到的（请求，层，token），并检查 `request_finished` 释放的页正好是该请求的平面内页。冻结臂一项确认原生路径和层序号映射。
- `runtime.test_shared_lru_config`、`runtime.test_shared_lru_launch`：配置规则和 launcher 执行规则。

A100 上的门禁尚未运行。`runtime.cuda_dynamic_offload_correctness` 的 history 分别以 eager 和每个（层，bucket）只捕获一次的 Graph 运行路由核，每个事件后与语句逐位比较输出、原生表、slab 和 host，`host_shadow` 两种都跑；production 在两种模型的生产几何下，切换前后都比较核与语句。配方第 2 阶段在 B=1 下比较切换后的动态臂（Graph 与 eager decode）和 `frozen_b` 的输出 token 哈希。

issue 中“与 FP32 reference 在容差内一致”这一项没有单独实现。路由只改变 KV 的寻址，不改注意力计算；上面的检查比较寻址得到的 BF16 字节和最终输出的 token，FP32 容差检查留给实验室的质量检查。

## 运行

CPU，在仓库根目录：

```bash
python -B -m unittest runtime.test_dynamic_offload runtime.test_dynamic_offload_device \
  runtime.test_dynamic_offload_install runtime.test_shared_lru_config runtime.test_shared_lru_launch
python -B -m runtime.dynamic_offload_device dry-run --sglang-src "$SGLANG_SRC" --model minimax_m3 \
  --config runtime/fixtures/dynamic_offload_arms/minimax_dynamic.json
python -B -m runtime.dynamic_offload_device dry-run --sglang-src "$SGLANG_SRC" --model deepseek_v4 \
  --config runtime/fixtures/dynamic_offload_arms/dsv4_dynamic.json
python -B -m runtime.build_route_overlay --source-root "$SGLANG_SRC/python/sglang"
```

`SGLANG_SRC` 指向固定 commit 的 checkout。dry-run 不导入 torch 和 SGLang，输出摘要校验、符号与调用点检查、层划分、slab 几何、账本和路由核的模板参数。`build_route_overlay` 重新派生路由核，结果与仓库中的文件不同就报错。

A100：

```bash
MODEL_DIR=... WORKLOAD=fixed-ids.jsonl TTFT_SLO=... TPOT_SLO=... \
  bash benchmarks/run_dynamic_offload_a100.sh "$OUTPUT_DIR" minimax_m3    # 或 deepseek_v4
```

各阶段使用独立进程：

1. 路由核的 JIT 构建收据和正确性门。
2. B=1 输出一致性：`frozen_b`、切换后的 `dynamic`（Graph）和切换后的 `dynamic`（eager decode），结论写入 `correctness-verdict.json`，只记录不拦截。`CORRECTNESS_WORKLOAD` 未设置时使用 `WORKLOAD`。
3. 七个臂各用新 runtime，AB/BA 交错重复三次，两个并发压力点（默认 16 和 48）。动态臂预热后切换，单独记录 HTTP 耗时和各 rank 的收据。每次运行保存 server 日志、启动收据和 GPU 显存采样。
4. `stock`、`frozen_b`、`indirect` 和 `dynamic` 的全 rank Nsight Systems（追踪 Graph 节点），导出 SQLite 后用 `benchmarks.nsys_graph_summary` 按进程汇总 Graph 节点。
5. 路由核在生产几何下的 Nsight Compute。

TTFT、TPOT p50/p95、吞吐、自然 EOS 和 SLO 来自 `benchmarks.sglang_stream_client` 的输出，配对差和 95% 区间按[配置与逐项消融](ablation-contract.md)的规则计算。

## pending_hardware

启动收据的 `pending_hardware` 字段列出 `sm80_jit_build`、`route_kernel_bit_equality`、`cuda_graph_switch_without_recapture`、`tp8_switch_receipts`、`physical_memory_receipt`、`serving_tpot_throughput`、`nsys_all_rank_timelines` 和 `ncu_route_kernel`。B=1 服务输出一致性、DeepSeek-V4 的全部 GPU 结果和 vLLM Hybrid HiSparse 对照也还没有数据。
