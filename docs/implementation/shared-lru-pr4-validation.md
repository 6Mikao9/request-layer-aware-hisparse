# PR #4 的 MiniMax TP8 验证

北京时间 2026-11-25 01:00–09:00。本页记录待执行的实验和验收条件。合并后固定最终 SHA、SGLang 源码、镜像摘要、CUDA/NVCC、驱动、模型版本及输入清单摘要。复核八张 A100 的 UUID、空闲显存、利用率及进程归属，容器创建前再核对一次，运行中遇外部占用即释放自有容器。旧夜间 worker 的日期和阶段均需更新；本页没有安装或启动 worker。

## 正确性和准入

在仓库根目录运行。输出位置须为本轮独占的新路径。

```sh
python -B -m unittest runtime.test_shared_lru runtime.test_shared_lru_config runtime.test_shared_lru_device runtime.test_shared_lru_gate runtime.test_shared_lru_kernel runtime.test_shared_lru_launch runtime.test_shared_lru_plan runtime.test_shared_lru_replay
python -m runtime.cuda_shared_lru_correctness --allow-cuda-test --output /tmp/pr4-gate.json
python -m runtime.cuda_shared_lru_production_correctness --allow-cuda-test --all-planes --output /tmp/pr4-production-gate.json
```

第一项是专项 CPU 检查。补充生产几何夹具后，本机完整主套件运行 238 项、跳过 1 项（Windows 缺 Clang CUDA backend），237 项通过；独立 GLM 参考套件运行 5 项、跳过 1 项（缺固定 vLLM 源），其余通过。dry-run 通过 33 个 pin 和 17 个符号检查。实际镜像安装目录需再次核验并保留哈希及失败项。

后两项须在 GPU 准入后执行。小型 fixture 覆盖共享、隔离、FIFO、软保底及旁路等策略。新增生产几何入口完整分配 60 层、H4096、TopK2048、page128、head dim128，使用两个槽（一实一 padding），以当前共享策略核对全部选中 K/V、owner/order/occupancy、返回索引、计数和状态；`--all-planes` 在 eager 与 Graph 各追加全 60 层一遍校验，否则只发射首尾层。第二代请求改变 host KV 内容，使用原 Graph 回放，检查旧 payload 隔离。它不覆盖真实高 batch、完整模型或全 TP rank；这些仍需随后真实模型小批检查。保留 NVCC 编译、sm_80 加载和 Graph replay 收据。状态检查放在计时外，负面错误注入另做。缺生产几何、真实 kernel 或 Graph 任一收据时，保留失败证据并暂停候选性能测量。所有合成 fixture 仅验证正确性。

## 三臂配对

打包检查另发现 launcher 已引用、但公开仓库未包含 `runtime/compat/minimax_sm80.py`。现补入旧实验所用的原字节版本（SHA256 `207df72e8294982f5153a236043871f38a26b37338c543d9c8322fd56d0c2bf0`），没有改动投影算法；新增安装路径检查后，相关 launch/contracts 的 21 项 CPU 测试全部通过。创建 Linux 源码包时关闭 Git 的 CRLF 转换，保证包内文件与已提交 blob 字节一致，避免固定 SHA 校验在加载前失败。

复制 runtime/fixtures/shared_lru_plan_spec.example.json 到独立运行目录，将零值摘要换成实际训练及留出清单摘要，解析模型路径并据八 rank 实测容量选择共同可行 B。以下命令使用准备好的私有 spec 和新输出目录。

```sh
python -m runtime.shared_lru_plan --spec "$RUN_SPEC" --output-dir "$PLAN_DIR"
```

依次装配 stock HiSparse、isolated arena、shared arena。验证 worker 实际选择的实现、热槽和地址范围。三臂使用相同前缀及顺序、生成长度、热身、采样、Graph bucket、并发和逐卡 HBM 上限。模板已计入每卡 254,639,076 B 额外 metadata；以最紧 rank 的实测峰值扣减可用容量，避免重复计费。记录实际可租页、占用、Graph 工作区、host pinned 字节及峰值。共同 B 不可行时列出该点并降到共同可行点。按轮交错三臂并重复 stock。分别报告无插桩吞吐、稳态逐 token p50/p95/p99、实际 B 与成功请求数，注明 prefill 屏障和稳态起止 token。

## Nsight 诊断

每臂另跑同前缀诊断波。使用既有 e2e/adaptive_sparse/native_nsys_window.py 的采集窗及 e2e/scripts/analyze_native_nsys.py 的解析入口。先检查 profiler 版本和容器权限。接入 CUDA Graph replay 内计时或可映射的 kernel 边界，每步覆盖全部八 rank、60 层的完整 decoder、attention 和 resolve/swap；核对实际 B、关联标识和完整步数。每层按完整 decoder 最慢 rank 对齐同 rank 的子区间。保留原始 .nsys-rep 和缺失项，避免把 CPU 发射区间用作设备耗时。

从 Systems 轨迹取得各臂真实 kernel 名称，单独采集匹配 resolve/swap 的 .ncu-rep，核对 profiler 权限、采集进程和命中 launch 数。采集 kernel 时长、寄存器、占用率、DRAM 读写和吞吐，结合每请求候选槽与探测次数检查扫描全部共享槽的代价。隔离命中、驱逐和重载条件，核对重复 replay 的输入状态。NCU replay 及逐层插桩有单独的运行编号。干净吞吐取无插桩波。

三臂外保留既定 ON/OFF 三长度专项，按原配置与清单配对；记录 8–16、16–32、32–64 Ki 每桶的实际并发和 Graph 证据。窗口余额用来补缺失的配对或采集。到 09:00 收尾，按收据逐项报告已完成、未执行与失败原因。
