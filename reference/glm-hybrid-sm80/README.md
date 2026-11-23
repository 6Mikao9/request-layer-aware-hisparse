# GLM Hybrid 的 SM80 attention 参考移植

这份材料为固定分支 `neuralmagic/vllm@e8ef1e07bd2f174bebfe34c3a3e35e952931efb1` 提供可审核的 sparse MLA attention 适配参考。它帮助编码助手检查 SM80 选核、BF16 主 KV 与分组消费接口。完整服务还要适配该 checkpoint 的索引和 extension、加载真实权重并验收 CUDA Graph。请按 [GLM 组首与 follower 设计](../../docs/implementation/model-adapters.md) 完成这些阶段。

## CPU 上复现接口和补丁

```bash
python -B -m unittest discover -s reference/glm-hybrid-sm80 -p test_abi_probe.py
python -B reference/glm-hybrid-sm80/prepare_attention_port.py \
  --source-root "$VLLM_SRC" --output-root "$NEW_SM80_SOURCE"
```

`$VLLM_SRC` 指向固定 Git 源，`$NEW_SM80_SOURCE` 是一个新目录。脚本验证原树七个摘要，生成独立副本、补丁和收据，保留原目录。`hybrid_triton_sm80.py` 给出 backend 注册、attention 与 Hybrid 读路接口。`gpu_attention_gate.py --help` 给出模型无关的设备门槛，实际 A100 运行由维护者执行。完整实测前不可报告 Graph 或性能达标。

无 vLLM 源码也能运行四项纯 CPU ABI 测试。将 `VLLM_SRC` 导出指向固定源码后，再运行同一条测试命令可附带核对第五项原生数据流。

## 来源

- vLLM Hybrid 固定运行时及 API 在 [neuralmagic/vllm 固定版本](https://github.com/neuralmagic/vllm/tree/e8ef1e07bd2f174bebfe34c3a3e35e952931efb1)。
- `triton_mla_sparse_kernel.py` 保留来自 [vLLM PR #38476](https://github.com/vllm-project/vllm/pull/38476) 的版权头，附 [Apache-2.0 许可证](third-party/LICENSE)。固定 kernel 字节 SHA-256 在构建脚本中验证。
- [GLM 权重格式](https://huggingface.co/cyankiwi/GLM-5.2-AWQ-INT4/blob/main/config.json)和[项目既有公开跨模型报告](../../CROSS_MODEL_20261122.md)说明可比性和结果边界。
