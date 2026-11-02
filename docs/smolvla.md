# SmolVLA

[返回 README](../README.md) · [接入指南](usage.md) · [协议说明](protocol.md)

Pneuma 用 Burn 原生执行 SmolVLA，运行时无需 Python。推理流程包含 SigLIP 视觉塔、
SmolVLM2 文本前缀、Llama 动作专家及 Euler flow matching，直接读取本地 safetensors。
每次观测的 `task` 都参与分词和推理，任务更新无需重启服务。推理内部复用前缀 KV，
下一次观测重新编码；reset/close 不会留下旧任务缓存。

## 模型资源与启动

checkpoint 目录需包含：

- `config.json` 和完整的 `model.safetensors`，LoRA 需预先合并。
- `policy_preprocessor.json`、`policy_postprocessor.json`。
- processor 中 `state_file` 指向的归一化统计 safetensors 文件。

另准备官方 SmolVLM2-500M-Video-Instruct 的本地资源目录，包含
`config.json`、`tokenizer.json` 和 `tokenizer_config.json`。VLM 权重已包含在完整策略 checkpoint，
无需在此目录重复保存。服务不会隐式联网下载。

```sh
cargo run --release -- convert --from=lerobot --to=pneuma \
  --input=/absolute/path/to/consistent/lerobot-checkpoint \
  --output=outputs/smolvla-native \
  --vlm-path=/absolute/path/to/SmolVLM2-500M-Video-Instruct

cargo run --release -- serve --config=examples/smolvla_libero.toml \
  --checkpoint=outputs/smolvla-native --device=metal
```

`model.policy=smolvla` 选择策略；VLM 路径只在 convert 导入时指定，省略时使用源 checkpoint 下的 `vlm/`。
serve 从原生 checkpoint 加载配置、`model.bpk` 和 pipeline。`--device=flex` 使用 CPU。预测步数与默认执行步数来自
`chunk_size`、`n_action_steps`；`--action-steps` 可缩短执行片段。客户端使用
`chunk_merge=append` 并在观测中发送 `task` 字符串。

[配置示例](../examples/smolvla_libero.toml) 是 8 维状态、三路 256×256 RGB、7 维动作的合约模板。
状态定义、相机名与动作语义必须由部署方确认，不能仅凭形状匹配连接机器人。
在线 RGB 为 HWC uint8，checkpoint 中为 CHW；相机编码顺序遵循 checkpoint JSON。

已有 `lerobot/smolvla_libero` 本地副本存在 `config.json` 状态 6 维、统计量 8 维的冲突。
服务会在启动时拒绝，不会裁剪统计或自动改变状态契约。该副本的 processor 还包含非空
`rename_map`，需要在准备部署模型时明确统一名称。数值参照脚本为此建立独立的 8 维测试副本，
记录全部修改，不改源文件；该测试副本用于数值验证，不能替代真实机器人数据契约的确认。

## 支持范围

当前支持 SmolVLM2-500M 的标准视觉/语言架构，VLM 与专家层数相同，
`self_attn` 或交替 `cross_attn`，`use_cache=true`、单次观测、float32 推理。
BF16/F32 checkpoint 参数在加载时统一为 F32，模块所需参数缺失或加载错误会阻止启动。
图像预处理结果必须为 512×512；双线性缩放采用 half-pixel 坐标、无 antialias，
从左侧和顶部补边，最后映射到 [-1,1]。状态在归一化之后补零到 `max_state_dim`。

processor 按保存的步骤和统计加载，支持 IDENTITY、MEAN_STD、MIN_MAX、QUANTILES、QUANTILE10。
分词遵循本地 tokenizer 的截断方向及 processor 的右侧 padding，在缺少末尾换行时追加一次换行。
非空 rename_map、未知步骤、统计维度冲突和不支持的 dtype 在启动时报错。

暂不支持缺失相机填充、图像特殊 token、不同 VLM、减少专家层数、RTC/blending、Aloha 适配、
delta joint action、aligned merge 或独立语言生成。`capabilities.language=false` 指后者，
不影响 `OBSERVATION.body.task` 的语言条件推理。

## 数值参照

需要具备官方 LeRobot SmolVLA 依赖的 Python 环境。脚本固定原始图像、状态、任务和完整初始噪声，
严格加载官方策略的全部参数，以 CPU/F32 运行完整 10 步推理，输出视觉层、connector、
逐层前缀 KV、首步速度场和最终动作。当前脚本针对上述 LIBERO 资源，要求 state8 统计、
chunk_size=50、max_action_dim=32、num_steps=10。

```sh
/path/to/python -I scripts/dump_smolvla_reference.py \
  --checkpoint=/absolute/path/to/smolvla_libero \
  --vlm-path=/absolute/path/to/SmolVLM2-500M-Video-Instruct \
  --lerobot-src=/absolute/path/to/lerobot/src \
  --output=outputs/smolvla-parity

PNEUMA_SMOLVLA_VLM=/absolute/path/to/SmolVLM2-500M-Video-Instruct \
  cargo test --release smolvla_parity -- --ignored --nocapture

BURN_DEVICE=metal \
  PNEUMA_SMOLVLA_VLM=/absolute/path/to/SmolVLM2-500M-Video-Instruct \
  cargo test --release smolvla_parity -- --ignored --nocapture
```

`manifest.json` 保存原始配置摘要、测试副本修改、依赖与上游版本及每个张量的 SHA256。
`PNEUMA_SMOLVLA_PARITY` 可指定其他参照目录。测试还验证经 `PolicyRunner` 返回的执行片段长度和有限性。
普通 `cargo test` 不依赖本机大权重，覆盖小型视觉塔、pixel shuffle、RoPE、因果注意力、
分词、归一化、图像缩放及真实协议中的任务传递。

2026-10-29 本机对上述独立测试副本完成 Flex 与 Metal 数值对齐，两者均通过。
测试逐层检查前缀 KV，完整 10 步动作及反归一化动作的最大绝对误差上限为 2e-3；
该结果不代表原始 6/8 维冲突 checkpoint 可直接部署，也不衡量 LIBERO 任务成功率。

真实 LIBERO 任务闭环的运行方法与统计口径见 [LIBERO benchmark](libero-benchmark.md)。

## 官方客户端冒烟

先按上述命令启动服务，再用含 `lerobot[remote]` 依赖的 Python 环境运行：

```sh
/path/to/python -I scripts/smolvla_smoke.py \
  --lerobot-src=/absolute/path/to/lerobot/src
```

脚本连接真实 Zenoh 服务，发送两条不同任务，检查动作形状、dtype、有限性，执行 reset 和 close。
它不根据两次随机动作不同来断言语言条件有效；任务传递由确定性协议测试验证，模型数值由固定噪声参照验证。
本机 Metal 冒烟返回两次 `[50,7]` 动作块，请求耗时约 1.36 秒和 1.87 秒；这是单次联调记录，
不代表实时控制截止时间或任务成功率评测。
