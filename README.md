# Pneuma

**基于 Rust 与 [Burn](https://github.com/tracel-ai/burn) 的机器人策略推理运行时。**

Pneuma 将策略模型加载、推理与远程服务放在 Rust 中，通过
[LeRobot](https://github.com/huggingface/lerobot) 的 remote-inference 协议接入仿真与机器人客户端。
LeRobot 提供训练、数据和仿真生态，Pneuma 负责接收观测并返回动作块。

Pneuma（πνεῦμα）意为“气息”：策略赋予身体行动。

> 项目处于早期开发阶段，接口与协议仍可能变化。策略可选 `act`、`smolvla`、`pi05`，另有返回全零动作的
`zero` 后端用于协议验证。当前支持 plain action chunk，客户端需使用 `chunk_merge=append`；
aligned merge、RTC/blending 与独立语言生成尚未支持。SmolVLA 与 π₀.₅ 的任务条件来自 `OBSERVATION.body.task`。

## 快速开始

### 启动协议验证服务

```sh
pneuma serve --config=examples/so101.toml
```

服务默认监听 `tcp/127.0.0.1:7447`。此示例使用 `zero`，无需模型权重或硬件，
返回全零动作块，用于验证客户端连接和会话流程。保持服务运行后，可按
[接入指南](docs/usage.md#无硬件协议冒烟)运行 Python 客户端冒烟。

### 启动 ACT 推理

准备本地 `lerobot/act_aloha_sim_transfer_cube_human` checkpoint 目录，包含
`config.json` 与 `model.safetensors`，将下方路径替换为实际目录：

```sh
pneuma convert --from=lerobot --to=pneuma \
  --input=/absolute/path/to/lerobot-checkpoint --output=outputs/act-native

pneuma serve --config=examples/aloha_sim.toml \
  --checkpoint=outputs/act-native \
  --device=flex
```

Apple GPU 可将 `--device=flex` 改为 `--device=metal`。
此命令启动服务；完整 ALOHA 闭环运行方法见[仿真评测指南](docs/benchmarks.md)。

配置由 TOML 定义，CLI 可覆盖常用启动选项。查看全部命令行参数：

```sh
pneuma --help
```

## 文档

| 文档 | 内容 |
|---|---|
| [运行与接入](docs/usage.md) | 配置、CLI 覆盖、Python 客户端冒烟与 LeRobot rollout 接入 |
| [协议与支持范围](docs/protocol.md) | 协议对齐版本、消息流、通道、能力限制与实现索引 |
| [ALOHA 配对评测](docs/benchmarks.md) | 依赖、运行命令、统计口径与已有评测记录 |
| [SmolVLA](docs/smolvla.md) | 本地模型资源、任务条件、支持范围与数值对齐 |
| [π₀.₅](docs/pi05.md) | ModelScope 权重、状态离散化、输入契约与验证状态 |
| [cosmos3_edge 评估](docs/cosmos3.md) | 未实现；权重结构、差异与推进顺序 |
| [LIBERO benchmark](docs/libero-benchmark.md) | 官方环境闭环、Python/Metal 对比、数据契约与结果口径 |
| [通用 pipeline 与产物](docs/pipeline.md) | 训练/推理数据转换、梯度、保存加载与部署导出 |
| [策略全景与进度](docs/policies.md) | LeRobot 策略清单、依赖关系与 Pneuma 已实现范围 |
| [设计快照](DESIGN.md) | 架构边界、运行时选择与开发路线 |
| [配置示例](examples/) | SO-101 zero、ALOHA ACT 与 SmolVLA LIBERO 部署合约 |
