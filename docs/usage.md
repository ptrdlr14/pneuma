# 运行与接入

[返回 README](../README.md) · [协议与支持范围](protocol.md) · [ALOHA 评测](benchmarks.md)

以下命令均在仓库根目录执行。工具链与常用命令也提供 mise 任务：
`mise install` 装好 Rust 与 uv，`mise tasks ls` 列出全部任务，
`mise run build` / `mise run test` / `mise run check` 对应下方的构建与检查命令。

## 启动服务

```sh
cargo run -- serve --config=examples/so101.toml
cargo run -- --help
```

启动方式为「配置文件 + CLI 覆盖」，文件格式为 TOML。覆盖顺序为：
结构体默认值 < TOML 文件 < CLI；合并后校验，未知字段、非法数值与不支持的配置在启动时报错。
必须显式指定 `--config`，服务端不读取 `PI_*` 环境变量。

```sh
# 临时改频率和监听地址。
cargo run -- serve --config=examples/so101.toml \
  --action-fps=50 \
  --listen=tcp/127.0.0.1:7555

# ACT：先转换本地 LeRobot checkpoint；metal 为 Apple GPU，flex 为纯 Rust CPU。
cargo run --release -- convert --from=lerobot --to=pneuma \
  --input=/absolute/path/to/lerobot-checkpoint --output=outputs/act-native
cargo run --release -- serve --config=examples/aloha_sim.toml \
  --checkpoint=outputs/act-native \
  --device=metal \
  --action-steps=50
```

## 配置与命令行

`model`、`execution`、`zenoh` 字段分组及 `features`、`action_feature` 命名与 LeRobot 对齐。
Pneuma 扩展 `model.policy=zero|act|smolvla|pi05`；`model.repo_or_path` 当前只接受本地 checkpoint，
`model.device=auto|flex|metal|cuda|cuda:N` 对应 Burn 后端，三个后端始终编译进二进制。
`auto` 按 CUDA → Metal → … 顺序挑选，也接受 `BURN_DEVICE` 环境变量。
`cuda` 由后端自选卡号，`cuda:N` 指定第 N 号卡（同 PyTorch 写法，例如 `cuda:0`）。
卡号越界不会静默回退到 CPU：CubeCL 在首次使用该设备时报错。
无 CUDA 运行时的机器上，`auto` 会选到其它已编译后端。
ACT 与 SmolVLA 的预测步数与默认执行步数来自 checkpoint，`execution.n_action_steps` 可覆盖执行步数；
π₀.₅ 同样从 `chunk_size` 与 `n_action_steps` 读取预测和执行步数。
zero 要求显式声明 `execution.prediction_steps`。`execution.warmup_calls` 默认为 2。
[so101.toml](../examples/so101.toml) 为 zero 零动作协议冒烟，
[aloha_sim.toml](../examples/aloha_sim.toml) 为 14 维 ACT 仿真合约。
[smolvla_libero.toml](../examples/smolvla_libero.toml) 为 SmolVLA 合约模板，
导入时通过 `convert --vlm-path` 指定本地 VLM 资源，转换后资源归原生 checkpoint 所有。
[pi05_libero.toml](../examples/pi05_libero.toml) 为 π₀.₅ 合约模板，`--vlm-path` 指向 PaliGemma
分词器目录；权重缺归一化统计时转换会明确失败。准备要求见 [π₀.₅ 指南](pi05.md)。
路径相对进程工作目录解析，含空格的路径应给整个 CLI 参数加 shell 引号。

`zenoh.config_file` 可指定 JSON5 安全配置底座（TLS、证书、ACL），
`zenoh.mode`、`zenoh.listen_endpoints`、`zenoh.connect_endpoints` 显式覆盖底座拓扑，
包括空数组；multicast/gossip 侦察关闭，连通性只靠显式 endpoints。

CLI 使用常规长选项，不接受 `--model.device` 等点号参数。完整 schema、semantics、
部署名称和安全配置在 TOML 中修改。常用选项为 `--checkpoint`、`--device`、`--policy`、
`--action-fps`、`--action-steps`、`--prediction-steps`、`--warmup-calls` 和 `--zenoh-mode`。
`--listen`、`--connect` 可重复使用，出现时替换对应的整个地址列表；未提供则保留文件值。
`--clear-listen`、`--clear-connect` 清空对应列表，分别与 `--listen`、`--connect` 互斥。
例如将 peer 部署临时切到 client：

```sh
cargo run -- serve --config=examples/so101.toml \
  --zenoh-mode=client --clear-listen --connect=tcp/127.0.0.1:7447
```

CLI 只覆盖显式提供的值，不经 TOML 字面量解析；应用覆盖后统一校验。
`--help` 和 `--version` 返回 0，参数错误返回 2，文件加载或配置校验失败返回 1。

## 无硬件协议冒烟

需要 Python 3.12、uv，以及包含 `lerobot.remote_inference` 和 `lerobot.inference`
的 LeRobot 源码。当前协议对齐版本见[协议说明](protocol.md)。
安装路径中的 `<lerobot 仓>` 替换为该源码仓库的本地路径。

```sh
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -e "<lerobot 仓>[remote]"
```

在一个终端启动 zero 服务：

```sh
cargo run -- serve --config=examples/so101.toml
```

在另一个终端运行官方 Python 客户端冒烟：

```sh
.venv/bin/python scripts/integration_smoke.py
```

脚本验证全零动作、reset/Stale 和 close/reopen。zero 只用于协议验证。

## LeRobot rollout 接入

下面的命令展示 SO-101 客户端所需的参数；机器人端口、相机配置和服务端地址需按实际环境填写。
SO-101 示例服务使用 zero，返回全零动作，不提供操控策略。

```sh
lerobot-rollout \
  --inference.type=remote \
  --inference.deployment=manipulation \
  --inference.semantics=so101-degrees-gripper-percent-v1 \
  --inference.chunk_merge=append \
  --inference.endpoint=tcp/<服务端IP>:7447 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.cameras='{front: ...}' \
  --fps=30
```

`chunk_merge` 默认 `auto` 会解析为 aligned，当前服务端会显式拒绝，接入时必须传 `append`。
`--fps` 决定客户端 action_interval，必须等于服务端的 `execution.action_fps`。
跨机器接入时，服务端监听地址需要允许客户端连接；示例默认只监听本机回环地址。

常见失败：

| 现象 | 排查方向 |
|---|---|
| `BUSY` + `admission_blocker` | 已有会话占用实例 |
| `INCOMPATIBLE feature/order` | 机器人相机名、rename_map 与 TOML 特征合约不一致 |
| `TimeoutError ... discovery` | endpoint 不通或 deployment 名不符 |
