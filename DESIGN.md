# DESIGN — 开发期快照（可推翻）

本文件记录 Pneuma 当前的边界与路线选择，供讨论先改文档、再改代码。
开发阶段一切可推翻：讨论改了边界，先改本文件。

## 边界

- **Pneuma（本仓，Rust）**：以 Burn 为模型执行基础，目标覆盖策略训练、推理部署及模型与 processor 的保存加载；remote-inference v1 是部署入口之一。当前已实现 ACT / SmolVLA 推理与远程服务，训练执行器尚未实现。
- **LeRobot（外部，Python）**：当前提供仿真环境、数据集、训练产物、rollout 客户端与数值参照；通过数据和产物适配接入，不限定 Pneuma 只能做推理。
- 当前边界：`policy::Policy` 表达模型预测，`processor::Pipeline` 组合数据转换，具体 `inference::PolicyRunner` 组合模型与两条 pipeline；`deployment` 绑定执行配置，`codec` 承担在线记录转换。库入口供本地 Rust 调用方复用。训练执行器与模型 loss 接口尚待实现。
- 一个部署 = 一套合约 profile；协议字段、semantics、类型名一处一名，不留第二套说法。

下图同时展示当前组合与训练扩展方向；pipeline 和部署快照保存加载已实现，训练执行器及训练恢复状态尚待实现。

```mermaid
flowchart TB
    client["LeRobot / 本地调用方"] --> server["协议与会话：解码、generation、调度"]
    server --> runner["PolicyRunner：推理组合与执行片段"]
    dataset["数据集适配"] --> trainer["训练执行器：采样、损失、优化与恢复"]
    runner --> pre["输入 Pipeline"]
    trainer --> pre
    pre --> policy["Policy：ACT / SmolVLA"]
    policy --> post["输出 Pipeline"]
    post --> runner
    policy -->|损失| trainer
    artifact["Checkpoint：bpk、配置、pipeline、统计与资源"] -.加载 / 保存.-> pre
    artifact -.加载 / 保存.-> policy
    artifact -.加载 / 保存.-> post
```

## 通用 processor pipeline

`src/processor/` 提供训练与部署共享的数据转换。ACT 的 `NormStats` 仅作为旧权重导入器，SmolVLA 的 processor JSON 转成同一套 `Step`；旧专用执行路径已删除。数值公式、处理顺序与 padding 行为遵循 checkpoint，后端整数 / 布尔存储通过显式元素转换读取。网络、MessagePack 与机器人环境结构留在适配层。

### 数据与执行

- 采用统一的 `Batch`：具名特征、动作目标、任务文本及有效性 mask；数值值域区分主机数组、Burn 浮点 / 整数 / 布尔 tensor，shape、dtype、layout 与语义由共享 schema 描述。训练显式携带 batch / time 维，单样本推理由入口补 batch 维，不按 rank 猜测图像或时间轴。
- 版本化 `Step` 枚举声明输入输出 schema，按顺序转换 `Batch`；`Pipeline` 在构造时推导并校验完整 schema，执行时校验动态维度及必需字段。模型接收处理后的 batch，返回模型坐标下的动作；输出 pipeline 还原 canonical 动作。训练目标的动作归一化由输入 pipeline 处理，loss 在模型坐标下计算。
- 训练与推理使用显式执行上下文，当前携带设备与 `Train/Eval` 模式。首批步骤均为确定性变换，未引入随机增强和 RNG API；未来随机步骤必须绑定可恢复 RNG，并显式记录模式分支。resize、normalize、tokenize 等共享步骤沿用同一份定义与统计。
- tensor 的 Affine / Linear / Crop / Device 步骤保留 Burn 设备与自动微分图，不经主机往返或隐式 detach；主机到设备转换由显式步骤完成。Host 在 Train 模式执行前拒绝，并拒绝自动微分 tensor 导出。RGB / resize / pad 当前只支持主机数据，传入 tensor 明确报错。统计估计仍待独立 fit 阶段实现，当前加载固定统计，不在请求期间更新。
- 首批步骤覆盖特征重命名、batch / layout 转换、RGB 数值转换、resize / pad、归一化 / 反归一化、任务换行、tokenize、特征 pad / crop 和设备迁移。只注册实现过的步骤，未知标识、版本、配置或不支持的组合加载时报错。

### 所有权与状态

模型权重、步骤定义和只读统计可共享；运行状态由训练执行器或推理 worker 独占，按 episode / session 隔离。输入输出 pipeline 成对构造，未来相对动作等转换通过显式调用上下文传递当前锚点，避免步骤间隐藏引用。逐次调用的状态与跨调用的历史分别管理。

`PolicyRunner` 负责输入 pipeline → 模型预测 → 输出 pipeline → 完整输出校验 → 执行片段截取。训练执行器负责输入 pipeline → 模型损失 → 反向传播与优化，评估时使用输出 pipeline。动作播放队列、generation 与协议回复不进入模型或 processor。

reset 与 predict 在拥有状态的 worker 上串行处理；server 可以先推进 generation 并丢弃旧结果，但旧推理实际退出前不重置它正在使用的 pipeline。当前无历史策略可继续使用逐请求上下文，支持有状态步骤时必须执行这一所有权约束。

### 保存、加载与互操作

原生 `Checkpoint` 是内部唯一部署产物：版本化 manifest、模型配置与 `model.bpk`、输入 / 输出 pipeline 的有序步骤配置、归一化统计及 tokenizer 等资源引用和内容摘要。`pneuma convert --from lerobot --to pneuma` 在边界导入 safetensors，`pneuma serve` 对模型策略只加载原生 checkpoint；反向 convert 导出 LeRobot 权重与配套资源。初版转换保留当前推理模块未使用的源参数，作为独立 bpk 附属权重参与导出，不宣称这些参数已实现训练。导出从当前原生模型参数生成，不返回原始权重文件冒充转换结果。默认运行参数为 F32，导入 BF16 时显式转换并记录；不承诺字节级往返。转换不包含优化器、调度器或训练恢复状态。

Checkpoint 包含步骤配置保存稳定类型标识、版本和参数；数值状态存 tensor 文件，不序列化 trait object。资源引用使用产物内相对路径，外部资源通过显式解析器绑定并校验摘要，不在加载时隐式联网。

`Checkpoint::save` 通过模型回调保存实际 Burn `ModuleRecord`，模型文件为 `model/model.bpk`；两条 pipeline 与配置随版本 2 manifest 发布。转换器保留源 processor、统计和未用参数 bpk 以支持权重导出，检查全量参数名、shape 与 dtype 变化；pipeline 修改后初版拒绝反向导出。服务不再识别旧版本 1 safetensors 产物或外部 checkpoint，临时目录校验后发布，拒绝覆盖已有产物，目标遵循单写者约定。未来训练恢复 checkpoint 额外保存优化器、训练进度、采样器和 RNG 等必要状态，并明确恢复粒度；部署产物不包含 episode cache。数据统计、模型参数与调用缓存是三类不同生命周期，不能合并为一个笼统的 processor state。

LeRobot checkpoint 由导入适配器翻译为同一组步骤与资源绑定，旧 ACT 内嵌归一化缓冲也在此转换；适配规则不进入运行时热路径。导出 LeRobot 仅支持能完整表达的步骤和模型，不能表达时明确失败，不宣称所有产物双向兼容。模型代码只保留网络计算与损失，不保留第二条专用预处理路径。

### 迁移与验证

先建立 schema / batch、pipeline 执行与步骤持久化，再迁移 ACT 和 SmolVLA，随后将具体 `PolicyRunner` 与协议适配分开接线。迁移时保持现有数值参照、canonical 动作语义、执行片段与会话调度行为；不保留旧 processor 作为长期兼容分支。

当前验证覆盖实际训练 / 推理模式分支、保存加载后等价、资源迁移与摘要检查，以及需要梯度路径的反向传播。未来新增随机步骤与跨调用状态时，分别补充固定 RNG 的增强复现与状态隔离 / reset。既有 ACT / SmolVLA 固定输入 parity 验证迁移数值，协议测试验证任务、generation 和失效结果隔离。训练循环、数据加载与优化器集成是后续实现，通用 pipeline 的完成不等同于训练能力已交付。

## 会话调度

`server.rs` 的 pump 独占会话状态、准入、控制回复和推理结果发布。同步 `PolicyRunner::predict` 在至多一个 blocking worker 中执行；reset/close/reopen 可以推进，但旧 worker 实际退出前仍占推理名额，不把 abort 当作同步推理取消。completion 回到 pump 后比对当前 session/generation，失效的成功和错误结果均丢弃。

回调入口使用容量 32 的有界通道和非阻塞发送。observation 在复制 payload 前取得全实例唯一名额，该名额跨排队、推理和 completion 处理存活，不另设等待观测队列。忙时最多保留一个额外 payload 用于发布关联的 Busy；此回复名额或入口已满时直接丢弃超量消息，不生成额外任务/overflow 事件。language 同样只有一个待处理 payload 名额；入口已满的 query 由客户端超时处理。控制不会等待 predict，有限的入口积压不会保留无限图像。

ACTION 的 `server_durations.queue` 从回调接收时刻计到 worker 内 predict 开始，包含调度等待；`worker` 仅计 predict 的实际执行时间，排除发布。慢策略回归通过进入/释放握手验证控制推进、过载和旧结果隔离，超时仅作为死锁保护。

## 运行时决策：Burn 单路径

1. **实现**：策略用 Burn 手写模块，结构体与字段名对齐 PyTorch 属性名（`ACTPolicy`、`resnet`、`encoder`…），使权重键无需批量重映射。不用 `burn-onnx`，不用 `ort`——lerobot checkpoint 本身就是 `model.safetensors` + `config.json`，ONNX 中介只会引入算子覆盖与动态 shape 两类无谓风险。
2. **版本与 features**：Cargo 版本要求为 `0.22.0-pre.3`，当前 lockfile 解析为 Burn `0.22.1`（0.22 与 0.21 API 断裂：`Tensor::<D>` 不再泛型于 Backend）。关闭 `burn`、`burn-store` 默认 features，分别显式启用 `std/flex/metal/autodiff` 和 `std/safetensors/memmap`，支持推理、pipeline 梯度与 safetensors 适配，未启用训练执行器、optimizer 和 PyTorch archive reader 依赖路径。MSRV 1.95，本机 nightly 1.101 满足。
3. **后端**：`model.device=auto|flex|metal` 选择设备；`auto` 使用 `Device::default()`（启用后端 GPU 优先于 CPU，遵循 Burn 的 `BURN_DEVICE` 默认设备机制），显式 `flex`/`metal` 调用对应构造器。编译期同时启 `flex`（纯 Rust CPU，数值可复现）与 `metal`（Apple GPU）。部署构建完成后按 `execution.warmup_calls` 通过完整 runner 做零观测预热：metal 首推理的内核 JIT（秒级）挪进启动期，兼作权重/算子的加载自检。dtype 固定 f32。
4. **权重**：convert 使用 `SafetensorsStore + PyTorchToBurnAdapter` 导入、`BurnToPyTorchAdapter` 导出并执行模型键名双向映射；内部保存与 serve 加载使用 Burn `ModuleRecord` 的 bpk，保留参数身份。键对不上时按键对照修正，不做名称猜测。
5. **推理形态**：首版只做 plain chunk——观测进、`[chunk_size, action_dim]` 出，配合客户端 `chunk_merge=append`。temporal ensembling、RTC/blending、aligned merge 不在首版，入口处显式拒绝（与骨架既有原则一致）。

## 合约 profile

部署合约写在 TOML 文件中，字段分组沿用 LeRobot（`model`、`execution`、`zenoh`、`features`、`action_feature`）。`--config server.toml` 指定文件，CLI 只提供常见启动调整：checkpoint、设备、策略、动作频率、执行步数、预热次数及网络拓扑；schema、semantics 和完整部署信息只在 TOML 中维护。so101 / aloha_sim 是 `examples/` 下的部署示例，不在源码中维护另一套 profile。

`cli.rs` 用 clap 子命令区分 `convert` 与 `serve`，serve 以 `ModelArgs`、`ExecutionArgs`、`NetworkArgs` 组织类型化参数，采用 `--device`、`--action-fps` 等长选项；仅将显式传入的值直接赋给 `ServerConfig` 字段，不经过 TOML 序列化和点号路径解析。字符串按 argv 原样接收；`--listen`、`--connect` 可重复指定地址，`--clear-listen`、`--clear-connect` 显式清空对应列表（与同一列表的地址参数互斥）。`ServerConfig::load` 只读取和反序列化 TOML；应用 CLI 覆盖后，由 `Server::new` 校验最终配置，再加载策略。`main` 通过 clap 处理帮助、版本和参数诊断，配置模块不退出进程。旧点号 CLI 参数移除，TOML 格式保持不变。

部署配置负责通用 schema、semantics、执行和网络约束；ACT 支持的固定 state/image/action 特征由 ACT 模块的纯校验函数维护，策略构建在读取 checkpoint 和预热之前校验，并在最终通告合约上复用。checkpoint 数值不变量在构造模块前校验，归一化缓冲在预热前校验；不合法输入返回启动错误。

`model.policy=echo|act|smolvla` 选择实现，`model.repo_or_path` 为本地原生 bpk checkpoint 目录；`model.device=auto|flex|metal` 显式选择 Burn 后端。ACT 与 SmolVLA 的预测/执行步数默认取 checkpoint，`execution.n_action_steps` 可限制执行片段；echo 的步数在示例中显式声明。配置中不支持的字段或取值在启动时拒绝。

| profile | state | 相机 | action | action_interval | semantics |
|---|---|---|---|---|---|
| `so101.toml`（echo 冒烟） | 6×f32 | front 480×640×3 u8 | 6×f32 | 1/30 | `so101-degrees-gripper-percent-v1` |
| `aloha_sim.toml`（仿真主线） | 14×f32 | top 480×640×3 u8 | 14×f32 | 1/50 | `aloha-sim-14d-v1` |

`aloha_sim` 的 `prediction_steps`/`execution_steps` 从 checkpoint `config.json` 的 `chunk_size`/`n_action_steps` 读取，不二次发明。图像 resize/归一化等预处理归 Burn 侧（对照 lerobot processor 管线实现），Python 端只发环境原图。

## 首个策略：官方 ACT（aloha 仿真）

权重取官方 `lerobot/act_aloha_sim_transfer_cube_human`（官方 namespace 唯一的仿真 ACT checkpoint；pusht 官方无 ACT）。移植面：ResNet18（conv/BN/残差/pool）+ Transformer encoder-decoder + 位置编码 + action/state 投影，`burn::nn` 构件齐备。

## SmolVLA

SmolVLA 沿用 `PolicyRunner` 与 Burn 单路径，直接加载 LeRobot 本地 checkpoint。新增 SmolVLM2 视觉/文本编码与动作专家，使用 flow matching 生成动作块。`OBSERVATION.body.task` 随请求传入策略，作为语言条件；`capabilities.language` 仍表示独立的语言生成能力，不因支持任务文本而开启。请求不在策略内保留任务或 KV cache，避免 reset/close 与旧 worker 之间共享可变状态；每次推理内部缓存视觉与文本前缀。

图像、状态与动作的 shape 来自 checkpoint 的特征配置，部署相机按 checkpoint 顺序匹配。tokenizer 和 VLM 配置作为本地模型资源加载，运行时不隐式联网。数值验证固定输入图像、任务 token、状态与初始噪声，对照官方 Python 的前缀表示、速度场及最终动作块；普通测试使用小尺寸模块覆盖注意力、位置编码、预处理和动作截取。

LIBERO 闭环由 `scripts/libero_benchmark.py` 驱动官方 `LiberoEnv` 和 `LiberoProcessorStep`，环境/控制归 LeRobot，Pneuma 只做远程动作块推理。原 SmolVLA LIBERO 配置的 state6/stats8 冲突通过独立评测配置副本显式修正为官方 state8，两路真实相机不补伪第三路；保留原权重与逐项修改记录。默认 spatial 十任务、每任务五局，逐局原子落盘，Python CPU 与 Metal 分别采样噪声，报告初始状态配对及描述性成功率；不将其视为固定噪声数值对齐或标准全量榜单。详见 `docs/libero-benchmark.md`。

## 里程碑

| # | 交付 | 通过判据 |
|---|---|---|
| M1 | `aloha_sim` profile + so101 回归 | `integration_smoke.py` 验证 so101 echo 生命周期与全零动作；ALOHA ACT 由 checkpoint parity 和仿真 harness 验证；`cargo test` 绿 |
| M2 | Burn ACT 移植 + safetensors 加载 | `scripts/dump_act_reference.py` 生成参照产物，ignored 测试 `act_parity` 对齐 Burn `run` 与 Python `predict_action_chunk`+Unnormalize（f32 容差 ~1e-4；实测 max abs diff 3e-7）；`act_bisect_stages` 按 backbone/encoder 分段定位 |
| M3 | 仿真 harness `scripts/sim_rollout.py` | aloha env ↔ 官方 `RemoteClient` 闭环（connect→admit→infer→step）；rollout CLI 是 robot-only、eval 不接 remote，此 harness 即「无硬件全链路」正式替身；记录 roundtrip 与任务成功率 |
| M4 | GPU 后端 | `BURN_DEVICE=metal` 下 parity 与 M3 重跑通过，给出延迟对比（实测 parity 2.7e-7；worker med 317→48ms，6.6×；成功率 metal 3/5 vs flex 4/5，5-ep 小样本，混沌敏感） |

## ALOHA 配对评测

`scripts/sim_rollout.py --backend=all` 对同一 checkpoint 运行 Python ACT（CPU）、Rust Flex、Rust Metal；首轮固定种子 0–49。三方共享环境、图像尺寸、归一化缓冲、预测与执行步数；每局上限显式设为 LeRobot AlohaEnv 的 400 步，记录初始物理状态和图像摘要（渲染可能非确定性）。执行 plain chunk，不做 temporal ensembling、裁剪或插值。Python 使用官方 `ACTPolicy.predict_action_chunk`，权重和归一化加载方式与数值对齐脚本相同。环境按步推进，不按真实时间调度，因此成功率评测不衡量控制截止时间违约。

每局记录种子、初始观测摘要、成功与否、步数、逐请求耗时及服务端 worker 耗时；报告成功率 Wilson 95% 区间、与 Python 基线逐种子差异和配对 bootstrap 95% 区间。Python 本地调用耗时包含前后处理，Rust 请求耗时包含协议和传输；两者分别标注，不将差值当成纯传输开销。结果写入 `outputs/`，记录 checkpoint 内容哈希、服务端二进制哈希、代码版本和仿真依赖版本，逐局原子保存；续跑要求配置和版本相同。服务端由脚本启动并持有进程句柄，只清理自己启动的进程。

该轮为描述性基线，尚未约定允许的成功率偏差，不自动宣称通过非劣性验收。

依赖安装走 lerobot extras：`.[remote,aloha]`（`gym-aloha` 由 pyproject 钉版；本机装不上则按既有备选降级 pusht + 社区权重）。每个里程碑收口都重跑 M1 判据，防真机 profile 回归。
