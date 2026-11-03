# cosmos3_edge 接入评估

[返回 README](../README.md) · [策略全景](policies.md)

本文记录 `lerobot/cosmos3-edge` 的接入评估结论。**状态：未实现**。
Pneuma 当前不提供 `model.policy=cosmos3_edge`，`convert` 遇到该 checkpoint 会失败。

## 事实

- ModelScope 仓库：`lerobot/cosmos3-edge`，`type=cosmos3_edge`，权重 9.14 GB，BF16 为主。
- LeRobot 侧实现只在未合并分支 `origin/feat/add_cosmos3_pr`（`0e53307b9`，2026-11-02），
  不在 `main`：
  - `configuration_cosmos3.py` 604 行
  - `modeling_cosmos3.py` 2799 行
  - `processor_cosmos3.py` 739 行
- 权重分 6 组（共 1188 个张量）：

| 前缀 | 张量数 | 内容 |
|---|---:|---|
| `model.network.mot_backbone` | 979 | Nemotron-3-Dense-VL 文本塔 + 视觉塔 + 动作 DiT |
| `model.network.vae` | 196 | 视频 VAE，含 48 维隐变量统计 |
| `model.network.time_embedder` | 4 | 时间嵌入 MLP |
| `model.network.action2llm` / `llm2action` | 4 | 动作与语言宽度互转 |
| `model.network.llm2vae` / `vae2llm` | 4 | 隐变量与语言宽度互转 |
| `model.network.action_modality_embed` | 1 | 模态嵌入 |

## 与已实现策略的差异

1. **两塔联合注意力（MoT）**：文本塔 28 层、hidden 2048、8 个 KV 头、head_dim 128，
   `hidden_act=relu2`，mRoPE 分节 `[24,20,20]`，`joint_attn_implementation=two_way`。
   与 π₀.₅ 的 PaliGemma + Gemma 专家结构不同，不能复用 `gemma.rs`。
2. **视觉塔**：SigLIP2 风格，27 层、hidden 1152、patch 16、`spatial_merge_size=2`、
   patch merger 输出 2048。与 SmolVLA / π₀.₅ 的视觉塔参数均不相同。
3. **采样器**：`sampler=unipc`、`num_inference_steps=4`、`guidance_scale=3.0`、
   `sampling_shift=5.0`，另有独立动作时间分布。Pneuma 现有只有 Euler 积分。
4. **世界模型分支**：`mode=wam`，含 VAE 与视频去噪分支。纯动作推理是否必须加载 VAE
   尚未确认；若不加载，必须证明动作分支不依赖它。
5. **状态与动作宽度**：`max_action_dim=64`、`chunk_size=32`，动作经 `action2llm` 映射到
   语言宽度，再每步解码。

## 已确认的模块结构（2026-11-03）

配置解析已实现并用真实 `config.json` 验证（`src/policy/cosmos3/config.rs`）。下一步所需的结构：

**文本塔**（`mot_backbone.model.language_model`）：28 层，hidden 2048，16 头、8 个 KV 头、
head_dim 128，`relu2`，MLP 只有 `up_proj`/`down_proj`（无 gate），RoPE theta 1e8、
mRoPE 分节 `[24,20,20]`、`mrope_interleaved=true`。每层有两条并行流：

- 理解流：`input_layernorm`、`q/k/v/o_proj`、`mlp`、`post_attention_layernorm`。
- 生成流：同名的 `_gen` 版本，额外有 `q_norm_gen`、`k_norm_gen`、`k_norm_und_for_gen`
  （对 head_dim 做 RMSNorm）。
- 最终范数：`norm` 与 `norm_gen`。

**视觉塔**（`mot_backbone.model.visual`）：SigLIP2，27 层，hidden 1152，16 头，
patch 16、位置嵌入 256×1152。patch 嵌入是 **Linear(768→1152)**，不是卷积。
投影层 `projector`：`norm`(1152) → 每 4 个相邻 token 拼成 4608 → `linear_fc1`(4608→11520)
→ **精确 GELU** → `linear_fc2`(11520→2048)。token 按 2×2 块优先（block-major）排列。

**VAE**（`vae`）：Wan2.2，3D 卷积，z_dim 48，patch 2。推理只需 encoder（decoder 仅用于
`save_predicted_video`）。编码器 4 个 down block（dim_mult `[1,2,4,4]`），每 block 2 个
resnet 与可选 attention。

**采样与条件**：动作与视频隐变量拼成 `[video | actions]` 的生成流；uniPC 4 步、
CFG 3.0、shift 5.0；`action2llm`/`llm2action` 是 domain-aware 线性层
（`fc` 与 `bias` 两组权重，按 `domain_id` 选择）。

## 进度（2026-11-03）

| 部分 | 状态 | 验证 |
|---|---|---|
| 配置解析 | 完成 | 真实 `config.json` 解析通过 |
| SigLIP2 视觉塔 | 完成 | 与 HF 参照比较：最大 6.3e-4，均值 3.9e-6 |
| patch merger | 完成 | 与参照比较：最大 2.1e-5，均值 1.1e-6 |
| mRoPE（交错） | 完成 | 由文本主干对齐覆盖 |
| MoT 文本主干与联合注意力 | 完成 | 理解流 max 1.3e-4，生成流 max 1.9e-5 |
| Wan2.2 VAE 编码器 | 完成 | 与 diffusers 参照比较：隐变量最大 7.4e-5，均值 8.1e-6 |
| 生成流嵌入（视频隐变量、动作、时间步） | 完成 | 与官方方法比较：嵌入最大 1.9e-6，速度最大 2.9e-6 |
| uniPC 采样（4 步） | 完成 | 与 diffusers 逐步比较：4 步误差均为 0 |
| 联合位置索引 | 完成 | 理解流与生成流均与官方逐步一致（误差 0） |
| 整网接线（视觉 + 文本 + 生成流） | 完成 | 与官方主干比较：理解流 max 2.4e-4，生成流 max 5.9e-5 |
| 序列布局（条件帧与条件动作） | 完成 | 与官方 `build_sequence_plan` 一致 |
| CFG 3.0 组合与速度清零 | 已实现（待端到端验证） | — |
| 处理器与协议接入 | 未实现 | — |

视觉对齐覆盖 patch 打包顺序、块优先重排、线性 patch 嵌入、位置嵌入、27 层编码器与
精确 GELU 的合并层。参照由 `scripts/dump_cosmos3_vision.py` 生成，Rust 侧测试为
`policy::cosmos3::vision::reference::vision_matches_reference`。

VAE 只实现编码路径。三个与常见实现不同的点：

1. 推理只编码条件帧（单帧），官方 `_encode` 的分块缓存路径只在训练或多帧时使用。
2. 空间打包的通道顺序是 `(c, 宽度偏移, 高度偏移)`，不是常见的 `(c, 高, 宽)`。
3. `is_residual=true` 时每个下采样块带一个**无参数**的平均捷径：把
   `factor_t × factor_s × factor_s` 的窗口折进通道，再按组取均值后加到主路输出。

参照由 `scripts/dump_cosmos3_vae.py` 生成（需要 diffusers），Rust 侧测试为
`policy::cosmos3::vae::reference::vae_matches_reference`，逐层比较 11 个中间张量。

生成流嵌入覆盖：隐变量 2×2 patch、`vae2llm`/`llm2vae`、按具身域选择的
`action2llm`/`llm2action`、模态嵌入、正弦时间步与两层 MLP。参照由
`scripts/dump_cosmos3_gen.py` 生成（直接调用官方类的嵌入方法），Rust 侧测试为
`policy::cosmos3::gen_embed::reference::gen_embeddings_match_reference`。

uniPC 按 diffusers 的 `UniPCMultistepScheduler` 设置实现：`flow_prediction`、
`use_flow_sigmas`、`solver_order=2`、`solver_type=bh2`、`predict_x0`、`lower_order_final`。
sigma 序列先把 `1 - 1/num_train_timesteps` 落成 float32，再线性取值并平移。
参照由 `scripts/dump_cosmos3_sampler.py` 用固定人造速度场生成，Rust 侧测试为
`policy::cosmos3::sampler::reference::sampler_matches_reference`，逐步比较。

位置索引分两段：理解流按 Qwen3-VL 规则（文本依次推进，图像/视频用合并后的 patch 网格，
`current_pos` 按 `max(h, w) / merge` 推进）；生成流从「文本最大位置 + 1 +
`temporal_modality_margin`」开始，按 `base_fps / temporal_compression` 在时间轴上推进。
参照由 `scripts/dump_cosmos3_positions.py` 生成，Rust 侧测试为
`policy::cosmos3::positions::reference::{understanding,joint}_positions_match_reference`。

提示见 `src/policy/cosmos3/prompt.rs`：任务写成 JSON（取景、时长、分辨率、宽高比），再套固定
模板。模板来自 checkpoint 的 `chat_template.jinja`；对「单条文本用户消息」这一固定形状可展开为
常量前后缀，Rust 侧不需要模板引擎。参照由 `scripts/dump_cosmos3_prompt.py` 生成。

推断：官方预处理器只写生成流用的 `pixel_values_videos_gen`，不写理解流的 `pixel_values`，
所以推理时 SigLIP 视觉塔不参与，理解流是纯文本。此项尚未用完整策略验证。

导入见 `Cosmos3Runtime::import_lerobot`：主干在 `model.network.mot_backbone.model.` 下，
生成流投影在 `model.network.` 下，VAE 在 `model.network.vae.` 下；`language_model` 映射为
本端口的 `text`，VAE 下采样的 `resample.1` 映射为 `resample`。

画布步骤为 `Step::Cosmos3Canvas`：读取各相机视角、按需缩放到 `view_size`、拼接、量化、
适配画布，写出 `[-1,1]` 的 `observation.images.canvas`。画布尺寸由「视角尺寸 × 视角数」
推出，三视角时下排占半高。

画布模块见 `src/policy/cosmos3/canvas.rs`。档位选择与内容尺寸完全按官方规则；缩放采用
Keys 双三次核（**`a = -0.5`**）加抗锯齿尺度，与 torch 的 `F.interpolate(antialias=True)`
最大差 2.6e-3。

注意：官方注释与常见文档都写 `a = -0.75`，但实测 PyTorch 的抗锯齿路径用的是 `-0.5`。
这一结论由脉冲法反推权重、再按比例拟合得出，并用 256x512 → 320x160 的随机平面复核。

部署侧已接入：配置枚举接受 `type = "cosmos3_edge"`，`serve` 会按 `action_feature` 设定动作
维度；`convert` 对 cosmos3 仍返回「不支持转换」，因为处理器与合约来源尚未实现。

`Cosmos3Policy::predict` 串起整条链路：画布经 VAE 编码并归一化 → 构造含噪声的生成流状态 →
4 步 uniPC（条件与无条件两条提示）→ 裁剪动作块。当前要求批次里已有三样**已准备**的输入：
`observation.images.canvas`、`observation.state`、`observation.language.tokens`。

运行时见 `src/policy/cosmos3/runtime.rs`：条件帧编码、噪声插值、速度掩码、CFG 组合与
4 步采样循环。布局与官方一致：条件帧 1 帧、条件动作 1 个（状态），`action_offset = 0`。

整网封装见 `src/policy/cosmos3/net.rs`：视觉塔输出经投影层后**按图像 token 的位置写回**
文本嵌入，再与生成流嵌入一起送进 MoT 主干。参照由 `scripts/dump_cosmos3_net.py` 用官方
`Nemotron3DenseVLMoTModel` 加载真实权重生成，Rust 侧测试为
`policy::cosmos3::net::reference::network_matches_reference`。

## 结论

这是一个独立的大型移植，工作量与现有 ACT、SmolVLA、π₀.₅ 三者之和相当。
在没有官方数值参照环境的情况下直接实现，无法验证正确性，因此本轮不做。

## 建议的推进顺序

1. 先把 LeRobot 的 `feat/add_cosmos3_pr` 合入基线，并在具备 Python 3.12 的环境里跑通
   官方 `Cosmos3EdgePolicy` 的固定输入推理，产出参照张量。
2. 在 Pneuma 侧先实现 `configuration_cosmos3` 的解析与合约校验，拒绝不支持的
   `pretrained_path`、PEFT、视频分支。
3. 按 `mot_backbone` → `time_embedder` → `action2llm/llm2action` 顺序移植，
   每层与参照逐值比对。
4. 最后实现 uniPC 采样器与 CFG；VAE 分支单独评估是否需要加载。
5. 验收判据：与官方 Python 的固定噪声动作块最大绝对误差可解释，且端到端服务能返回
   `[1,32,动作维度]` 的有限动作块。
