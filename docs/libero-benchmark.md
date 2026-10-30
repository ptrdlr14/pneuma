# LIBERO benchmark

[返回 README](../README.md) · [SmolVLA 接入与数值对齐](smolvla.md)

`scripts/libero_benchmark.py` 使用官方 LeRobot `LiberoEnv` 和 `LiberoProcessorStep`，
通过真实 LIBERO / MuJoCo 环境评测 SmolVLA。默认比较官方 Python CPU/F32 与 Pneuma Metal，
运行 `libero_spatial` 的 10 个任务，每任务 5 局。此轮是小样本描述性评测，
不是 LIBERO 全套四个 suite 各任务 50 局的标准大规模得分。

## 运行

Python 环境需包含 LeRobot 的 SmolVLA、LIBERO、remote 依赖，以及可用的离屏 OpenGL 渲染。
需要完整本地 checkpoint 和 SmolVLM2-500M 的配置/tokenizer 资源。

```sh
cargo build --release

/path/to/python -I scripts/libero_benchmark.py \
  --checkpoint=/absolute/path/to/smolvla_libero \
  --vlm-path=/absolute/path/to/SmolVLM2-500M-Video-Instruct \
  --lerobot-src=/absolute/path/to/lerobot/src \
  --backend=all --episodes-per-task=5 \
  --output=outputs/libero-spatial-5.json
```

`--backend=all` 为 Python + Metal；也可单独指定 `python`、`metal` 或 `flex`。
`--suite` 支持 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`、`libero_90`。
`--tasks=0,1` 缩小任务范围，`--episodes-per-task` 修改每任务局数。
续跑时使用相同命令加 `--resume`；源文件、模型、资源、参数或二进制摘要变化会拒绝续跑。
程序只管理自己启动的服务进程，端口自动分配。

默认每局执行 checkpoint 动作块的前 50 步，`--action-steps` 可统一缩短。
Python 使用 4 个 CPU 线程；`--threads` 可覆盖。仿真使用 20 Hz 相对末端控制、
每局开始执行 10 个 no-op 稳定步骤，并使用官方初始状态数组的第 `episode` 项。
默认上限为 spatial/object 280 步、goal 300 步、libero_10 520 步、libero_90 400 步。
任务成功使用环境自身的 `check_success()`；不对动作额外裁剪、插值或变更夹爪符号。

## 评测模型副本与输入契约

本机 checkpoint 的原始配置存在 state6 / stats8 冲突，还声明第三路相机；
官方 LIBERO 环境输出两路相机和 8 维状态。脚本验证归一化统计确为 8 维后，
在输出目录创建单独的 checkpoint 配置副本，并记录每项修改：

- 状态为末端位置 3 维、四元数转轴角 3 维、夹爪 qpos 2 维。
- camera1 使用 agentview，camera2 使用 wrist；两者按官方 processor 翻转高、宽两个轴。
- 删除不存在的 camera3 配置，不填充伪图像。这与官方 `prepare_images` 在 `empty_cameras=0`
  且输入缺 camera3 时只编码实际两路相机的行为一致。
- 协议直接使用 checkpoint 特征名，因此副本将 rename_map 清空。

JSON 配置复制后修改，safetensors 使用符号链接；源 checkpoint 不修改。
两端使用相同副本、归一化统计和 256×256 环境图像，模型内部 resize 至 512×512。
该口径应与原始 checkpoint 直接运行、三相机数值对齐测试、其他 LIBERO 模型得分分开阅读。

## 结果口径

每局完成后原子写入 JSON，包含任务编号、语言、初始状态索引、种子、成功与否、步数、
耗时及初始物理状态/观测摘要。元数据包含模型与配置 SHA256、官方源码、任务 BDDL、
初始状态数组、VLM/tokenizer、脚本和服务端二进制摘要，以及依赖版本。

汇总给出总体及逐任务成功数、总体 Wilson 95% 区间、请求耗时中位/P95 与 Rust worker 耗时。
任务内样本相关，简单总体区间仅作描述；每任务 5 局不足以做性能优劣或非劣性结论。
Python 请求计时含本地 processor、模型和反归一化；Rust 请求计时另含协议与传输。
Rust 服务启动预热 1 次，Python 未另做预热，首请求包含其冷启动开销；延迟按各自闭环请求汇总。
仿真按步推进，不按墙钟实时调度，因此这些延迟不能直接解释为控制截止时间是否满足。

配对仅依据相同任务、初始状态和 episode 编号。两端分别采样初始高斯噪声，
Python 每局播种，Rust 沿用后端 RNG，噪声流不配对；成功率差异同时包含采样和闭环敏感性。
数值移植是否正确应以固定完整噪声的 [SmolVLA parity](smolvla.md#数值参照) 为准。

## 正式结果：2026-10-29

`libero_spatial` 全部 10 个任务，每任务使用官方初始状态 0–4，共两端各 50 局。
运行时间为本机 UTC+08:00 的 21:55:43–22:20:49，所有局完成，进程正常退出。
动作块执行 50 步，每局最多 280 步，使用完整 10 步 flow matching。

| 指标 | 官方 Python CPU/F32 | Pneuma Metal/F32 |
|---|---:|---:|
| 成功数 / 局数 | 34 / 50 | 38 / 50 |
| 成功率 | 68% | 76% |
| Wilson 95% 区间 | 54.2%–79.2% | 62.6%–85.7% |
| 推理请求数 | 182 | 167 |
| 请求耗时中位数 | 1,409.7 ms | 739.5 ms |
| 请求耗时 P95 | 2,197.7 ms | 1,185.5 ms |
| 服务端 worker 中位数 | — | 736.5 ms |
| 服务端 worker P95 | — | 1,182.4 ms |

本轮 Metal 的请求耗时中位数约为 Python 的 52.5%（Python / Metal 为 1.91）。
两端请求包含的前后处理、协议开销与预热口径见上文，且闭环轨迹不同，不能将此比值视作纯模型算子的加速比。

所有任务均为将黑碗放到盘子上，下表按黑碗的初始位置区分；编号与 JSON 的 `task_id` 一致。

| 任务编号 | 黑碗初始位置 | Python 成功数 | Metal 成功数 |
|---|---|---:|---:|
| 0 | 盘子与小烤碗之间 | 3/5 | 5/5 |
| 1 | 小烤碗旁 | 4/5 | 5/5 |
| 2 | 桌面中央 | 5/5 | 4/5 |
| 3 | 饼干盒上 | 4/5 | 4/5 |
| 4 | 木柜顶部抽屉内 | 2/5 | 1/5 |
| 5 | 小烤碗上 | 2/5 | 5/5 |
| 6 | 饼干盒旁 | 5/5 | 5/5 |
| 7 | 炉灶上 | 5/5 | 5/5 |
| 8 | 盘子旁 | 1/5 | 1/5 |
| 9 | 木柜上 | 3/5 | 3/5 |

50 组初始物理状态和输入观测 SHA256 全部一致。Metal 独有成功 6 局，Python 独有成功 2 局，
成功率差为 +8 个百分点。每任务仅 5 局，噪声流独立，当前结果仅作描述性基线，
不据此宣称成功率优于 Python 或通过非劣性验收。两端共同的低成功任务为顶部抽屉内和盘子旁；
本轮没有记录视频，失败原因尚未逐局诊断。

## 本机运行环境

2026-10-29 的评测使用 Apple M5、macOS arm64、CPU/F32 官方 Python 与 Metal/F32 Pneuma。
Python 配置为 4 个 CPU 线程；PyTorch 2.11.0、NumPy 1.26.4、Transformers 5.5.4、
MuJoCo 3.1.6、robosuite 1.4.1、eclipse-zenoh 1.10.2。

- LeRobot 源码提交：`f9e9d739ea7055b282ea1b165e8785412a674693`。
- LIBERO 源码提交：`8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- 原始模型 `model.safetensors` SHA256：`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`。
- 正式结果与完整来源记录：`outputs/libero-spatial-5.json`。
- 评测配置副本：`outputs/libero-spatial-5-checkpoint/`。
- 服务配置与日志：`outputs/libero-spatial-5-server.toml`、`outputs/libero-spatial-5-metal.log`。

`outputs/` 不纳入版本控制；本文保留结果概述，逐局记录和完整摘要保存在本机 JSON 中。

## 完整预检

2026-10-29，spatial task 0、初始状态 0，Python 与 Metal 均完成任务，分别 79 / 82 步。
预检结果位于 `outputs/libero-spatial-preflight.json`，仅用于验证从仿真到服务的完整链路，
不计入正式评测结果。
