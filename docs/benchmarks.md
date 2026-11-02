# ALOHA 配对评测

[返回 README](../README.md) · [运行与接入](usage.md)

本评测比较同一 ACT checkpoint 在 Python CPU、Rust Flex 和 Rust Metal 上的 ALOHA 闭环表现。
结果用于建立描述性基线；允许的成功率偏差尚未约定，不输出非劣性验收通过结论。

## 运行

需要 Python 3.12、LeRobot 的 `.[remote,aloha]` 依赖及 ACT 所用的 PyTorch、torchvision、einops。
LeRobot 源码要求见[接入说明](usage.md#无硬件协议冒烟)。已有 `.venv` 时，可安装仿真依赖：

```sh
uv pip install -p .venv/bin/python -e "<lerobot 仓>[remote,aloha]"
cargo build --release
.venv/bin/python scripts/sim_rollout.py --backend=all --episodes=50 \
  --checkpoint=/absolute/path/to/checkpoint \
  --output=outputs/aloha-paired.json
```

中断后使用相同参数，加 `--resume` 继续已完成的逐局结果。
未指定 `--checkpoint` 时沿用 `PNEUMA_ACT_CKPT`，再查找本机官方 Transfer Cube checkpoint 缓存。

默认种子 0–49，每局上限 400 步（沿用 LeRobot AlohaEnv 配置）；`--seed-start`
和 `--episode-length` 可覆盖。gym-aloha 的渲染可能有轻微非确定性，报告会核对初始
物理状态，并单列初始图像不一致的种子。Python ACT 在 CPU 上执行，默认 PyTorch
单线程；Rust Flex 和 Metal 服务由脚本依次启动，使用空闲本地端口并在结束时清理。
三方采用相同 checkpoint、480×640 图像及 plain chunk，默认执行 checkpoint 的
`n_action_steps`（当前官方权重为 100）；`--action-steps` 可以统一覆盖三方执行长度。
`--backend=python|flex|metal` 可单独运行；`--backend=remote --endpoint=...`
可连接已有服务，此模式需自行保证服务权重与 `--checkpoint` 一致。

## 结果格式与统计口径

JSON 结果保存 checkpoint/源文件/服务端二进制哈希、依赖版本、初始观测摘要及每局成功率、
步数和逐请求延迟；每完成一局原子写入。续跑要求元数据相同，防止混入不同实现的结果。
汇总给出成功率 Wilson 95% 区间、逐种子胜负及配对 bootstrap 差值区间。
Python 的请求计时包含本地前后处理和推理，Rust 的请求计时还包含协议与传输；
Rust `worker_ms` 单列。仿真按步推进，不能据此判断真实时间控制截止时间是否满足。
生成结果和服务日志存放于已忽略的 `outputs/` 中。

## 2026-10-28 本机首轮记录

以下为已有运行记录，本次文档整理未重跑评测。官方 `act_aloha_sim_transfer_cube_human` checkpoint
`ba73b2766f1371cdc133ca4efb97eb090d744625`，种子 0–49，执行块 100 步、每局上限
400 步，Python CPU 单线程，三方各 50 局。原始完整记录的本地路径为
`outputs/aloha-paired-50.json`；该文件被 Git 忽略，不随仓库分发。

| 实现 | 成功率 | Wilson 95% 区间 | 请求耗时中位 / P95 | 服务端 worker 中位 / P95 |
|---|---|---|---|---|
| Python ACT（CPU） | 43/50，86% | 73.8%–93.0% | 177.7 / 568.7 ms | — |
| Rust Flex | 44/50，88% | 76.2%–94.4% | 290.9 / 483.3 ms | 288.9 / 474.0 ms |
| Rust Metal | 44/50，88% | 76.2%–94.4% | 46.1 / 72.6 ms | 44.6 / 69.7 ms |

所有配对初始物理状态和图像摘要一致。Flex、Metal 与 Python 的成功与否均只在种子 0
不同（Python 失败、Rust 成功），成功率差值均为 +2 个百分点，配对 bootstrap 95%
区间为 [0, +6] 个百分点。这个有限样本经验区间不能证明 Rust 优于 Python，亦不能
替代预先定义偏差界限的非劣性检验；本轮未观察到成功率下降。
Metal 的 worker 中位耗时较 Flex 加速约 6.47 倍。请求耗时统计按实际推理请求汇总，
不同轨迹的请求数分别为 171、170、172，属于各自闭环工作负载的描述性延迟。
