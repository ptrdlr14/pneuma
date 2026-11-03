# 项目说明

# Rules
本项目所有面向用户的回复、进度说明和技术解释均须读取；采用 ASD-STE100 的简化表达原则。

- 默认用中文。采用 ASD-STE100 的受控写作原则；该标准面向英语，不把中文回复宣称为完整标准合规。
- 先写结论，再写原因、操作和验证结果。用短句、主动语态和明确主语。一句话只表达一个主要意思。
- 同一概念始终使用同一术语。保留必要的代码、命令和 API 原名；首次使用专业术语时说明含义。避免含糊代词、无必要缩写和修饰语。
- 操作步骤按执行顺序编号，每步只写一个主要动作。把前置条件写在操作之前。
- 明确区分已验证事实、推断、建议和未完成事项。失败时说明失败阶段、实际影响及下一步。不得把任务受理、进程退出或厂商退出码当作刷写及启动成功。
- 保留任务 ID、关键结果和必要限制。不得为缩短回复而省略执行条件、未知状态或恢复要求。

## 两台机器的分工

**当前机器（macOS，Apple Silicon）只负责开发**：写代码和改文档。
不在本机执行构建、测试和运行——本机没有 CUDA toolkit，结果不代表 server1。

**server1 负责构建、检查、运行**：`mise run build` / `test` / `lint` / `check` 等
Rust 任务会先 rsync 同步代码，再 ssh 到 server1 执行。所有结论以 server1 结果为准。只使用 GPU 7 显卡运行。

server1 实测环境（`~/.ssh/config` 里的 `server1`，User `luyawang`）：
Ubuntu 22.04，192 核，CUDA **13.0**，8× NVIDIA RTX PRO 5000 72GB Blackwell（`nvidia-smi -L` 实测 GPU 0–7），
rustc/cargo **1.99.1** 在 `~/.cargo/bin`（非交互 shell 默认 PATH 里没有），
mise **2026.8.3** 在 `~/.local/bin`。
`/data/nvme1/luyawang/Github/pneuma` 由首次 `mise run sync` 创建。

### 常用命令（都在本机敲，实际在 server1 跑）

```sh
mise run sync           # 只同步代码
mise run test           # server1：cargo test
mise run build:release  # server1：cargo build --release（含 CUDA 后端）
mise run check          # server1：fmt + clippy + test + doc（一次同步）
```

需要手工登到 server1 时：

```sh
ssh server1
export PATH="$HOME/.cargo/bin:$PATH"
cd /data/nvme1/luyawang/Github/pneuma
```

server1 首次编译要下载并编译全部依赖，耗时数分钟；之后为增量编译。

### CUDA 后端与 LLVM 预编译包

`burn` 的 `cuda` 特性始终启用，它经 `cubecl-cuda` → `cubecl-llvm`（特性 `nvptx`）
拉入 `tracel-llvm-bundler`。该 build script 会从 GitHub Releases 下载
`tracel-llvm-23.1.0-4-linux-x64.tar.xz`（45 MB）。

server1 到 GitHub 只有约 9 KB/s，TLS 还常被中断，连 300 B 的 `checksums.json` 都可能失败。
已做两件事：

1. 归档与 sidecar 放在 `~/.cache/tracel/`；解压安装到
   `~/.local/share/tracel/tracel-llvm-23.1.0-4/`，`.tracel-llvm-installed` 存在即不再下载。
2. server1 的 `~/.cargo/config.toml` 设置了 `TRACEL_LLVM_BUNDLER_SKIP_CHECKSUM_DOWNLOAD=1`，
   跳过 sidecar 下载。校验未放宽：归档 sha256 与解压内容 sha256 仍与 sidecar 比对。

归档丢失时，在本机下载后 `scp` 到 server1 的 `~/.cache/tracel/`，文件名必须一致：

```sh
curl -LO https://github.com/tracel-ai/tracel-llvm/releases/download/v23.1.0-4/linux-x64.tar.xz
scp linux-x64.tar.xz server1:~/.cache/tracel/tracel-llvm-23.1.0-4-linux-x64.tar.xz
```

### GPU 卡号

`--device` 支持 PyTorch 写法：`cuda` 由后端自选，`cuda:N` 指定第 N 号卡
（例：`cuda:0`…`cuda:7`，server1 共 8 张卡）。
TOML 里同样写 `device = "cuda:7"`。越界序号不会回退到 CPU，
CubeCL 在首次使用该设备时 panic（`RecvError`），不是启动时的清晰报错。

真机自检（在 server1 建卡并跑一次矩阵乘，不加载权重）：

```sh
PNEUMA_DEVICE=cuda:0 mise run test:device
```

已验证：`cuda:0`、`cuda:3`、`cuda:7` 通过；`cuda:8` 失败。

## 模型权重路径

**统一用 ModelScope 下载**。server1 到 HuggingFace 不稳定，不再使用 `huggingface-cli`。

| 机器 | 权重根目录 | 下载来源 |
|---|---|---|
| 本机（mac） | `/Users/riddler/modelscope` | ModelScope |
| server1 | `/data/nvme1/modelscope` | ModelScope |

ModelScope CLI 在 server1 的绝对路径：

```sh
MS=/home/luyawang/.local/share/mise/installs/pipx-modelscope/latest/bin/modelscope
```

下载用 `--local-dir` 直接落成普通目录，不用 HF 风格的 `snapshots/<hash>` 路径：

```sh
$MS download --repo-type model lerobot/smolvla_libero \
  --local-dir /data/nvme1/modelscope/lerobot/smolvla_libero
```

或在本机敲等价任务（`MODEL` 必填，任务内部 ssh 到 server1）：

```sh
MODEL=lerobot/pi05_libero mise run fetch:model
```

目录约定为 `<根目录>/<owner>/<name>`，所有读权重的入口直接指向该目录：
`convert --input`、`--vlm-path`、Python 侧 `from_pretrained`、评测脚本。

本机 HuggingFace 缓存中的历史 checkpoint 可以继续用，但新下载一律走 ModelScope。
脚本里凡是写本机绝对路径的地方，同步到 server1 后都要换成 `/data/nvme1/modelscope` 下的路径——
这是两台机器最容易出错的差异点。Python 侧已有 `HF_HUB_OFFLINE=1` 和
`local_files_only=True`（见 `scripts/dump_smolvla_reference.py`、`scripts/libero_benchmark.py`），
权重缺失会直接报错而不是静默联网。

### server1 的 Python 参照环境

生成官方参照需要 Python 3.12 与 LeRobot。server1 已有：

```sh
UV=~/.local/share/mise/installs/uv/0.12.20/uv-x86_64-unknown-linux-musl/uv
$UV python find 3.12     # ~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
```

- LeRobot 源码：`/data/nvme1/lerobot`（由本机 rsync，不含 `.git`）。
- 虚拟环境：`/data/nvme1/venvs/lerobot`（已装 `[pi]`，torch + transformers 5.5.4）。
- 跑参照时加 `CUDA_VISIBLE_DEVICES=` 强制 CPU/F32；14 GB 级模型需要数分钟。

```sh
CUDA_VISIBLE_DEVICES= /data/nvme1/venvs/lerobot/bin/python -I \
  scripts/dump_pi05_reference.py --checkpoint=... --vlm-path=... \
  --lerobot-src=/data/nvme1/lerobot/src --output=outputs/pi05-parity
```

### 已确认可下载的模型

| ModelScope 仓库 | `type` | 权重大小 | Pneuma 状态 |
|---|---|---:|---|
| `lerobot/act_aloha_sim_transfer_cube_human` | `act` | 207 MB | 已实现 |
| `lerobot/smolvla_libero` | `smolvla` | 907 MB | 已实现 |
| `lerobot/pi05_libero` | `pi05` | 14.47 GB | 已实现；缺归一化统计，需用 v044 或外部统计 |
| `lerobot/pi05_libero_finetuned_v044` | `pi05` | 7.47 GB | 已实现；带统计，可直接转换 |
| `lerobot/cosmos3-edge` | `cosmos3_edge` | 9.14 GB | 已实现；动作与官方 `generate` 差 2e-6 |
