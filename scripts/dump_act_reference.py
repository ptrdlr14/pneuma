#!/usr/bin/env python
"""M2 判据：生成与 Rust Burn ACT 逐元素对齐的 python 参照产物。

对齐目标 = 官方服务链路里 Rust 端复刻的那一段：
  normalize（checkpoint 内嵌缓冲）→ ACT 前向（eval、z=0）→ unnormalize。
权重加载绕开 from_pretrained 的键名（checkpoint 带 model. 前缀与归一化缓冲），
直接剥前缀后 strict=True 灌进 policy.model，保证没有键静默漏载。

产物：
  scripts/fixtures/pos2d_small.json        ACTSinusoidalPositionEmbedding2d(8) 在 3×4 布局
  scripts/fixtures/parity/state.bin        14×f32 LE，原始（未归一化）观测状态
  scripts/fixtures/parity/image_hwc.bin    480×640×3 uint8，行主序 HWC
  scripts/fixtures/parity/actions_ref.bin  100×14 f32 LE，反归一化动作（canonical 语义）

用法：PNEUMA_ACT_CKPT=<ckpt目录> .venv/bin/python scripts/dump_act_reference.py
      （缺省用本机 HF 缓存里的 lerobot/act_aloha_sim_transfer_cube_human）
"""

import glob
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTSinusoidalPositionEmbedding2d

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "scripts" / "fixtures"


def checkpoint_dir() -> str:
    env = os.environ.get("PNEUMA_ACT_CKPT")
    if env:
        return env
    # ModelScope 目录约定：<根目录>/<owner>/<name>，优先级高于旧的 HF 缓存。
    for candidate in (
        Path("/data/nvme1/modelscope/lerobot/act_aloha_sim_transfer_cube_human"),
        Path.home() / "modelscope" / "lerobot" / "act_aloha_sim_transfer_cube_human",
    ):
        if (candidate / "config.json").is_file():
            return str(candidate)
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    snaps = sorted(
        glob.glob(
            str(hub / "models--lerobot--act_aloha_sim_transfer_cube_human" / "snapshots" / "*")
        )
    )
    if not snaps:
        raise SystemExit(
            "未找到 checkpoint：用 modelscope download 下到 /data/nvme1/modelscope，"
            "或设 PNEUMA_ACT_CKPT"
        )
    return snaps[-1]


def dump_pos2d_fixture() -> None:
    """Rust sinusoidal_2d(device, 3, 4, 8) 的参照：行主序 (h*dim, 2*dim) 展平。"""
    emb = ACTSinusoidalPositionEmbedding2d(8)
    emb.eval()
    x = torch.zeros(1, 16, 3, 4)  # 仅用形状；通道数 2*dimension
    pos = emb(x)[0]  # (2*dim, H, W)
    rows = pos.permute(1, 2, 0).reshape(-1)  # (h w) 行主序、每行 2*dim 列
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "pos2d_small.json").write_text(json.dumps(rows.tolist()))
    print(f"pos2d_small.json: {rows.numel()} 个值")


def _python_encoder_inputs(model, batch):
    """复刻 ACTModel.forward 的 encoder 输入组装（(S,B,D) 布局，B=1）。"""
    import einops

    latent = torch.zeros([1, model.config.latent_dim], dtype=torch.float32)
    tokens = [model.encoder_latent_input_proj(latent)]
    pos = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
    tokens.append(model.encoder_robot_state_input_proj(batch["observation.state"]))
    cam_features = model.backbone(batch["observation.images.top"])["feature_map"]
    cam_pos = model.encoder_cam_feat_pos_embed(cam_features)
    cam_features = model.encoder_img_feat_input_proj(cam_features)
    tokens.extend(list(einops.rearrange(cam_features, "b c h w -> (h w) b c")))
    pos.extend(list(einops.rearrange(cam_pos, "b c h w -> (h w) b c")))
    return torch.stack(tokens, 0), torch.stack(pos, 0)


def load_reference_policy(ckpt: str):
    """与数值对齐和仿真共用的官方 ACT 加载路径，设备固定为 CPU。"""
    config = ACTConfig.from_pretrained(ckpt)
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    policy = ACTPolicy(config)
    policy.eval()

    raw = load_file(str(Path(ckpt) / "model.safetensors"))
    stripped = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    missing, unexpected = policy.model.load_state_dict(stripped, strict=True)
    print(f"权重加载: {len(stripped)} 个键, missing={len(missing)}, unexpected={len(unexpected)}")

    return policy, raw


def predict_reference(policy, raw, state, image):
    """原始观测 → checkpoint 归一化 → 官方前向 → canonical 动作。"""
    img_mean = raw["normalize_inputs.buffer_observation_images_top.mean"]  # (3,1,1)
    img_std = raw["normalize_inputs.buffer_observation_images_top.std"]
    s_mean = raw["normalize_inputs.buffer_observation_state.mean"]  # (14,)
    s_std = raw["normalize_inputs.buffer_observation_state.std"]
    a_mean = raw["unnormalize_outputs.buffer_action.mean"]  # (14,)
    a_std = raw["unnormalize_outputs.buffer_action.std"]

    # 官方 policy_runner._batch + Normalize 的等价式。
    image_n = (torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 - img_mean) / img_std
    state_n = (torch.from_numpy(state) - s_mean) / s_std
    batch = {
        "observation.state": state_n.unsqueeze(0),
        "observation.images.top": image_n.unsqueeze(0),
    }
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, 14) 归一化
    actions = chunk * a_std + a_mean  # Unnormalize 等价
    return actions, batch


def dump_parity_dir() -> None:
    policy, raw = load_reference_policy(checkpoint_dir())
    config = policy.config
    # 确定性伪随机观测（seed=0），Rust parity 测试读同一份二进制。
    rng = np.random.default_rng(0)
    state = rng.uniform(-1.0, 1.0, 14).astype(np.float32)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    actions, batch = predict_reference(policy, raw, state, image)

    # 二分定位用的中间值（Rust debug 测试逐段比对）。
    model = policy.model
    with torch.no_grad():
        feat = model.backbone(batch["observation.images.top"])["feature_map"]
        enc_in = _python_encoder_inputs(model, batch)
        enc_out = model.encoder(enc_in[0], pos_embed=enc_in[1])
    assert actions.shape[1] == config.chunk_size and actions.shape[2] == 14

    out = FIXTURES / "parity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.bin").write_bytes(state.astype("<f4").tobytes())
    (out / "image_hwc.bin").write_bytes(image.tobytes())
    ref = actions[0].to(torch.float32).numpy().astype("<f4")
    (out / "actions_ref.bin").write_bytes(ref.tobytes())
    (out / "backbone_out.bin").write_bytes(feat[0].numpy().astype("<f4").tobytes())
    (out / "encoder_out.bin").write_bytes(
        enc_out.squeeze(1).numpy().astype("<f4").tobytes()
    )
    print(f"parity/: actions {ref.shape}, abs 均值 {np.abs(ref).mean():.4f}")


if __name__ == "__main__":
    dump_pos2d_fixture()
    dump_parity_dir()
