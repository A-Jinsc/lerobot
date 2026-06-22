"""
SmolVLA CPU Benchmark
=====================
跑多轮推理，对比基线版 vs 优化版的各环节耗时，
将结果输出为 JSON，供 benchmark.html 使用。

运行方式：
  python demo_benchmark.py
"""

import copy
from collections import deque
import json
import os
import statistics
import time
import torch
import torch.nn.functional as F

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching, make_att_2d_masks
from lerobot.policies.smolvla.smolvlm_with_expert import build_rope_cache
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

torch.set_num_threads(4)
# Force reload modules to pick up code changes
import sys
import importlib
for _mod in [
    "lerobot.policies.smolvla.smolvlm_with_expert",
    "lerobot.policies.smolvla.modeling_smolvla",
    "lerobot.policies.smolvla",
]:
    if _mod in sys.modules:
        importlib.reload(sys.modules[_mod])
DEVICE = torch.device("cpu")
NUM_WARMUP = 1
NUM_RUNS = 5


# ---- 兼容 3D mask 的 eager attention（4D 广播版）----

def compat_eager_attention(num_attention_heads, num_key_value_heads, head_dim, attn_mask, q, k, v):
    """
    Handles 3D mask (B, Q, KV) by expanding to 4D (B, 1, Q, KV) before masking.
    Compatible with both square (prefix prefix) and rectangular (suffix prefix+suffix) masks.
    """
    num_kv_groups = num_attention_heads // num_key_value_heads
    b, seq_k = k.shape[:2]
    _, seq_q = q.shape[:2]

    # Expand K/V to num_attention_heads: (B, seq, num_kv, num_groups, head_dim) -> (B, seq, num_heads, head_dim)
    k = k[:, :, :, None, :].expand(b, seq_k, num_key_value_heads, num_kv_groups, head_dim)
    k = k.reshape(b, seq_k, num_attention_heads, head_dim)
    v = v[:, :, :, None, :].expand(b, seq_k, num_key_value_heads, num_kv_groups, head_dim)
    v = v.reshape(b, seq_k, num_attention_heads, head_dim)

    q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, H, Q, D) and (B, H, KV, D)
    att = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
    att = att.to(torch.float32)
    big_neg = torch.finfo(att.dtype).min

    # Expand 3D mask (B, Q, KV) -> (B, 1, Q, KV) to broadcast over num_heads
    mask_4d = attn_mask.unsqueeze(1)
    # Also handle rectangular: if KV dim doesn't match, slice to correct size
    if mask_4d.shape[3] != seq_k:
        mask_4d = mask_4d[:, :, :, :seq_k]
    att = torch.where(mask_4d.bool(), att, big_neg)
    probs = F.softmax(att, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.transpose(1, 2))
    out = out.transpose(1, 2).reshape(b, seq_q, num_attention_heads * head_dim)
    return out


# ---- 优化版 attention（SDPA）----

def optimized_sdpa(num_attention_heads, num_key_value_heads, head_dim, attn_mask, q, k, v):
    num_kv_groups = num_attention_heads // num_key_value_heads
    b, seq_k = k.shape[:2]

    k = k[:, :, :, None, :].expand(
        b, seq_k, num_key_value_heads, num_kv_groups, head_dim
    )
    k = k.reshape(b, seq_k, num_key_value_heads * num_kv_groups, head_dim)

    v = v[:, :, :, None, :].expand(
        b, seq_k, num_key_value_heads, num_kv_groups, head_dim
    )
    v = v.reshape(b, seq_k, num_key_value_heads * num_kv_groups, head_dim)

    k = k.to(dtype=q.dtype)
    v = v.to(dtype=q.dtype)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(b, -1, num_attention_heads * head_dim)
    return out


# ---- 基线 VLA 模型（保留原 attention 实现）----

class BaselineModel(VLAFlowMatching):
    def __init__(self, config, rtc_processor=None, processor=None):
        super().__init__(config, rtc_processor, processor=processor)
        self._patch_attention_interface()

    def _patch_attention_interface(self):
        cfg = self.vlm_with_expert
        cfg.get_attention_interface = lambda: (
            lambda mask, bs, hd, q, k, v:
            compat_eager_attention(cfg.num_attention_heads, cfg.num_key_value_heads,
                                  cfg.vlm.config.text_config.head_dim, mask, q, k, v)
        )


# ---- 优化版 VLA 模型（StaticCache + RoPE Cache + 消除 torch.cat）----

class OptimizedModel(VLAFlowMatching):
    def __init__(self, config, rtc_processor=None, processor=None):
        super().__init__(config, rtc_processor, processor=processor)
        self._init_static_cache(config)
        self._patch_attention_interface()

    def _patch_attention_interface(self):
        cfg = self.vlm_with_expert
        cfg.get_attention_interface = lambda: (
            lambda mask, bs, hd, q, k, v:
            compat_eager_attention(cfg.num_attention_heads, cfg.num_key_value_heads,
                                  cfg.vlm.config.text_config.head_dim, mask, q, k, v)
        )

    def _init_static_cache(self, config):
        # Allocate enough space for prefix + full chunk (covers all suffix tokens written during denoising)
        chunk_size = config.chunk_size if config.chunk_size > 0 else 50
        self._max_prefix_len = (config.prefix_length if config.prefix_length > 0 else 1024) + chunk_size
        self._num_layers = self.vlm_with_expert.num_vlm_layers
        self._num_kv_heads = self.vlm_with_expert.num_key_value_heads
        head_dim = self.vlm_with_expert.vlm.config.text_config.head_dim
        device = next(self.parameters()).device
        b = 1
        print(f"[OptimizedModel] _init_static_cache: max_prefix={self._max_prefix_len}, layers={self._num_layers}, kv_heads={self._num_kv_heads}, head_dim={head_dim}")
        self._static_k = torch.zeros(
            b, self._num_layers, self._num_kv_heads,
            self._max_prefix_len, head_dim,
            dtype=torch.bfloat16, device=device
        )
        self._static_v = torch.zeros(
            b, self._num_layers, self._num_kv_heads,
            self._max_prefix_len, head_dim,
            dtype=torch.bfloat16, device=device
        )
        self._static_k_offset = 0

        # Pre-build RoPE cache at maximum prefix length
        self._rope_cache = build_rope_cache(
            self._max_prefix_len, head_dim, device
        )

    def _reset_static_cache(self):
        self._static_k_offset = 0
        self._static_k.zero_()
        self._static_v.zero_()

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        """Optimized sample_actions: build masks once, reuse across denoise steps."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache with RoPE cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            rope_cache=self._rope_cache,
            static_k=self._static_k,
            static_v=self._static_v,
            static_offset=0,
            use_static_cache=True,
        )

        # Infer actual prefix length from past_key_values
        first_layer = past_key_values.get(0, {})
        kv_len = first_layer.get("kv_len", 0)
        self._static_k_offset = kv_len

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        # Build suffix mask ONCE before the loop
        suffix_len = self.config.chunk_size
        bs = bsize
        pl = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bs, suffix_len, pl)
        suffix_pad = torch.ones(bs, suffix_len, device=device, dtype=torch.bool)
        suffix_att = torch.ones(bs, suffix_len, device=device, dtype=torch.bool)
        suffix_att_2d_template = make_att_2d_masks(suffix_pad, suffix_att)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids_base = (prefix_offsets - 1).expand(bs, 1)  # last prefix position

        x_t = noise
        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)

            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time_tensor)
            actual_suffix_len = suffix_pad_masks.shape[1]
            full_att_2d = torch.cat([prefix_pad_2d[:, :, :pl], suffix_att_2d_template[:, :actual_suffix_len, :]], dim=2)
            suffix_pos = position_ids_base + torch.cumsum(suffix_pad_masks, dim=1)
            position_ids = torch.cat([prefix_position_ids, suffix_pos], dim=1)

            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
                rope_cache=self._rope_cache,
                static_k=self._static_k,
                static_v=self._static_v,
                static_offset=self._static_k_offset,
                use_static_cache=True,
            )
            suffix_out = outputs_embeds[1][:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t

        return x_t

    def reset(self):
        self._reset_static_cache()


# ---- SmolVLAPolicy 变体 ----

class BaselinePolicy(SmolVLAPolicy):
    def __init__(self, config, processor=None, rtc_processor=None, **kwargs):
        # 用 load_vlm_weights=False 避免联网下载 VLM 权重
        cfg = copy.deepcopy(config)
        cfg.load_vlm_weights = False
        model = BaselineModel(cfg, rtc_processor=rtc_processor, processor=processor)
        super().__init__(config, processor=processor, rtc_processor=rtc_processor, _model=model, **kwargs)
        self.reset()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        if hasattr(self.model, "_reset_static_cache"):
            self.model._reset_static_cache()


class OptimizedPolicy(SmolVLAPolicy):
    def __init__(self, config, processor=None, rtc_processor=None, **kwargs):
        cfg = copy.deepcopy(config)
        cfg.load_vlm_weights = False
        model = OptimizedModel(cfg, rtc_processor=rtc_processor, processor=processor)
        super().__init__(config, processor=processor, rtc_processor=rtc_processor, _model=model, **kwargs)

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        if hasattr(self.model, "_reset_static_cache"):
            self.model._reset_static_cache()


# ---- 辅助函数 ----

def make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return (att_2d_masks & pad_2d_masks)


def make_mock_batch(policy, batch_size=1):
    batch = {
        "observation.images.camera1": torch.randn(batch_size, 3, 224, 224),
        f"{OBS_LANGUAGE_TOKENS}": torch.full((batch_size, 48), 3, dtype=torch.long),
        f"{OBS_LANGUAGE_ATTENTION_MASK}": torch.ones(batch_size, 48, dtype=torch.bool),
        f"{OBS_STATE}": torch.randn(batch_size, 7),
    }
    return batch


def make_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    import math
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = (1.0 / period) * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :vector.shape[-1]] = vector
    return new_vector


def pad_tensor(tensor, max_len, pad_value=0):
    b, d = tensor.shape[:2]
    padded = torch.full((b, max_len, *tensor.shape[2:]), pad_value,
                        dtype=tensor.dtype, device=tensor.device)
    padded[:, :d] = tensor
    return padded


# ---- 分步推理计时 ----

def infer_with_timing(policy, batch):
    timings = {}
    policy.eval()

    t0 = time.perf_counter()

    # Check if static cache is available (optimized policy)
    use_static = hasattr(policy.model, "_static_k") and policy.model._static_k is not None
    rope_cache = getattr(policy.model, "_rope_cache", None)
    if use_static:
        policy.model._reset_static_cache()

    # Prefix Embed
    t1 = time.perf_counter()
    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)
    timings["prefix_embed"] = time.perf_counter() - t1

    # KV Cache Build
    t2 = time.perf_counter()
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    with torch.no_grad():
        _, past_key_values = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=policy.config.use_cache,
            fill_kv_cache=True,
            rope_cache=rope_cache,
            static_k=policy.model._static_k if use_static else None,
            static_v=policy.model._static_v if use_static else None,
            static_offset=0,
            use_static_cache=use_static,
        )
    static_offset = 0
    if use_static and past_key_values:
        static_offset = past_key_values.get(0, {}).get("kv_len", 0)
    timings["kv_build"] = time.perf_counter() - t2

    # Denoise Loop
    bsize = state.shape[0]
    device = state.device
    noise = torch.normal(0.0, 1.0,
                         size=(bsize, policy.config.chunk_size, policy.config.max_action_dim),
                         dtype=torch.float32, device=device)
    num_steps = policy.config.num_steps
    dt = -1.0 / num_steps

    pl = prefix_pad_masks.shape[1]
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    # The global position id of the last suffix token (increases by 1 each step)
    last_suffix_pos = (prefix_offsets - 1 + policy.config.chunk_size).long()

    x_t = noise

    t3 = time.perf_counter()
    for step in range(num_steps):
        current_time = 1.0 + step * dt
        time_tensor = torch.tensor(current_time, dtype=torch.float32, device=device).expand(bsize)

        with torch.no_grad():
            suffix_embs, suffix_pad_masks, suffix_att_masks = policy.model.embed_suffix(x_t, time_tensor)
            # Only forward the LAST token (Q=1) — required for use_cache=True to match KV len
            last_suffix_emb = suffix_embs[:, -1:]   # (B, 1, D)
            last_suffix_pad = suffix_pad_masks[:, -1:]  # (B, 1)
            last_suffix_att = suffix_att_masks[:, -1:]  # (B, 1)

            # Build mask for the single last token attending to full prefix+suffix
            # Row shape: (B, 1, prefix+suffix) — all ones (attends to everything valid)
            prefix_mask_row = torch.ones(bsize, 1, pl, device=device, dtype=torch.bool)
            # Cumulative prefix LM mask within suffix part (always last token → all valid)
            cumsum = torch.cumsum(suffix_att_masks, dim=1)  # (B, suffix_len)
            last_cumsum = cumsum[:, -1:]  # (B, 1)
            valid_suffix_mask = cumsum[:, None, :] <= last_cumsum[:, :, None]  # (B, 1, suffix_len)
            last_valid_mask = valid_suffix_mask[:, :, -1:]  # (B, 1, 1) — last token's valid positions
            pad_2d = suffix_pad_masks[:, None, :] * suffix_pad_masks[:, :, None]  # (B, S, S)
            suffix_mask_row = (last_valid_mask & pad_2d[:, -1:, :])  # (B, 1, suffix_len)
            full_att_row = torch.cat([prefix_mask_row, suffix_mask_row], dim=2)  # (B, 1, prefix+suffix)

            position_ids = torch.cat([prefix_position_ids.long(), last_suffix_pos], dim=1)

            outputs_embeds, past_key_values_out = policy.model.vlm_with_expert.forward(
                attention_mask=full_att_row,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, last_suffix_emb],
                use_cache=policy.config.use_cache,
                fill_kv_cache=False,
                rope_cache=rope_cache,
                static_k=policy.model._static_k if use_static else None,
                static_v=policy.model._static_v if use_static else None,
                static_offset=static_offset,
                use_static_cache=use_static,
            )
            # Update offset: +1 for the single new suffix token written to static cache
            if use_static and past_key_values_out:
                static_offset = past_key_values_out.get(0, {}).get("kv_len", static_offset)
            v_t = outputs_embeds[1].to(dtype=torch.float32)  # (B, 1, D)
            v_t = policy.model.action_out_proj(v_t)  # (B, 1, action_dim)

        # Expand v_t back to full chunk_size for the Euler update
        v_t_full = v_t.squeeze(1).unsqueeze(1).expand(bsize, policy.config.chunk_size, -1)
        x_t = x_t + dt * v_t_full
        # Advance position counter for next step
        last_suffix_pos = last_suffix_pos + 1

    timings["denoise_steps"] = time.perf_counter() - t3
    timings["total"] = timings["prefix_embed"] + timings["kv_build"] + timings["denoise_steps"]

    # 单步 denoise 耗时
    timings["per_denoise_step"] = timings["denoise_steps"] / num_steps
    return timings


# ---- 主函数 ----

def main():
    print("=" * 60)
    print("  SmolVLA CPU Benchmark")
    print("  Baseline vs Optimized (SDPA + StaticCache)")
    print("=" * 60)

    # 加载模型
    print("\nLoading model...")
    t_load = time.time()
    base_policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    base_policy.to(DEVICE)
    base_policy.eval()
    print(f"Model loaded in {time.time() - t_load:.2f}s")

    batch = make_mock_batch(base_policy)

    # 创建两个版本（权重共享）
    base_processor = base_policy.model.vlm_with_expert.processor
    baseline = BaselinePolicy(base_policy.config, processor=base_processor)
    baseline.load_state_dict(base_policy.state_dict(), strict=False)
    baseline.to(DEVICE)
    baseline.eval()

    optimized = OptimizedPolicy(base_policy.config, processor=base_processor)
    optimized.load_state_dict(base_policy.state_dict(), strict=False)
    optimized.to(DEVICE)
    optimized.eval()

    # 重置优化版的 StaticCache
    optimized.model._reset_static_cache()

    print(f"\nWarming up ({NUM_WARMUP} run)...")
    infer_with_timing(baseline, batch)
    infer_with_timing(optimized, batch)

    print(f"\nRunning benchmark ({NUM_RUNS} runs per version)...")

    baseline_runs = []
    optimized_runs = []

    for i in range(NUM_RUNS):
        print(f"  Run {i+1}/{NUM_RUNS}...", end=" ", flush=True)
        baseline_runs.append(infer_with_timing(baseline, batch))
        optimized.model._reset_static_cache()
        optimized_runs.append(infer_with_timing(optimized, batch))
        print("done")

    # 汇总统计
    def stats(runs, key):
        vals = [r[key] for r in runs]
        return {
            "mean": round(statistics.mean(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0, 4),
            "runs": [round(v, 4) for v in vals],
        }

    results = {
        "config": {
            "device": str(DEVICE),
            "num_threads": torch.get_num_threads(),
            "num_runs": NUM_RUNS,
            "num_warmup": NUM_WARMUP,
            "chunk_size": base_policy.config.chunk_size,
            "num_steps": base_policy.config.num_steps,
            "action_dim": base_policy.config.max_action_dim,
        },
        "baseline": {k: stats(baseline_runs, k) for k in [
            "prefix_embed", "kv_build", "denoise_steps", "per_denoise_step", "total"
        ]},
        "optimized": {k: stats(optimized_runs, k) for k in [
            "prefix_embed", "kv_build", "denoise_steps", "per_denoise_step", "total"
        ]},
    }

    # 计算 speedup
    for k in results["baseline"]:
        b_mean = results["baseline"][k]["mean"]
        o_mean = results["optimized"][k]["mean"]
        if b_mean > 0:
            results["optimized"][k]["speedup_percent"] = round((b_mean - o_mean) / b_mean * 100, 1)
        else:
            results["optimized"][k]["speedup_percent"] = 0.0

    # 打印结果
    print("\n" + "=" * 70)
    print("  Benchmark Results")
    print("=" * 70)
    print(f"{'Stage':<20} {'Baseline (s)':>14} {'Optimized (s)':>14} {'Speedup':>10}")
    print("-" * 70)

    stage_labels = {
        "prefix_embed": "Prefix Embed",
        "kv_build": "KV Cache Build",
        "denoise_steps": "Denoise Loop (10x)",
        "per_denoise_step": "Per Denoise Step",
        "total": "Total Inference",
    }

    for k, label in stage_labels.items():
        b = results["baseline"][k]["mean"]
        o = results["optimized"][k]["mean"]
        sp = results["optimized"][k]["speedup_percent"]
        print(f"{label:<20} {b:>14.4f} {o:>14.4f} {sp:>9.1f}%")

    # 保存 JSON（写到 leborot_infer_demo/static/，与 benchmark.html 同目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(script_dir, "static")
    os.makedirs(static_dir, exist_ok=True)

    output_path = os.path.join(static_dir, "benchmark_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # 多轮耗时波动数据（用于折线图）
    multi_run_data = {
        "baseline": {
            "total": [r["total"] for r in baseline_runs],
            "kv_build": [r["kv_build"] for r in baseline_runs],
            "denoise_steps": [r["denoise_steps"] for r in baseline_runs],
        },
        "optimized": {
            "total": [r["total"] for r in optimized_runs],
            "kv_build": [r["kv_build"] for r in optimized_runs],
            "denoise_steps": [r["denoise_steps"] for r in optimized_runs],
        },
    }
    multi_path = os.path.join(static_dir, "multi_run_data.json")
    with open(multi_path, "w", encoding="utf-8") as f:
        json.dump(multi_run_data, f, indent=2)

    print(f"Multi-run data saved to: {multi_path}")
    print("\nDone. Open leborot_infer_demo/static/benchmark.html to view results.")


if __name__ == "__main__":
    main()
