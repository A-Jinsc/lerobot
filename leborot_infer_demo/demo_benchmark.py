"""
SmolVLA CPU Benchmark
=====================
跑多轮推理，对比基线版 vs 优化版的各环节耗时，
将结果输出为 JSON，供 benchmark.html 使用。

运行方式：
  python demo_benchmark.py
"""

import json
import statistics
import time
import torch
import torch.nn.functional as F

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

torch.set_num_threads(4)
DEVICE = torch.device("cpu")
NUM_WARMUP = 1
NUM_RUNS = 5


# ---- 原版 attention（来自 smolvlm_with_expert.py）----

def baseline_eager_attention(num_attention_heads, num_key_value_heads, head_dim, attn_mask, q, k, v):
    num_kv_groups = num_attention_heads // num_key_value_heads
    b, seq_k = k.shape[:2]

    k = k[:, :, :, None, :].expand(b, seq_k, num_key_value_heads, num_kv_groups, head_dim)
    k = k.reshape(b, seq_k, num_key_value_heads * num_kv_groups, head_dim)
    v = v[:, :, :, None, :].expand(b, seq_k, num_key_value_heads, num_kv_groups, head_dim)
    v = v.reshape(b, seq_k, num_key_value_heads * num_kv_groups, head_dim)

    q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    att = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
    att = att.to(torch.float32)
    big_neg = torch.finfo(att.dtype).min
    att = torch.where(attn_mask[:, None, :, :], att, big_neg)
    probs = F.softmax(att, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.permute(0, 2, 1, 3))
    out = out.permute(0, 2, 1, 3).reshape(b, -1, num_attention_heads * head_dim)
    return out


# ---- 优化版 attention（SDPA）----

def optimized_sdpa(num_attention_heads, num_key_value_heads, head_dim, attn_mask, q, k, v):
    num_kv_groups = num_attention_heads // num_key_value_heads
    b, seq_k = k.shape[:2]

    k = k.view(b, seq_k, num_key_value_heads, num_kv_groups, head_dim).transpose(1, 2)
    v = v.view(b, seq_k, num_key_value_heads, num_kv_groups, head_dim).transpose(1, 2)
    q = q.transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(b, -1, num_attention_heads * head_dim)
    return out


# ---- 基线 VLA 模型（保留原 attention 实现）----

class BaselineModel(VLAFlowMatching):
    def __init__(self, config, rtc_processor=None):
        super().__init__(config, rtc_processor)
        self._patch_attention_interface()

    def _patch_attention_interface(self):
        cfg = self.vlm_with_expert
        cfg.get_attention_interface = lambda: (
            lambda q, k, v, mask, bs, hd:
            baseline_eager_attention(cfg.num_attention_heads, cfg.num_key_value_heads,
                                     cfg.vlm.config.text_config.head_dim, mask, q, k, v)
        )


# ---- 优化版 VLA 模型（StaticCache + SDPA）----

class OptimizedModel(VLAFlowMatching):
    def __init__(self, config, rtc_processor=None):
        super().__init__(config, rtc_processor)
        self._init_static_cache(config)
        self._patch_attention_interface()

    def _init_static_cache(self, config):
        self._max_prefix_len = config.prefix_length if config.prefix_length > 0 else 1024
        self._num_layers = self.vlm_with_expert.num_vlm_layers
        self._num_kv_heads = self.vlm_with_expert.num_key_value_heads
        self._head_dim = self.vlm_with_expert.vlm.config.text_config.head_dim
        device = next(self.parameters()).device
        b = 1
        self._static_k = torch.zeros(b, self._num_layers, self._num_kv_heads,
                                     self._max_prefix_len, self._head_dim,
                                     dtype=torch.bfloat16, device=device)
        self._static_v = torch.zeros(b, self._num_layers, self._num_kv_heads,
                                     self._max_prefix_len, self._head_dim,
                                     dtype=torch.bfloat16, device=device)
        self._static_k_offset = 0

    def _patch_attention_interface(self):
        cfg = self.vlm_with_expert
        cfg.get_attention_interface = lambda: (
            lambda q, k, v, mask, bs, hd:
            optimized_sdpa(cfg.num_attention_heads, cfg.num_key_value_heads,
                            cfg.vlm.config.text_config.head_dim, mask, q, k, v)
        )


# ---- SmolVLAPolicy 变体 ----

class BaselinePolicy(SmolVLAPolicy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BaselineModel(config, rtc_processor=self.rtc_processor)


class OptimizedPolicy(SmolVLAPolicy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = OptimizedModel(config, rtc_processor=self.rtc_processor)


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
        )
    timings["kv_build"] = time.perf_counter() - t2

    # Denoise Loop
    bsize = state.shape[0]
    device = state.device
    noise = torch.normal(0.0, 1.0,
                         size=(bsize, policy.config.chunk_size, policy.config.max_action_dim),
                         dtype=torch.float32, device=device)
    num_steps = policy.config.num_steps
    dt = -1.0 / num_steps
    x_t = noise

    t3 = time.perf_counter()
    for step in range(num_steps):
        current_time = 1.0 + step * dt
        time_tensor = torch.tensor(current_time, dtype=torch.float32, device=device).expand(bsize)

        with torch.no_grad():
            suffix_embs, suffix_pad_masks, suffix_att_masks = policy.model.embed_suffix(x_t, time_tensor)
            suffix_len = suffix_pad_masks.shape[1]
            bs = prefix_pad_masks.shape[0]
            pl = prefix_pad_masks.shape[1]
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bs, suffix_len, pl)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            outputs_embeds, _ = policy.model.vlm_with_expert.forward(
                attention_mask=full_att_2d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=policy.config.use_cache,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1][:, -policy.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = policy.model.action_out_proj(suffix_out)

        x_t = x_t + dt * v_t

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
    baseline = BaselinePolicy(base_policy.config)
    baseline.load_state_dict(base_policy.state_dict(), strict=False)
    baseline.to(DEVICE)
    baseline.eval()

    optimized = OptimizedPolicy(base_policy.config)
    optimized.load_state_dict(base_policy.state_dict(), strict=False)
    optimized.to(DEVICE)
    optimized.eval()

    # 重置优化版的 StaticCache
    optimized.model._static_k_offset = 0

    print(f"\nWarming up ({NUM_WARMUP} run)...")
    infer_with_timing(baseline, batch)
    infer_with_timing(optimized, batch)

    print(f"\nRunning benchmark ({NUM_RUNS} runs per version)...")

    baseline_runs = []
    optimized_runs = []

    for i in range(NUM_RUNS):
        print(f"  Run {i+1}/{NUM_RUNS}...", end=" ", flush=True)
        baseline_runs.append(infer_with_timing(baseline, batch))
        optimized.model._static_k_offset = 0
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

    # 保存 JSON
    output_path = "static/benchmark_data.json"
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
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
    multi_path = "static/multi_run_data.json"
    with open(multi_path, "w", encoding="utf-8") as f:
        json.dump(multi_run_data, f, indent=2)

    print(f"Multi-run data saved to: {multi_path}")
    print("\nDone. Open static/benchmark.html to view results.")


if __name__ == "__main__":
    main()
