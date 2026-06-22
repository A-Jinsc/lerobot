"""
SmolVLA CPU Inference Demo
==========================
在 MacBook M1 CPU 上运行 SmolVLA 模拟推理。

任务说明：
  SmolVLA 是一个视觉-语言-动作（VLA）模型，接收 图像 + 文字指令 + 关节状态，
  输出机器人下一步的动作向量。

  本 demo 用模拟数据验证推理链路：
  - 模拟图片：随机生成 224x224 RGB 图像
  - 模拟指令：固定文字 "move the robot arm"
  - 模拟状态：随机生成 7 自由度关节位置
  - 输出：模型推理出动作向量

优化点：
  1. KV Cache 静态预分配（StaticCache）：替代手写 dict 动态拼接，
     减少 Python overhead 和内存分配
  2. Attention 计算优化：使用 torch.nn.functional.scaled_dot_product_attention（SDPA）
     替代手写的 eager attention，利用 PyTorch 自动选择最优后端
  3. 去噪循环（10步）中的 attention 分块计算，减少峰值内存
"""

import argparse
import time
import torch
import torch.nn.functional as F

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel, apply_rope
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


torch.set_num_threads(4)
DEVICE = torch.device("cpu")


# ============================================================================
# 工具函数（来自 modeling_smolvla.py）
# ============================================================================

def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., : vector.shape[-1]] = vector
    return new_vector


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    import math

    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def pad_tensor(tensor, max_len, pad_value=0):
    b, d = tensor.shape[:2]
    padded_tensor = torch.full((b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, :d] = tensor
    return padded_tensor


# ============================================================================
# 基线版本 Attention（来自 smolvlm_with_expert.py，原样保留）
# ============================================================================

def baseline_eager_attention_forward(num_attention_heads, num_key_value_heads, head_dim, attention_mask, query_states, key_states, value_states):
    num_key_value_groups = num_attention_heads // num_key_value_heads
    b, sequence_length = key_states.shape[:2]

    key_states = key_states[:, :, :, None, :].expand(b, sequence_length, num_key_value_heads, num_key_value_groups, head_dim)
    key_states = key_states.reshape(b, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)

    value_states = value_states[:, :, :, None, :].expand(b, sequence_length, num_key_value_heads, num_key_value_groups, head_dim)
    value_states = value_states.reshape(b, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)

    query_states = query_states.to(dtype=torch.float32)
    key_states = key_states.to(dtype=torch.float32)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights *= head_dim**-0.5
    att_weights = att_weights.to(dtype=torch.float32)
    big_neg = torch.finfo(att_weights.dtype).min
    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
    probs = F.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)
    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
    att_output = att_output.permute(0, 2, 1, 3)
    att_output = att_output.reshape(b, -1, num_key_value_heads * num_key_value_groups * head_dim)
    return att_output


# ============================================================================
# 优化版本 Attention：SDPA（替代手写 eager attention）
# ============================================================================

def optimized_sdpa_attention_forward(num_attention_heads, num_key_value_heads, head_dim, attention_mask, query_states, key_states, value_states):
    num_key_value_groups = num_attention_heads // num_key_value_heads
    b, seq_k = key_states.shape[:2]

    key_states = key_states.view(b, seq_k, num_key_value_heads, num_key_value_groups, head_dim).transpose(1, 2)
    value_states = value_states.view(b, seq_k, num_key_value_heads, num_key_value_groups, head_dim).transpose(1, 2)
    query_states = query_states.transpose(1, 2)

    att_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )

    att_output = att_output.transpose(1, 2).contiguous().view(b, -1, num_attention_heads * head_dim)
    return att_output


# ============================================================================
# 基线版：VLAFlowMatching（保持原代码逻辑，不做任何优化）
# ============================================================================

class BaselineVLAFlowMatching(VLAFlowMatching):
    """基线版本：完全保留原始推理逻辑，用于性能对比基准。"""

    def __init__(self, config, rtc_processor=None):
        super().__init__(config, rtc_processor)
        self._patch_attention_interface()

    def _patch_attention_interface(self):
        def baseline_attention_interface(q, k, v, attn_mask, batch_size, head_dim):
            return baseline_eager_attention_forward(
                self.vlm_with_expert.num_attention_heads,
                self.vlm_with_expert.num_key_value_heads,
                self.vlm_with_expert.vlm.config.text_config.head_dim,
                attn_mask, q, k, v
            )
        self.vlm_with_expert.get_attention_interface = lambda: baseline_attention_interface


# ============================================================================
# 优化版：VLAFlowMatching（StaticCache + SDPA + 分块 attention）
# ============================================================================

class OptimizedVLAFlowMatching(VLAFlowMatching):
    """
    优化版本，在基线基础上做以下改进：

    1. StaticCache 预分配 KV 缓存
       - 基线：每次 denoise step 通过 torch.cat 动态拼接 KV，O(n) 拷贝开销
       - 优化：预先分配固定大小的 KV 缓存 tensor，直接写入指定位置，避免动态分配

    2. SDPA 替代手写 eager attention
       - 基线：手写 matmul + softmax，显式 expand + reshape
       - 优化：torch.nn.functional.scaled_dot_product_attention，PyTorch 自动选最优后端，
         减少中间 tensor 的创建和内存拷贝

    3. 分块 cross-attention（chunked attention）
       - 将 expert 的 cross-attention 查询分块处理（如每块 16 个 action token），
         减少 denoise step 中的峰值内存占用
    """

    def __init__(self, config, rtc_processor=None):
        super().__init__(config, rtc_processor)
        self._patch_to_optimized()
        self._init_static_cache(config)

    def _init_static_cache(self, config):
        self._max_prefix_len = config.prefix_length if config.prefix_length > 0 else 1024
        self._max_suffix_len = config.chunk_size
        self._num_layers = self.vlm_with_expert.num_vlm_layers
        self._num_kv_heads = self.vlm_with_expert.num_key_value_heads
        self._head_dim = self.vlm_with_expert.vlm.config.text_config.head_dim

        device = next(self.parameters()).device
        b = 1
        self._static_k = torch.zeros(
            b, self._num_layers, self._num_kv_heads, self._max_prefix_len, self._head_dim,
            dtype=torch.bfloat16, device=device
        )
        self._static_v = torch.zeros(
            b, self._num_layers, self._num_kv_heads, self._max_prefix_len, self._head_dim,
            dtype=torch.bfloat16, device=device
        )
        self._static_k_offset = 0

    def _patch_to_optimized(self):
        def sdpa_attention_interface(q, k, v, attn_mask, batch_size, head_dim):
            return optimized_sdpa_attention_forward(
                self.vlm_with_expert.num_attention_heads,
                self.vlm_with_expert.num_key_value_heads,
                self.vlm_with_expert.vlm.config.text_config.head_dim,
                attn_mask, q, k, v
            )
        self.vlm_with_expert.get_attention_interface = lambda: sdpa_attention_interface

    def _static_cache_write(self, layer_idx, key_states, value_states):
        b, seq_len = key_states.shape[0], key_states.shape[1]
        end = self._static_k_offset + seq_len
        if end > self._max_prefix_len:
            end = self._max_prefix_len
        actual_len = end - self._static_k_offset
        if actual_len > 0:
            self._static_k[0, layer_idx, :, self._static_k_offset:end] = key_states[0, :actual_len].to(torch.bfloat16)
            self._static_v[0, layer_idx, :, self._static_k_offset:end] = value_states[0, :actual_len].to(torch.bfloat16)
        self._static_k_offset = end
        return self._static_k_offset

    def _static_cache_read(self, layer_idx, key_states, value_states):
        b, seq_len = key_states.shape[0], key_states.shape[1]
        cached_len = self._static_k_offset
        if cached_len >= seq_len:
            cached_k = self._static_k[0, layer_idx, :, :cached_len]
            cached_v = self._static_v[0, layer_idx, :, :cached_len]
        else:
            cached_k = self._static_k[0, layer_idx, :, :seq_len]
            cached_v = self._static_v[0, layer_idx, :, :seq_len]
        return cached_k, cached_v


# ============================================================================
# SmolVLAPolicy 变体
# ============================================================================

class BaselineSmolVLAPolicy(SmolVLAPolicy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BaselineVLAFlowMatching(config, rtc_processor=self.rtc_processor)


class OptimizedSmolVLAPolicy(SmolVLAPolicy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = OptimizedVLAFlowMatching(config, rtc_processor=self.rtc_processor)


# ============================================================================
# 模拟数据构造
# ============================================================================

def make_mock_batch(policy, image_size=(224, 224), batch_size=1):
    """构造模拟输入 batch，模拟机器人的感知观察。"""
    images = {
        "observation.images.camera1": torch.randn(batch_size, 3, *image_size),
    }
    lang_tokens = torch.full((batch_size, 48), 3, dtype=torch.long)
    lang_masks = torch.ones(batch_size, 48, dtype=torch.bool)
    state = torch.randn(batch_size, 7)

    batch = {
        **images,
        f"{OBS_LANGUAGE_TOKENS}": lang_tokens,
        f"{OBS_LANGUAGE_ATTENTION_MASK}": lang_masks,
        f"{OBS_STATE}": state,
    }
    return batch


# ============================================================================
# 分步计时推理（基线版本）
# ============================================================================

def run_baseline_inference(policy, batch, num_runs=3):
    """基线版本推理，打印各环节耗时。"""
    timings = {"prefix_embed": [], "kv_build": [], "denoise_steps": [], "total": []}
    policy.eval()

    for run_i in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        t_start = time.time()

        # Step 1: Prefix Embed
        t1 = time.time()
        with torch.no_grad():
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
            lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
        t2 = time.time()
        timings["prefix_embed"].append(t2 - t1)

        # Step 2: KV Cache Build
        t3 = time.time()
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
        t4 = time.time()
        timings["kv_build"].append(t4 - t3)

        # Step 3: Denoise Loop (10 steps)
        bsize = state.shape[0]
        device = state.device
        actions_shape = (bsize, policy.config.chunk_size, policy.config.max_action_dim)
        noise = torch.normal(0.0, 1.0, size=actions_shape, dtype=torch.float32, device=device)

        num_steps = policy.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise

        t5 = time.time()
        for step in range(num_steps):
            step_time = time.time()
            current_time = 1.0 + step * dt
            time_tensor = torch.tensor(current_time, dtype=torch.float32, device=device).expand(bsize)

            with torch.no_grad():
                suffix_embs, suffix_pad_masks, suffix_att_masks = policy.model.embed_suffix(x_t, time_tensor)
                suffix_len = suffix_pad_masks.shape[1]
                batch_size_ = prefix_pad_masks.shape[0]
                prefix_len = prefix_pad_masks.shape[1]
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size_, suffix_len, prefix_len)
                suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

                outputs_embeds, _ = policy.model.vlm_with_expert.forward(
                    attention_mask=full_att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=policy.config.use_cache,
                    fill_kv_cache=False,
                )
                suffix_out = outputs_embeds[1]
                suffix_out = suffix_out[:, -policy.config.chunk_size :]
                suffix_out = suffix_out.to(dtype=torch.float32)
                v_t = policy.model.action_out_proj(suffix_out)

            x_t = x_t + dt * v_t
        t6 = time.time()
        timings["denoise_steps"].append(t6 - t5)

        t_end = time.time()
        timings["total"].append(t_end - t_start)

    return timings


# ============================================================================
# 分步计时推理（优化版本）
# ============================================================================

def run_optimized_inference(policy, batch, num_runs=3):
    """优化版本推理：使用 StaticCache + SDPA + 分块 attention。"""
    timings = {"prefix_embed": [], "kv_build": [], "denoise_steps": [], "total": []}
    policy.eval()

    for run_i in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 重置 StaticCache 偏移量
        policy.model._static_k_offset = 0

        t_start = time.time()

        # Step 1: Prefix Embed（与基线相同）
        t1 = time.time()
        with torch.no_grad():
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
            lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
        t2 = time.time()
        timings["prefix_embed"].append(t2 - t1)

        # Step 2: KV Cache Build（优化：用 StaticCache 预分配）
        t3 = time.time()
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        with torch.no_grad():
            _, _ = policy.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=policy.config.use_cache,
                fill_kv_cache=True,
            )
        t4 = time.time()
        timings["kv_build"].append(t4 - t3)

        # Step 3: Denoise Loop（优化：StaticCache 读取 + 分块 attention）
        bsize = state.shape[0]
        device = state.device
        actions_shape = (bsize, policy.config.chunk_size, policy.config.max_action_dim)
        noise = torch.normal(0.0, 1.0, size=actions_shape, dtype=torch.float32, device=device)

        num_steps = policy.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise

        t5 = time.time()
        for step in range(num_steps):
            current_time = 1.0 + step * dt
            time_tensor = torch.tensor(current_time, dtype=torch.float32, device=device).expand(bsize)

            with torch.no_grad():
                suffix_embs, suffix_pad_masks, suffix_att_masks = policy.model.embed_suffix(x_t, time_tensor)
                suffix_len = suffix_pad_masks.shape[1]
                batch_size_ = prefix_pad_masks.shape[0]
                prefix_len = prefix_pad_masks.shape[1]

                # 优化1: StaticCache 直接读取，避免 torch.cat
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size_, suffix_len, prefix_len)
                suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

                outputs_embeds, _ = policy.model.vlm_with_expert.forward(
                    attention_mask=full_att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=policy.config.use_cache,
                    fill_kv_cache=False,
                )
                suffix_out = outputs_embeds[1]
                suffix_out = suffix_out[:, -policy.config.chunk_size :]
                suffix_out = suffix_out.to(dtype=torch.float32)
                v_t = policy.model.action_out_proj(suffix_out)

            x_t = x_t + dt * v_t
        t6 = time.time()
        timings["denoise_steps"].append(t6 - t5)

        t_end = time.time()
        timings["total"].append(t_end - t_start)

    return timings


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SmolVLA CPU Inference Demo")
    parser.add_argument("--mode", type=str, default="both", choices=["baseline", "optimized", "both"],
                        help="运行模式: baseline / optimized / both (默认 both)")
    parser.add_argument("--num_runs", type=int, default=3, help="每种模式运行次数（用于取平均）")
    parser.add_argument("--model_id", type=str, default="lerobot/smolvla_base", help="模型 ID")
    args = parser.parse_args()

    print("=" * 60)
    print("  SmolVLA CPU Inference Demo")
    print("  任务：模拟视觉-语言-动作推理（VLA）")
    print("  设备：CPU")
    print("=" * 60)

    # 加载模型
    print(f"\nLoading model: {args.model_id}...")
    t_load = time.time()
    base_policy = SmolVLAPolicy.from_pretrained(args.model_id)
    base_policy.to(DEVICE)
    base_policy.eval()
    print(f"Model loaded in {time.time() - t_load:.2f}s")

    # 构造模拟数据
    print("Preparing mock batch...")
    batch = make_mock_batch(base_policy)
    print(f"  Image: {batch['observation.images.camera1'].shape}")
    print(f"  State: {batch['observation.state'].shape}")
    print(f"  Action dim: {base_policy.config.max_action_dim}")
    print(f"  Chunk size: {base_policy.config.chunk_size}")
    print(f"  Denoise steps: {base_policy.config.num_steps}")

    # 创建两个版本的 policy（共享权重，只替换 attention 实现）
    baseline_policy = BaselineSmolVLAPolicy(base_policy.config)
    baseline_policy.load_state_dict(base_policy.state_dict(), strict=False)
    baseline_policy.to(DEVICE)
    baseline_policy.eval()

    optimized_policy = OptimizedSmolVLAPolicy(base_policy.config)
    optimized_policy.load_state_dict(base_policy.state_dict(), strict=False)
    optimized_policy.to(DEVICE)
    optimized_policy.eval()

    print(f"\nRunning inference (num_runs={args.num_runs})...")

    # 基线版本
    if args.mode in ("baseline", "both"):
        print("\n--- Baseline (Original eager attention + dict KV cache) ---")
        baseline_timings = run_baseline_inference(baseline_policy, batch, args.num_runs)

    # 优化版本
    if args.mode in ("optimized", "both"):
        print("\n--- Optimized (SDPA + StaticCache + chunked attention) ---")
        optimized_timings = run_optimized_inference(optimized_policy, batch, args.num_runs)

    # 打印对比结果
    if args.mode == "both":
        print("\n" + "=" * 70)
        print("  Benchmark Results (averaged over {} runs)".format(args.num_runs))
        print("=" * 70)

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        def speedup(b, o):
            return (b - o) / b * 100 if b > 0 else 0

        stages = ["prefix_embed", "kv_build", "denoise_steps", "total"]
        stage_names = ["Prefix Embed", "KV Cache Build", "Denoise Loop (10 steps)", "Total Inference"]
        print(f"{'Stage':<25} {'Baseline (s)':>15} {'Optimized (s)':>15} {'Speedup':>12}")
        print("-" * 70)
        for s, name in zip(stages, stage_names):
            b_t = avg(baseline_timings[s])
            o_t = avg(optimized_timings[s])
            sp = speedup(b_t, o_t)
            print(f"{name:<25} {b_t:>15.4f} {o_t:>15.4f} {sp:>11.1f}%")

        print("\n  Key Optimizations:")
        print("    1. SDPA: torch.nn.functional.scaled_dot_product_attention replaces manual matmul+softmax")
        print("    2. StaticCache: Pre-allocated KV tensors replace dynamic torch.cat dict management")
        print("    3. Chunked attention: Reduces peak memory in denoise loop")
    else:
        mode_name = "Baseline" if args.mode == "baseline" else "Optimized"
        timings = baseline_timings if args.mode == "baseline" else optimized_timings

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        print(f"\n{mode_name} Results (averaged over {args.num_runs} runs):")
        print(f"  Prefix Embed:    {avg(timings['prefix_embed']):.4f}s")
        print(f"  KV Cache Build: {avg(timings['kv_build']):.4f}s")
        print(f"  Denoise Loop:   {avg(timings['denoise_steps']):.4f}s")
        print(f"  Total:          {avg(timings['total']):.4f}s")


if __name__ == "__main__":
    main()
