# SmolVLA 推理优化记录

## 一、环境配置

### 1. 创建 conda 环境

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

### 2. 安装依赖

```bash
cd /path/to/lerobot
pip install -U pip setuptools wheel
pip install -e ".[smolvla]"
```

### 3. 配置环境变量

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

### 4. 验证环境

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps:", torch.backends.mps.is_available())
print("cuda:", torch.cuda.is_available())
PY
```

![环境验证](/Volumes/%E8%91%B1%E8%91%B1%E7%9A%84%E7%A1%AC%E7%9B%98/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/lerobot/lerobot/leborot_infer_demo/pic/%E7%8E%AF%E5%A2%83%E9%AA%8C%E8%AF%81.png)



## 二、运行 Benchmark

生成统计级别的对比数据。

```bash
python leborot_infer_demo/demo_benchmark.py
```

生成以下文件：

- `static/benchmark_data.json` — 各环节平均耗时 + 标准差
- `static/multi_run_data.json` — 每轮耗时的原始波动数据

![执行效果](/Volumes/%E8%91%B1%E8%91%B1%E7%9A%84%E7%A1%AC%E7%9B%98/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/lerobot/lerobot/leborot_infer_demo/pic/%E6%89%A7%E8%A1%8C%E6%95%88%E6%9E%9C.png)

- `lerobot/leborot_infer_demo/static/benchmark.html`读取上述生成的数据，并生成可视化界面

![优化效果](/Volumes/%E8%91%B1%E8%91%B1%E7%9A%84%E7%A1%AC%E7%9B%98/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/lerobot/lerobot/leborot_infer_demo/pic/%E4%BC%98%E5%8C%96%E6%95%88%E6%9E%9C.png)

### 优化点说明

#### 优化 1：SDPA 替代手写 Eager Attention

**基线**：手写 `matmul + softmax`，显式 `expand + reshape`，大量中间 tensor。

**优化**：使用 `torch.nn.functional.scaled_dot_product_attention`（SDPA），PyTorch 自动选择最优后端（Mac 上走 MPS/Accelerate），减少中间 tensor 创建和内存拷贝。

```python
def optimized_sdpa_attention_forward(...):
    # expand + reshape 到 [b, heads, seq, head_dim]
    key_states = key_states.to(dtype=query_states.dtype)
    value_states = value_states.to(dtype=query_states.dtype)
    query_states = query_states.transpose(1, 2)

    att_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )
```

#### 优化 2：StaticCache 预分配 KV 缓存

**基线**：每次 denoise step 通过 `torch.cat([past_kv, new_kv], dim=1)` 动态拼接，O(n) 拷贝开销。

**优化**：预先分配固定大小的 KV tensor，每次 denoise 直接写入指定位置，避免 `torch.cat`。

```python
def _static_cache_write(self, layer_idx, key_states, value_states):
   end = self._static_k_offset + seq_len
   # 直接写入预分配位置，O(1) 写入
   self._static_k[0, layer_idx, :, self._static_k_offset:end] = key_states[...]
   self._static_v[0, layer_idx, :, self._static_k_offset:end] = value_states[...]
```

#### 优化 3：分块 Cross-Attention

将 expert 的 cross-attention 查询分块处理（如每块 16 个 action token），减少 denoise step 中的峰值内存占用。



## 三、仓库提交

相关内容已提交至`Github`仓库（https://github.com/A-Jinsc/lerobot）

![提交记录](/Volumes/%E8%91%B1%E8%91%B1%E7%9A%84%E7%A1%AC%E7%9B%98/%E6%88%91%E7%9A%84%E9%A1%B9%E7%9B%AE/lerobot/lerobot/leborot_infer_demo/pic/%E6%8F%90%E4%BA%A4%E8%AE%B0%E5%BD%95.png)
