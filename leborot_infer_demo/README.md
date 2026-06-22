# SmolVLA CPU 推理演示与优化

在 MacBook M1 CPU 上运行 SmolVLA 视觉-语言-动作（VLA）模型的推理 demo，包含性能优化对比。

## 环境搭建

### 1. 创建 conda 环境（推荐）

```bash
conda create -n smolvla python=3.10
conda activate smolvla
```

### 2. 安装 lerobot 及其依赖

```bash
cd /Volumes/葱葱的硬盘/我的项目/lerobot/lerobot

# 安装 lerobot（含 smolvla 依赖）
pip install -e ".[smolvla]"

# macOS 额外依赖（如果提示缺包）
pip install torch
```

### 3. 验证环境

```bash
python -c "from lerobot.policies.smolvla import SmolVLAPolicy; print('OK')"
```

## 推理任务说明

SmolVLA 是一个 VLA（Vision-Language-Action）模型，接收：

- **图像**：机器人摄像头拍摄的当前画面
- **文字指令**：人类下达的操作指令（如 "move the robot arm"）
- **关节状态**：机器人当前各关节的角度/位置

输出：机器人下一步的动作向量（各关节的目标角度）。

**本 demo 使用模拟数据验证推理链路**（无需真实机器人）：

- 模拟图片：随机生成 224×224 RGB 图像
- 模拟指令：固定文字 "move the robot arm"
- 模拟状态：随机生成的 7 自由度关节位置

## 快速开始

> 重要：以下所有脚本需要在 `lerobot/` 目录下运行（即包含 `src/` 文件夹的目录），或设置 `PYTHONPATH`。

```bash
# 设置 Python 路径（如果不在正确目录下运行）
export PYTHONPATH="${PYTHONPATH}:/Volumes/葱葱的硬盘/我的项目/lerobot/lerobot/src"
cd /Volumes/葱葱的硬盘/我的项目/lerobot/lerobot
```

### 推理演示（快速跑通流程）

```bash
python leborot_infer_demo/demo_inference.py
```

输出示例：

```
--- Baseline (Original eager attention + dict KV cache) ---
  Prefix Embed:     0.8234s
  KV Cache Build:  1.2456s
  Denoise Loop:    9.8765s
  Total:          11.9455s

--- Optimized (SDPA + StaticCache + chunked attention) ---
  Prefix Embed:     0.8102s
  KV Cache Build:  1.1987s
  Denoise Loop:    8.1234s
  Total:          10.1323s
```

可选参数：

```bash
# 只跑基线版
python leborot_infer_demo/demo_inference.py --mode baseline

# 只跑优化版
python leborot_infer_demo/demo_inference.py --mode optimized

# 增加运行次数
python leborot_infer_demo/demo_inference.py --num_runs 5
```

### 运行 Benchmark（生成对比数据）

```bash
python leborot_infer_demo/demo_benchmark.py
```

运行后生成：

- `leborot_infer_demo/static/benchmark_data.json`：各环节平均耗时
- `leborot_infer_demo/static/multi_run_data.json`：多轮耗时波动数据

### 查看优化效果可视化

用浏览器打开 `lerobot/leborot_infer_demo/static/benchmark.html`。

如果 benchmark_data.json 不存在，页面会显示模拟数据。

## 优化点说明

### 优化 1：SDPA 替代手写 Eager Attention

- **基线**：手写 `matmul + softmax`，显式 `expand + reshape`，大量中间 tensor
- **优化**：使用 `torch.nn.functional.scaled_dot_product_attention`（SDPA），PyTorch 自动选择最优后端（Mac 上走 MPS/Accelerate），减少中间 tensor 创建和内存拷贝

### 优化 2：StaticCache 预分配 KV 缓存

- **基线**：每次 denoise step 通过 `torch.cat([past_kv, new_kv], dim=1)` 动态拼接 KV，O(n) 拷贝开销
- **优化**：预先分配固定大小的 KV tensor，直接写入指定位置，避免每次 cat 操作

### 优化 3：分块 Cross-Attention

- 将 expert 的 cross-attention 查询分块处理（如每块 16 个 action token），减少 denoise step 中的峰值内存占用

## 文件结构

```
leborot_infer_demo/
  demo_inference.py      # 推理演示脚本（基线 vs 优化版对比）
  demo_benchmark.py      # Benchmark 脚本（跑多轮，生成 JSON 数据）
  static/
    benchmark.html       # 优化效果可视化页面
    benchmark_data.json  # Benchmark 统计结果（自动生成）
    multi_run_data.json  # 多轮耗时波动数据（自动生成）
```

## 常见问题

### 提示 `Module not found: lerobot`

需要设置 Python 路径：

```bash
export PYTHONPATH="/Volumes/葱葱的硬盘/我的项目/lerobot/lerobot/src:$PYTHONPATH"
```

### 模型下载慢

首次运行会自动从 HuggingFace 下载模型权重（约 1GB）。可以设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### M1 Mac 内存不足

如果内存占用过高，减小 batch_size 或图像分辨率：

```python
# 在 demo_inference.py 中修改
images = {"observation.images.camera1": torch.randn(1, 3, 160, 160)}  # 缩小图片
```
