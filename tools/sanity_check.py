"""
1-iter sanity check: Camera+LiDAR BEVFusion (swint_v0p075, TransFusionHead)
- MMDataParallel 单卡, bs=1, workers=0
- 不触发额外下载 (swin_tiny 已缓存)
- 跑 forward → loss.backward() → optimizer.step()
"""

import os, time, sys, warnings
import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ── 0. 路径 ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# ── 1. 加载 torchpack config ──
from torchpack.utils.config import configs

# 主配置链 (recursive=True 会向上遍历 default.yaml)
CONFIG_PATH = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/default.yaml"
configs.load(CONFIG_PATH, recursive=True)

# 额外加载 convfuser (fuser 配置)
configs.load(
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
    recursive=True,
)

# ── 2. 覆盖为 sanity 参数 ──
configs.update({
    "data.samples_per_gpu": 1,
    "data.workers_per_gpu": 0,
    "max_epochs": 1,
})

# ── 3. 转成 mmcv Config ──
from mmcv import Config
from mmdet3d.utils import recursive_eval

cfg = Config(recursive_eval(configs), filename=CONFIG_PATH)

# ── 4. 构建数据集 + DataLoader ──
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader

print("=" * 60)
print("Building dataset (train, nuscenes-mini, 404 samples)...")
dataset = build_dataset(cfg.data.train)
print(f"  dataset length: {len(dataset)}")

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    num_gpus=1,
    dist=False,
    shuffle=True,
    seed=cfg.seed,
)
print("  dataloader ready (bs=1, workers=0)")
print("=" * 60)

# ── 5. 构建模型 ──
from mmdet3d.models import build_model

print("Building model (swint_v0p075, ~40.3M params)...")
model = build_model(cfg.model)
model.init_weights()
model = model.cuda()

# ── 6. MMDataParallel 包裹 ──
from mmcv.parallel import MMDataParallel

model = MMDataParallel(model, device_ids=[0])
print("  Model wrapped in MMDataParallel")
print("=" * 60)

# ── 7. 优化器 ──
from mmcv.runner import build_optimizer

optimizer = build_optimizer(model.module, cfg.optimizer)
print(f"  Optimizer: {type(optimizer).__name__}, lr={cfg.optimizer.lr}")
print("=" * 60)

# ── 8. 拉一个 batch ──
print("Loading 1 batch from dataloader...")
batch = next(iter(data_loader))
print(f"  batch keys: {list(batch.keys())}")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: Tensor{tuple(v.shape)}, {v.dtype}")
    elif isinstance(v, list):
        print(f"    {k}: list[{len(v)}]")
    elif hasattr(v, "data"):
        print(f"    {k}: DataContainer{getattr(v.data, 'shape', '?')}")
    else:
        print(f"    {k}: {type(v).__name__}")

# ── 9. Forward + Backward + Step ──
print("=" * 60)
print("Running 1 iteration...")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()

optimizer.zero_grad()
out = model.train_step(batch, optimizer)
loss = out["loss"]

torch.cuda.synchronize()
t1 = time.perf_counter()

loss.backward()
torch.cuda.synchronize()
t2 = time.perf_counter()

optimizer.step()
torch.cuda.synchronize()
t3 = time.perf_counter()

# ── 10. 输出结果 ──
print("=" * 60)
print("RESULTS")
print("-" * 60)
print(f"  loss (scalar): {loss.item():.6f}")
print(f"  log_vars:      {out.get('log_vars', 'N/A')}")
print(f"  num_samples:   {out.get('num_samples', 'N/A')}")
print("-" * 60)
print(f"  forward time:  {t1 - t0:.3f}s")
print(f"  backward time: {t2 - t1:.3f}s")
print(f"  step time:     {t3 - t2:.3f}s")
print(f"  total time:    {t3 - t0:.3f}s")
print("-" * 60)
peak_gib = torch.cuda.max_memory_allocated() / 1024**3
peak_mib = torch.cuda.max_memory_allocated() / 1024**2
print(f"  peak memory:   {peak_gib:.3f} GiB ({peak_mib:.1f} MiB)")
print(f"  current mem:   {torch.cuda.memory_allocated() / 1024**3:.3f} GiB")
print("=" * 60)
print("Sanity check completed successfully.")