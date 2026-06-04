# BEVFusionx RTX 5070 学习记录

> 本仓库是 [rathaumons/bevfusionx](https://github.com/rathaumons/bevfusionx) 的个人学习 fork，用于在 **RTX 5070** 上配置和验证 camera-LiDAR 融合 3D 目标检测流程。以下内容为个人记录，非官方文档。

---

## ✅ 已验证状态

| 组件 | 状态 |
|---|---|
| **GPU** | RTX 5070 / compute capability 12.0 |
| **CUDA** | 12.8 (Driver 576.80) |
| **torch** | 2.11.0+cu128 |
| **Python** | 3.11.15 |
| **mmcv-full** | 1.7.4+bevfusionx |
| **mmdet3d** | 1.4.0+bevfusionx |
| **数据集** | nuScenes v1.0-mini (404 train / 323 val) |
| **数据预处理** | ✅ `create_data.py` 通过 |
| **Dataset 加载** (LiDAR-only) | ✅ |
| **Dataset 加载** (Camera+LiDAR) | ✅ |
| **Model build** (swint_v0p075, 40.3M) | ✅ |
| **Model init_weights** (swin_tiny 缓存) | ✅ |
| **Model.cuda()** | ✅ |
| **1-iter sanity check** (forward + backward + step) | ✅ |
| **Peak VRAM** | ~4.84 GiB |

## ⚡ 快速运行 Sanity Check

```bash
conda activate bevfusionx_test
cd ~/fireowl_ws/bevfusionx
python tools/sanity_check.py
```

预期输出：loss ~2758、log_vars 含 loss_heatmap/loss_cls/loss_bbox、forward ~1.0s、peak VRAM ~4.84 GiB。

## 🩹 已知 Patch

**文件**: `mmdet3d/models/vtransforms/depth_lss.py`

```diff
-    def get_cam_feats(self, x, d):
+    def get_cam_feats(self, x, d, mats_dict=None):
```

`BaseDepthTransform.forward` 以 3 个参数 (`img`, `depth`, `mats_dict`) 调用 `get_cam_feats`，但 `DepthLSSTransform` 的实现只接收 2 个参数，导致 camera-LiDAR 模式运行时报 `TypeError: takes 3 positional arguments but 4 were given`。

## ⚠️ 注意事项

- **不要提交** `data/nuscenes/` — 数据集不属于代码库
- **不要提交** `*.pkl` / `*.pth` / `__pycache__` / `.cache` — 预处理产物和缓存
- **不要提交** `build/` / `dist/` / `*.so` — 编译产物
- 原版 BEVFusion (mit-han-lab) 基于 PyTorch 1 + CUDA 11，mmcv 1.x 栈，**不适合 RTX 5070 直接跑**。Ratha Siv 的 [bevfusionx](https://github.com/rathaumons/bevfusionx) 已适配 PyTorch 2 + CUDA 12/13，是 RTX 50 系列的可行起点。

---

---

# Upstream README

# BEVFusion · 𝕏

![demo](assets/demo.gif)

[**BEVFusionx**](https://github.com/rathaumons/bevfusionx) builds on top of [BEVFusion · db75150](https://github.com/mit-han-lab/bevfusion/tree/db75150717a9462cb60241e36ba28d65f6908607), introducing extra documentation, impactful enhancements, and compatibility with both legacy **PyTorch 1 + CUDA 11**, and new **PyTorch 2 + CUDA 12/13**. ✨

All sensor modalities (**Camera ✓ LiDAR ✓ RADAR ✓**) are supported and have been thoroughly tested. 😉

## 🚀 Quick Start

### Installation:

- [CUDA 13.0](installation/CU130.md) · [Docker 🐳](https://github.com/rathaumons/bevfusionx/tree/main/installation/docker#readme) · 🆕
- [CUDA 12.8](installation/CU128.md) · [Docker 🐳](https://github.com/rathaumons/bevfusionx/tree/main/installation/docker#readme)
- [CUDA 12.6](installation/CU126.md) · [Docker 🐳](https://github.com/rathaumons/bevfusionx/tree/main/installation/docker#readme)
- [CUDA 12.1](installation/CU121.md) · [Docker 🐳](https://github.com/rathaumons/bevfusionx/tree/main/installation/docker#readme)
- [CUDA 11.3](installation/CU113.md) · [Docker 🐳](https://github.com/rathaumons/bevfusionx/tree/main/installation/docker#readme)

### Data and Model Preparation:

- [PREPARATION.md](tools/PREPARATION.md)

### Model Evaluation and Training:

- [RUN.md](tools/RUN.md)

### Visualization:

- [VISUALIZATION.md](tools/VISUALIZATION.md)

## 📝 License

[![NOTICE](https://img.shields.io/badge/NOTICE-Present-blue)](NOTICE)
[![LICENSE](https://img.shields.io/badge/LICENSE-Apache_2.0-blue)](LICENSE)
