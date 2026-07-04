# Grounded-SAM 二维到三维分割脚本

本目录保存基于 Grounded-SAM / SAM 的植物前景提取与叶片实例 2D-to-3D 融合脚本。它对应报告中 “二维大模型生成掩码，再投影融合到三维点云” 的分割路线。

## 脚本说明

```text
segmentation/scripts/grounded_sam/
├── batch_plant_segment.py       # 批量生成多视角整株植物前景掩码
├── pointcloud_segment.py        # 将二维前景掩码投影到重建点云，裁剪背景点
├── run_perfect_2d_masks.py      # 准备叶片级二维掩码，用于实例分割实验
├── final_3d_fusion.py           # 融合多视角叶片掩码，生成三维点级实例标签
└── readme.md
```

## 典型流程

### 1. 整株植物前景提取

输入：

```text
data/dataGS/plant_xxx/images/
```

运行 `batch_plant_segment.py` 后，为每个视角生成植物前景掩码。随后使用 `pointcloud_segment.py` 将二维掩码投影回点云，得到自动裁剪的植物主体点云，可放入：

```text
data/segmentData_seg/
```

### 2. 叶片二维掩码准备

`run_perfect_2d_masks.py` 用于准备叶片级二维掩码。通常需要：

- 原始多视角图像。
- 干净植物点云。
- COLMAP 相机文件 `sparse/0/{cameras.bin, images.bin, points3D.bin}`。

### 3. 三维实例融合

`final_3d_fusion.py` 将多视角叶片掩码投影并融合为三维点级实例标签，输出带实例 ID 的 PLY 点云，用于后续语义/实例指标计算。

## 与报告结果的关系

报告中 2D-to-3D 路线在 Handcraft 输入下取得 `83.35%` Leaf IoU、`90.37%` F1；在 Seg 自动裁剪输入下取得 `67.11%` Leaf IoU、`79.46%` F1。结果说明二维大模型先验有帮助，但最终效果仍强依赖点云清理质量、相机投影精度和多视角融合策略。

## 注意事项

脚本中可能仍包含本地路径常量，运行前需要按当前机器的数据目录修改。生成的掩码、融合点云和评估中间文件应放入 `segmentation/outputs/` 或其他被 Git 忽略的本地目录。
