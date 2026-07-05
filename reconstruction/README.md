# 三维重建模块

本目录对应报告中的三维重建阶段，负责管理 3D Gaussian Splatting 相关论文源码、本项目自写处理脚本，以及本地训练/渲染/点云指标输出。

## 目录结构

```text
reconstruction/
├── README.md
├── source-codes/       # 论文源码或第三方复现代码
├── Engineering integration/
│   ├── roi_fusion/     # ROI 约束、RaDe-GS/GOF 融合和 Gaussian fine-tune 脚本
│   └── metrics_tools/  # 点云结构指标计算工具
├── scripts/            # 本项目重建预处理、PLY 修复和点云指标脚本
└── outputs/            # 本地输出目录，不上传 GitHub
```

## 论文源码

`source-codes/` 用于存放重建方法源码：

```text
source-codes/
├── 3dgs/
├── 2dgs/
├── depthsplat/
├── gof/
├── lightgaussian/
├── mip-splatting/
├── RaDe-GS/
├── scaffold-gs/
└── wheat3dgs/
```

这些目录对应论文或第三方实现。整理上传时应保留原项目的许可证、引用信息和 README；如果源码或子模块太大，建议改用 Git submodule，或在本 README 中记录官方仓库链接。

## Engineering integration

`Engineering integration/` 存放本项目在重建阶段进行的工程性融合实验代码，不属于论文源码，也不包含训练数据或实验输出。该目录主要服务于报告中的“重建方法的工程性融合实验”，用于验证 ROI 预处理、RaDe-GS 与 GOF 融合、Gaussian 参数级融合、融合后 fine-tune、植物区域加权、植物点云去噪以及无人工标注结构指标统计。

主要子目录如下：

| 路径 | 作用 |
| --- | --- |
| `Engineering integration/roi_fusion/` | ROI 数据生成、ROI-RaDe-GS/ROI-GOF 队列训练、B1 融合、保留 Gaussian 参数的融合、融合后 3DGS fine-tune、植物区域加权和去噪。 |
| `Engineering integration/metrics_tools/` | 计算 PLY 点云结构指标，包括点数、异常点比例、主连通比例、包围盒体积、局部厚度中位数和边缘强度等。 |
| `Engineering integration/clean_3dgs_ply_rgb.py` | 修复或补充 3DGS PLY 中的 RGB 颜色字段。 |
| `Engineering integration/txt_xyzrgb_to_ply.py` | 将 `x y z r g b` 文本点云转换为 PLY。 |

其中，`roi_fusion/finetune_from_gaussian_ply.py` 用于从已有 Gaussian PLY 初始化并继续 fine-tune。该流程默认关闭 densification，重点调整已有高斯的颜色、不透明度、尺度、旋转和少量位置参数，用于评估融合高斯点云能否改善 render 图质量。实验结论表明，工程融合和后处理去噪可以带来一定局部改善，但对植物叶片边缘模糊的改善有限，后续仍需要从 3DGS 的植物主体感知损失、边缘加权 densification、薄片高斯约束和结构一致性优化等方向继续改进。

## 本项目脚本

`reconstruction/scripts/` 存放我们为课程项目编写或整理的辅助脚本：

| 文件 | 作用 |
| --- | --- |
| `preprocess.py` | 调用 COLMAP 完成特征提取、匹配、稀疏重建和图像去畸变，生成 3DGS 输入。 |
| `add_rgb_to_ply.py` | 为缺少颜色字段的 PLY 点云补充或修复 RGB 属性。 |
| `fix_full_gaussian_ply.py` | 保留完整 3DGS PLY 属性，同时修复 NaN/Inf 等非法浮点值。 |
| `ply_metrics.py` | 计算点云异常点比例、主连通比例、包围盒体积、局部厚度和渲染边缘强度等指标。 |
| `README_ply_metrics.md` | 点云无标注指标计算说明。 |
| `PLY指标计算操作指南.md` | 中文操作指南和命令示例。 |

## 数据输入

重建输入统一从根目录 `data/` 读取，例如：

```text
data/dataGS/plant_013/
├── images/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

报告中的图像原始分辨率为 `1080 x 1920`，后续主要实验采用 `1/2` 分辨率，在训练开销和细节保留之间折中。

## 报告中的重建设置

- DepthSplat 同时测试零样本推理和 `1000` 步微调。
- Mip-Splatting 使用 `15000` 次迭代。
- 其余大多数基于场景优化的方法使用 `30000` 次迭代。
- 评价指标包含 PSNR、SSIM、LPIPS，以及点云异常点比例、主连通比例、BBox 体积、模型大小和训练时间。

分辨率实验在 Plant1 和 RTX 4090 32G 上完成：

| 下采样倍率 | PSNR | SSIM | LPIPS | 模型大小 | 训练时间 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.1313 | 0.5972 | 0.3135 | 1731 MB | 64 m 1 s |
| 2 | 19.5774 | 0.7041 | 0.2245 | 1229 MB | 37 m 6 s |
| 4 | 21.8779 | 0.8026 | 0.1451 | 677 MB | 20 m 47 s |
| 8 | 24.5072 | 0.8690 | 0.0959 | 277.6 MB | 11 m 36 s |

低分辨率会让渲染指标变好且训练更快，但会削弱叶片边缘和细结构。因此报告后续主要使用 `1/2` 分辨率。

## 主要实验结论

| 样本 | 结果摘要 |
| --- | --- |
| Plant2 | LightGaussian PSNR 最高；RaDe-GS SSIM/LPIPS 最优；GOF 主连通比例 `0.9957`；Wheat3DGS 训练最快 `3.66 min`。 |
| Plant3 | 2DGS PSNR 最高 `21.3602`；RaDe-GS SSIM `0.7508`、LPIPS `0.1714` 最优；Mip-Splatting 主连通比例 `0.9964`。 |
| Plant13 | LightGaussian PSNR 最高 `26.3913` 且模型仅 `49 MB`；GOF SSIM/LPIPS 最优；2DGS 主连通比例 `0.9973`。 |
| Plant16 | RaDe-GS PSNR `27.1185` 和 LPIPS `0.2814` 最优；LightGaussian SSIM 最高；GOF 主连通比例 `0.9960`。 |
| Plant19 | GOF PSNR `29.3338` 和 SSIM `0.9197` 最优；Mip-Splatting LPIPS `0.0984`、主连通比例 `0.9929`；LightGaussian 模型最小但异常点较多。 |

综合来看：标准 3DGS 渲染稳定但强遮挡下几何容易断裂；DepthSplat 受植物场景域差异影响较大；RaDe-GS 适合薄边界；2DGS 适合叶片薄表面；GOF 主体连通性稳定但训练较慢；LightGaussian 压缩明显但可能保留离群点；Mip-Splatting 对强遮挡更稳但模型较大。

## 输出目录

本地输出建议放在：

```text
reconstruction/outputs/
├── 3dgs_plant_013/
├── gof_plant_019/
├── radegs_plant_016/
└── pointcloud_metrics/
```

`outputs/` 已被忽略，不上传 GitHub。
