# 点云分割模块

本目录对应报告中的植物点云语义分割和单叶实例分割阶段，负责管理分割论文源码、本项目脚本、评估程序和本地输出。

## 目录结构

```text
segmentation/
├── README.md
├── source-codes/       # 分割论文源码或第三方实现
├── scripts/            # 本项目分割、聚类、融合和评估脚本
└── outputs/            # 本地中间结果和评估输出，不上传 GitHub
```

## 脚本结构

```text
scripts/
├── grounded_sam/       # Grounded-SAM / SAM 二维掩码与 2D-to-3D 融合
├── s2am3d/             # S2AM3D 交互、自动分割、点击合并与实例细化
├── partfield/          # PartField 特征聚类、簇导出和点云清理
└── evaluation/         # 叶片语义/实例分割评估脚本
```

主要流程如下：

| 路线 | 说明 |
| --- | --- |
| 2D-to-3D SAM / Grounded-SAM | 多视角二维掩码生成，再根据相机参数投影并融合到三维点云。 |
| S2AM3D | 在三维点云上进行可控尺度分割、交互点击、自动实例生成和后处理。 |
| PartField | 提取三维特征场，通过聚类得到候选叶片部件。 |
| 传统几何/聚类 | 使用法向、局部几何、区域生长、DBSCAN 等规则作为可解释基线。 |

## 数据输入

分割输入统一放在根目录 `data/`：

```text
data/segmentData_seg/       # 自动裁剪的植物主体点云
data/segmentData_hand/      # 人工清理的 Handcraft 点云
data/segmentData_labeled/   # CloudCompare 标注的叶片实例真值
```

报告中使用 GOF 重建点云，评估 Plant2、Plant3、Plant13、Plant16、Plant19。人工标注标签数量分别为 `11 / 7 / 27 / 25 / 101`。

## 评价指标

语义分割指标：

- Leaf IoU
- Precision
- Recall
- F1-score

实例分割指标：

- F1@0.5
- F1@0.75
- PQ
- ARI

## 语义分割平均结果

| 指标 | 2D-to-3D Seg | S2AM3D Seg | PartField Seg | 传统 Seg | 2D-to-3D Handcraft | S2AM3D Handcraft | PartField Handcraft | 传统 Handcraft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Leaf IoU | 67.11% | 47.77% | 69.24% | 57.27% | 83.35% | 81.21% | **86.38%** | 51.42% |
| Precision | 87.94% | 72.41% | 90.86% | **98.16%** | 87.75% | 92.03% | 93.04% | **97.29%** |
| Recall | **76.10%** | 50.69% | 75.69% | 57.88% | **95.35%** | 87.82% | 92.39% | 52.28% |
| F1 | 79.46% | 58.69% | 80.66% | 70.94% | 90.37% | 89.39% | **92.41%** | 65.81% |

结论：Handcraft 输入整体明显优于 Seg 自动裁剪输入，说明主体点云清理质量对后续分割影响很大。PartField 在 Handcraft 输入下取得最高 Leaf IoU 和 F1；传统方法 Precision 较高但 Recall 较低，表现出“保守、少误检、多漏检”的特点。

## 实例分割平均结果

| 指标 | 2D-to-3D Seg | S2AM3D Seg | PartField Seg | 传统 Seg | 2D-to-3D Handcraft | S2AM3D Handcraft | PartField Handcraft | 传统 Handcraft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1@0.5 | 35.42% | 0.32% | 8.84% | **64.38%** | 34.60% | 23.42% | 48.91% | **49.82%** |
| F1@0.75 | 16.86% | 0.00% | 4.92% | **21.83%** | 25.44% | 12.65% | **32.33%** | 27.15% |
| PQ | 25.89% | 0.22% | 6.12% | **46.55%** | 27.85% | 17.05% | **39.89%** | 37.33% |
| ARI | **0.3956** | 0.0720 | 0.2134 | 0.3234 | 0.5366 | 0.2986 | **0.5732** | 0.2529 |

结论：单叶实例分割明显比叶/非叶语义分割更难。传统几何方法在 Seg 输入下粗粒度匹配较好，但边界精度有限；PartField 在 Handcraft 输入下 F1@0.75 和 ARI 最高，说明干净点云更有利于特征聚类形成稳定实例；S2AM3D 在 Seg 输入下不稳定，可能受提示点、尺度参数和点云缺失影响。

## 输出目录

本地结果建议放在：

```text
segmentation/outputs/
├── grounded_sam/
├── s2am3d/
├── partfield/
└── evaluation/
```

中间掩码、特征数组、聚类结果、评估表格和可视化文件都不上传 GitHub。
