# PLY 点云无标注指标计算操作指南

## 作用

这个脚本用于把不同模型输出的 `.ply` 点云放在同一套规则下计算指标，适合比较 baseline 3DGS、Scaffold-GS、GaussianPlant 等方法。

## 环境安装

```bash
pip install numpy pillow plyfile scikit-learn scikit-image
```

## 输入格式

每个模型至少需要一个 `.ply` 文件：

```text
方法名或实验名|/path/to/point_cloud.ply
```

如果还想计算渲染图边缘强度，额外提供该模型的 test 渲染图目录：

```text
方法名或实验名|/path/to/point_cloud.ply|/path/to/renders/test
```

## 基本命令

只比较点云结构指标：

```bash
python scripts/metrics_tools/ply_metrics.py \
  --item "baseline|/path/to/baseline/point_cloud.ply" \
  --item "scaffold|/path/to/scaffold/point_cloud.ply" \
  --item "gaussianplant|/path/to/gaussianplant/point_cloud.ply" \
  --out-csv metrics.csv \
  --out-json metrics.json
```

同时计算渲染图边缘强度：

```bash
python scripts/metrics_tools/ply_metrics.py \
  --item "baseline|/path/to/baseline/point_cloud.ply|/path/to/baseline/renders/test" \
  --item "scaffold|/path/to/scaffold/point_cloud.ply|/path/to/scaffold/renders/test" \
  --item "gaussianplant|/path/to/gaussianplant/point_cloud.ply|/path/to/gaussianplant/renders/test" \
  --out-csv metrics.csv \
  --out-json metrics.json
```

## 当前服务器示例

```bash
python /root/autodl-tmp/ml_course/scripts/compare_30k/ply_metrics.py \
  --item "scaffold_001|/root/autodl-tmp/ml_course/compare_30k/scaffold/plant_001/model/point_cloud/iteration_30000/point_cloud.ply|/root/autodl-tmp/ml_course/compare_30k/scaffold/plant_001/renders/test" \
  --item "scaffold_002|/root/autodl-tmp/ml_course/compare_30k/scaffold/plant_002/model/point_cloud/iteration_30000/point_cloud.ply|/root/autodl-tmp/ml_course/compare_30k/scaffold/plant_002/renders/test" \
  --item "scaffold_003|/root/autodl-tmp/ml_course/compare_30k/scaffold/plant_003/model/point_cloud/iteration_30000/point_cloud.ply|/root/autodl-tmp/ml_course/compare_30k/scaffold/plant_003/renders/test" \
  --out-csv scaffold_30k_ply_metrics.csv \
  --out-json scaffold_30k_ply_metrics.json
```

## 输出字段

| 字段 | 含义 | 趋势 |
|---|---|---|
| `point_count` | `.ply` 顶点数量 | 不是越多越好，过多可能是冗余或噪声 |
| `outlier_ratio` | 异常点比例，基于 kNN 距离和 `median + 3*MAD` 阈值 | 一般越小越好 |
| `main_component_ratio` | DBSCAN 最大簇点数 / 总采样点数 | 一般越大越好 |
| `bbox_volume` | 轴对齐包围盒体积 | 需结合真实尺度看，离群点会放大它 |
| `local_thickness_median` | 局部 PCA 最小特征值开方的中位数 | 反映局部厚度，不能简单越大越好 |
| `render_edge_strength` | test 渲染图 Sobel/Laplacian 平均边缘强度 | 通常越大边缘越清楚，但噪声也会抬高 |

## 统一比较要求

| 要求 | 原因 |
|---|---|
| 所有方法使用同一数据集、同一 train/test 划分 | 否则指标不公平 |
| `--k`、`--sample-n`、`--seed` 必须一致 | 否则异常点、主连通、厚度结果不可比 |
| 不同方法的 `.ply` 必须使用同一坐标尺度 | 否则 `bbox_volume` 和厚度不能直接比较 |
| 边缘强度必须使用同一组 test 相机渲染图 | 否则视角不同会导致边缘强度不公平 |

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--k` | `16` | kNN 邻居数量 |
| `--sample-n` | `50000` | 复杂指标最多采样点数，点太多时避免计算过慢 |
| `--seed` | `20260605` | 随机采样种子，固定后结果可复现 |
