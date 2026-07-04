# 数据目录

本目录是整个项目共享的本地数据根目录。它放在项目根目录下，而不是放进 `reconstruction/`、`segmentation/` 或 `phenotyping/`，因为三条流程都会复用同一批植物图像、COLMAP 相机文件、重建点云和人工标注。

大体积数据已在 `.gitignore` 中忽略，GitHub 上只保留本说明文件。

## 多视角图像数据

报告中使用的原始数据是真实场景下采集的单株植物多视角图像：

- 每株植物围绕中心轴线进行 `360°` 采集。
- 每个场景约 `60-200` 张图像。
- 原始图像分辨率为 `1080 x 1920`。
- 背景包含盆栽土壤、塑料布、实验台和自然光照变化等干扰。
- 使用 COLMAP / SfM 估计相机位姿和稀疏点云。
- 训练/测试划分沿用 3DGS 默认策略，每隔 8 张图像抽取 1 张作为测试视角。

推荐本地结构如下：

```text
data/dataGS/
├── plant_001/
│   ├── images/
│   │   ├── 0.bmp
│   │   ├── 1.bmp
│   │   └── ...
│   └── sparse/
│       └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
├── plant_002/
└── ...
```

如果某个场景缺少 `sparse/0/`，可以使用 `reconstruction/scripts/preprocess.py` 调用 COLMAP 生成 3DGS 所需输入。

## 重建与分割实验样本

报告重点使用以下 5 个植物样本进行重建和分割实验：

| 样本 | 结构特点 | 主要难点 |
| --- | --- | --- |
| plant_002 | 线状细长叶片 | 大长宽比叶片、薄边界保持 |
| plant_003 | 低矮簇生植株 | 近景小曲面、密集叶片、局部遮挡 |
| plant_013 | 穗状器官和高频细节 | 细粒度结构、自遮挡、局部模糊 |
| plant_016 | 薄壁结构边缘 | 零厚度边界、薄结构断裂控制 |
| plant_019 | 高度重叠繁茂植株 | 多层遮挡、主体连通性、几何补全 |

## 分割点云数据

分割阶段使用 GOF 等重建方法导出的植物点云，并区分自动裁剪输入与人工清理输入：

```text
data/segmentData_seg/       # 自动裁剪得到的植物主体点云
data/segmentData_hand/      # 人工清理后的 Handcraft 点云
data/segmentData_labeled/   # CloudCompare 人工标注的叶片实例点云
```

报告中 Plant2、Plant3、Plant13、Plant16、Plant19 的人工标注标签数量分别为 `11 / 7 / 27 / 25 / 101`。这些 `.ply` 文件体积较大，且属于实验数据，不应提交到 GitHub。

## 表型实验数据

```text
data/phenotypeData/
├── soya.ply
├── soya-leaf.ply
└── soya-handcraft.ply
```

该部分用于大豆样本的表型参数提取实验。报告中对比了自动实例分割结果、人工确认叶片实例和 GT 叶面积，其中 GT 叶片数为 `23`，GT 总叶面积为 `46148.09 mm²`。

## 上传策略

不要提交以下内容：

- 原始多视角图像和 COLMAP 中间结果。
- `.ply`、`.npy`、模型权重、渲染图、掩码图和聚类特征。
- `segmentData_*`、`phenotypeData`、各模块 `outputs/`。

如果后续希望提供可运行 demo，建议只放极小规模样例，或在 README 中说明数据下载/生成方式。
