# 基于 3D Gaussian Splatting 的植物三维重建与点云分割

植物三维重建与表型提取是智能农业与植物表型组学中的重要研究内容。相比传统 SfM/MVS 三维重建流程，3DGS 在渲染速度和视图合成质量方面具有明显优势。然而植物场景中仍面临特有挑战：叶片细长且边界薄弱、植株自遮挡严重、叶面纹理重复、少视角条件下重建不稳定，导致标准 3DGS 在植物场景中出现浮点、模糊及几何表达不准确等问题。

本项目系统复现并对比了标准3D Gaussian Splatting（3DGS）及若干代表性改进方法，在此基础上完成植物点云的叶片分割与表型参数提取，实现从**可视化重建**到**可测量重建**的完整流程。


## 环境配置

可通过以下命令创建并激活名为 `plant3dgs` 的 [conda](https://conda.io/) 环境：

```
conda env create -f environment.yaml
conda activate plant3dgs
```


## 项目结构

本项目按“数据、重建、分割、表型提取”四个任务模块组织文件。空目录通过 `.gitkeep` 保留，便于多人协作时保持统一目录结构。

```text
p3gs-recon-seg/
├── README.md              # 项目说明文档
├── environment.yaml       # Conda 环境配置文件
├── data/                  # 数据集、预处理结果与中间数据
├── reconstruction/        # 三维重建相关代码与实验配置
├── segmentation/          # 叶片语义分割与实例分割相关代码
└── phenotyping/           # 表型参数提取相关代码
```


## 复现方法

本项目复现并对比以下方法。

| # | 方法 | 核心改进 |
|---|------|----------|
| 1 |[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)| 基线方法 |
| 2 |[DepthSplat: Connecting Gaussian Splatting and Depth](https://haofeixu.github.io/depthsplat/)| 将预训练单目/多视角深度估计与 3DGS 联合，以深度先验直接预测高斯位置，支持无需逐场景优化的前馈式重建 |
| 3 |[LangSplat: 3D Language Gaussian Splatting](https://langsplat.github.io/)| 在每个高斯上附加 CLIP 语言特征，通过场景语言场支持开放词汇三维查询与语义分割，将自然语言理解提升至三维空间 |
| 4 |[2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://dl.acm.org/doi/abs/10.1145/3641519.3657428)| 以二维圆盘状高斯替代三维椭球，使基元天然对齐局部表面；引入视角一致性深度失真与法线一致性正则，大幅提升几何重建精度 |
| 5 |[Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes](https://github.com/autonomousvision/gaussian-opacity-fields)| 基于射线-高斯交叉定义体素不透明度场（GOF），引入几何正则约束高斯扁平化，并以自适应四面体网格实现无界场景的高质量表面提取 |
| 6 |[Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering](https://arxiv.org/abs/2312.00109)| 引入稀疏锚点网格作为结构化骨架，由锚点神经特征动态预测周围局部高斯的属性，实现视角自适应渲染并显著减少冗余高斯数量 |
| 7 |[LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS](https://github.com/VITA-Group/LightGaussian)| 基于全局重要性评分对高斯进行剪枝，结合球谐系数蒸馏与向量量化压缩高斯属性，在保持渲染质量的同时实现 15× 体积压缩与 200+ FPS |
| 8 |[FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting](https://arxiv.org/abs/2312.00451)| 针对稀疏输入（少视角）场景，提出邻近度引导的高斯增密策略，并融合单目深度先验与伪视图监督，有效缓解少样本条件下的过拟合与几何退化 |
| 9 |[Mip-Splatting: Alias-free 3D Gaussian Splatting](https://arxiv.org/abs/2311.16493)| 分析 3DGS 在多尺度渲染时的走样根源，引入三维平滑滤波器限制高斯最小尺寸，并以二维 Mip 滤波器取代原始圆形截断，消除缩放与分辨率变化带来的混叠伪影 |
| 10 |[SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://arxiv.org/abs/2311.12775)| 引入表面对齐正则化驱使高斯贴合场景表面，随后以泊松重建从对齐高斯中提取显式三角网格，并将薄层高斯绑定至网格面上，实现高质量网格渲染与编辑 |


## 数据集

本项目使用以下植株数据集。

| 数据集 | 植物种类 | 图像数量 | 分辨率 | 格式 | 预处理 |
|--------|----------|----------|--------|------|--------|
| dataGS | 26种不同形态植物 | 2,924张（每株61–235张） | 1080×1920 | .bmp | COLMAP SfM |

### 数据预处理

所有场景的 `sparse/0/` 文件夹均已通过 COLMAP 完成 Structure-from-Motion (SfM) 结构复原，可直接被 3DGS 代码读取。

如需对新增植物数据进行预处理，可使用 `data/preprocess.py`，自动执行以下流程：

1. **特征提取** — `colmap feature_extractor` 提取图像 SIFT 特征
2. **特征匹配** — `colmap exhaustive_matcher` 穷举匹配不同视角照片
3. **稀疏重建** — `colmap mapper` 进行 SfM 三维骨架重建
4. **目录整理** — 将 `.bin` 位姿文件规范化归类到 `sparse/0/`
5. **去畸变** — `colmap image_undistorter` 去除镜头畸变并更新位姿

```bash
python data/preprocess.py
```

## 三维重建

### 训练

```

```

### 渲染与评估

```

```

### 点云提取

```

```

## 叶片分割

本项目采用 **2D-to-3D 分割**方案：利用二维分割模型在图像中分割出叶片（需人工标注），再通过相机参数反投影至三维空间，完成语义分割与单叶实例分离。

### 语义分割

```

```

### 实例分割（单叶分离）

```

```

## 表型参数提取

基于实例分割后的点云，提取以下表型参数：

| 参数 | 描述 |
|------|------|
| 叶片面积 | |
| 叶片长度 | |
| 叶片宽度 | |
| 叶片倾角 | |
| 叶片数量 | |
| 株高 | |

```

```

## 实验结果

### 渲染质量

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 训练时间 |
|------|--------|--------|---------|----------|
| | | | | |
| | | | | |
| | | | | |

### 分割质量

| 数据集 | 语义 mIoU | 实例 AP |
|--------|-----------|---------|
| | | |
| | | |
| | | |

## 致谢



## 引用

```bibtex
@article{kerbl3Dgaussians,
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal = {ACM Transactions on Graphics},
  year    = {2023},
  volume  = {42},
  number  = {4}
}

% TODO: 补充其他复现方法的引用

```
