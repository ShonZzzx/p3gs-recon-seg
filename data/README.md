# 数据目录

本目录用于统一管理项目数据、预处理结果和中间文件。

## 数据集dataGS
本数据集包含26种不同种类的植物多视角照片及COLMAP预处理结果，专门用于3D Gaussian Splatting (3DGS) 以及相关改进方法的等三维重建算法的训练与评估。
1. 植物总数：26个不同形态的植物
2. 相机位姿获取：所有场景的 sparse/ 文件夹均已通过COLMAP进行了结构复原（Structure-from-Motion, SfM），可直接被3DGS代码读取。
3. 图片格式：.bmp

## 数据集文件结构
数据集整体存放在`dataGS/`目录下，包含从`plant_001`到`plant_026`共26个独立的植物场景。每个场景的内部结构如下：

```text
dataGS/
├── plant_001/                  # 001号植物场景
│   ├── images/                 # 存储多视角拍摄的高清原始照片
│   │   ├── 0.bmp
│   │   ├── 1.bmp
│   │   └── ...
│   └── sparse/                 # COLMAP 稀疏重建和相机位姿估计结果
│       └── 0/
│           ├── cameras.bin     # 相机内参
│           ├── images.bin      # 相机外参（位姿）
│           └── points3D.bin    # 稀疏点云
├── plant_002/                  # 002号植物场景
│   ├── images/
│   └── sparse/
└── ...
└── plant_026/                  # 026号植物场景
```

## 数据集使用指南
### 3DGS标准方法使用
本数据集的结构完全对齐了3DGS官方标准输入。使用以下命令直接开始训练（以 `plant_001`为例，开始前请将路径改为自己的电脑路径）：
```text
python train.py -s /path/to/dataGS/plant_001 --model_path output/plant_001
```

