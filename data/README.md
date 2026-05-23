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
## 关于预处理
### 数据集前置要求
在使用预处理代码`preprocess.py`之前，请先确认你的文件夹满足以下要求：
```text
你的植物文件夹/（例如 dataGS/plant_003/）
└── images/         # 必须包含此文件夹，里面存放你拍摄的所有原始照片（如 .bmp, .jpg, .png）
```
### 环境准备
由于脚本在后台需要调用 `colmap` 命令行工具，运行前请确保服务器已安装 COLMAP
如果在云服务器（如 AutoDL）环境中，请先在终端运行以下命令一键安装：
```text
apt-get update && apt-get install -y colmap

```
### 运行方法
1. 将本脚本命名为 preprocess.py 并保存。
2. 在终端运行:
   ```text
   python preprocess.py
   ```
3. 运行后，程序会暂停并提示你输入目标植物文件夹的路径。直接在终端中键盘输入或粘贴路径，按下回车即可：
   ```text
   请输入植物场景文件夹的路径 (例如 datasets/plant_003): dataGS/plant_003
   ```
### preprocess.py说明
脚本启动后，将严格按照以下 5 个步骤自动执行：
1. 提取图像特征点：调用 `colmap feature_extractor`，扫描 `images/`文件夹，提取每张照片的数学特征，存入临时数据库。
2. 进行特征点匹配：调用 `colmap exhaustive_matcher`，通过穷举匹配建立不同视角照片之间的联系。
3. 三维稀疏重建：调用 `colmap mapper`（SfM 算法），反推相机空间机位，计算镜头畸变参数，生成初始三维骨架。
4. 整理目录结构：自动将初次重建产生的 `.bin` 核心位姿文件规范化归类到 `sparse/0` 目录。
5. 去畸变：调用 `colmap image_undistorter`，利用刚刚算出的相机参数将原始照片在数学上去畸变（存入`plant_0xx/images_undistorted`），并将去除畸变的新位姿覆盖写入最终目的地 `sparse/0`。
### 运行后目录说明
```text
plant_003/
├── images/               # 你的原始图像（存在畸变）
├── images_undistorted/   # 去畸变后的新图像
└── sparse/
    └── 0/                # 对应去畸变图像的相机位姿
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## 数据集使用指南
### 3DGS标准方法使用
本数据集的结构完全对齐了3DGS官方标准输入。使用以下命令直接开始训练（以 `plant_001`为例，开始前请将路径改为自己的电脑路径）：
```text
python train.py -s /path/to/dataGS/plant_001 --model_path output/plant_001
```

