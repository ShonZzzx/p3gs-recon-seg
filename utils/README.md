# 通用处理脚本

本目录保存本项目自己编写或整理的通用处理脚本，应随 GitHub 仓库一起上传。它和 `reports/`、`data/`、`outputs/` 不同，不属于报告材料或实验大文件。

## 脚本说明

| 文件 | 作用 |
| --- | --- |
| `preprocess.py` | 调用 COLMAP 完成特征提取、匹配、稀疏重建和图像去畸变，生成 3DGS 风格输入。 |
| `add_rgb_to_ply.py` | 为 PLY 点云补充或修复 RGB 字段，便于后续可视化和分割。 |
| `fix_full_gaussian_ply.py` | 修复 3DGS PLY 中的 NaN/Inf 等非法浮点值，同时保留完整高斯属性。 |

## 与模块脚本的关系

当前这几个脚本也整理到了 `reconstruction/scripts/` 下，便于按重建模块查找。保留 `utils/` 是为了保留项目级通用工具入口；后续如果脚本只服务某一个阶段，可以继续放入对应模块的 `scripts/`，如果会被多个阶段复用，则放在这里更合适。

## 上传策略

`utils/` 中的 `.py` 脚本应上传。大型查看器压缩包、二进制程序或临时输出不要上传，例如 `.gitignore` 中已忽略的 `utils/viewers.zip`。
