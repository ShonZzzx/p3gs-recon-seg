import os
import struct
import numpy as np
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement

# ==================== 1. COLMAP 二进制文件读取核心底层 ====================
def qvec2rotmat(qvec):
    """将 COLMAP 的四元数转换为 3x3 旋转矩阵"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def read_cameras_binary(path):
    """读取 cameras.bin 获取内参 K (兼容 SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL)"""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<IiQQ", fid.read(24))
            camera_id, model_id, width, height = camera_properties[:4]
            
            # 根据 COLMAP 官方标准自动判断参数个数
            # 0: SIMPLE_PINHOLE (3 params), 1: PINHOLE (4 params), 2: SIMPLE_RADIAL (4 params)
            if model_id == 0:
                num_params = 3
            elif model_id in [1, 2]:
                num_params = 4
            else:
                num_params = 5 # 其他更复杂的畸变模型
                
            params = struct.unpack(f"<{num_params}d", fid.read(8 * num_params))
            
            # 构造内参矩阵 K
            K = np.eye(3)
            if model_id == 0 or model_id == 2: 
                # SIMPLE_PINHOLE 或 SIMPLE_RADIAL: params = [f, cx, cy, ...]
                f, cx, cy = params[0], params[1], params[2]
                K[0, 0] = f
                K[1, 1] = f
                K[0, 2] = cx
                K[1, 2] = cy
            elif model_id == 1: 
                # PINHOLE: params = [fx, fy, cx, cy]
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                K[0, 0] = fx
                K[1, 1] = fy
                K[0, 2] = cx
                K[1, 2] = cy
            else:
                # 保底处理
                K[0, 0], K[1, 1] = params[0], params[0]
                K[0, 2], K[1, 2] = width / 2, height / 2
                
            cameras[camera_id] = {"K": K, "width": width, "height": height}
    return cameras

def read_images_binary(path):
    """读取 images.bin 获取每张照片的 R, T 外参"""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_header = struct.unpack("<idddddddi", fid.read(64))
            image_id = binary_image_header[0]
            qvec = np.array(binary_image_header[1:5])
            tvec = np.array(binary_image_header[5:8])
            camera_id = binary_image_header[8]
            
            # 读取图片文件名（以 \0 结尾）
            image_name = ""
            while True:
                char = fid.read(1).decode("utf-8")
                if char == "\0":
                    break
                image_name += char
                
            # 跳过点云特征点对应的二进制字节
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            fid.seek(num_points2D * 24, os.SEEK_CUR)
            
            R = qvec2rotmat(qvec)
            images[image_name] = {"R": R, "t": tvec, "camera_id": camera_id}
    return images

# ==================== 2. 核心 2D-3D 投影投票算法 ====================
def main():
    # 路径配置
    BASE_DIR = Path("/root/autodl-tmp/Grounded-Segment-Anything-main/plant/plant_003")
    SPARSE_DIR = BASE_DIR / "sparse" / "0"
    MASKS_DIR = BASE_DIR / "masks"
    
    PLY_INPUT = Path("/root/autodl-tmp/Grounded-Segment-Anything-main/pointcloud/plant3/plant_003.ply")
    PLY_OUTPUT = Path("/root/autodl-tmp/Grounded-Segment-Anything-main/pointcloud/plant3/plant_003_segmented.ply")
    
    print(" COLMAP 相机二进制位姿...", flush=True)
    cameras = read_cameras_binary(SPARSE_DIR / "cameras.bin")
    images_meta = read_images_binary(SPARSE_DIR / "images.bin")
    
    print("加载 3D 原始高斯点云...", flush=True)
    plydata = PlyData.read(PLY_INPUT)
    xyz = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']), axis=-1)
    num_points = xyz.shape[0]
    
    # 记录每个 3D 点被多少张 2D 掩码认定为植物
    plant_votes = np.zeros(num_points, dtype=np.int32)
    # 记录该 3D 点总共被多少张照片看到（分母）
    visible_counts = np.zeros(num_points, dtype=np.int32)

    print("空间多视角投影与掩码重合度投票...", flush=True)
    valid_image_count = 0
    
    for img_name, meta in images_meta.items():
        # 寻找对应的 2D 黑白掩码
        mask_stem = Path(img_name).stem
        mask_path = MASKS_DIR / f"{mask_stem}_mask.png"
        
        if not mask_path.exists():
            continue
            
        valid_image_count += 1
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        H, W = mask.shape
        
        # 获取当前相机的内外参
        K = cameras[meta["camera_id"]]["K"]
        R = meta["R"]
        t = meta["t"].reshape(3, 1)
        
        # P_camera = R * P_world + t
        pts_cam = (R @ xyz.T) + t # 形状 (3, num_points)
        
        # 过滤掉人在相机背后的点 (深度 Z <= 0)
        front_mask = pts_cam[2, :] > 0
        
        #  P_pixel = K * P_camera 并归一化 
        pts_pixel = K @ pts_cam
        u = pts_pixel[0, :] / pts_pixel[2, :]
        v = pts_pixel[1, :] / pts_pixel[2, :]
        
        # 过滤出成功投影在 2D 照片图像视网膜内部的点
        in_image_mask = front_mask & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        
        if np.sum(in_image_mask) == 0:
            continue
            
        # 提取这些有效点的像素坐标
        u_coords = np.round(u[in_image_mask]).astype(np.int32)
        v_coords = np.round(v[in_image_mask]).astype(np.int32)
        
        # 检查对应 2D 掩码上的像素值是否为白色 (255 代表植物)
        is_plant_pixel = mask[v_coords, u_coords] == 255
        
        # 记入投票
        visible_indices = np.where(in_image_mask)[0]
        visible_counts[visible_indices] += 1
        plant_votes[visible_indices[is_plant_pixel]] += 1

    print(f"📸 共有 {valid_image_count} 张视角的 2D 掩码参与了空间交叉验证投票。", flush=True)

    #  判定规则：
    # 只要被至少 3 个视角看到，且其中有 50% 以上的视角认为它是叶子
    is_plant_point = (visible_counts >= 3) & ((plant_votes / np.maximum(visible_counts, 1)) >= 0.60)

    print(f"裁剪背景...", flush=True)
    segmented_vertex = plydata['vertex'][is_plant_point]
    
    # 写回ply 文件
    el = PlyElement.describe(segmented_vertex, 'vertex')
    PlyData([el], text=False).write(PLY_OUTPUT)
    
    print("==================================================")
    print(f"原始高斯点数: {num_points}")
    print(f"分割后叶片点数: {np.sum(is_plant_point)} (成功剔除了 {num_points - np.sum(is_plant_point)} 个背景噪点)")
    print(f"新模型已保存至: {PLY_OUTPUT}")
    print("==================================================")

if __name__ == "__main__":
    main()