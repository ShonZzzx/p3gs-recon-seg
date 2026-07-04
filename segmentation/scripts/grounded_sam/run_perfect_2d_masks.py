import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import torch
import numpy as np
from PIL import Image

from groundingdino.util.inference import predict, load_model, load_image
from segment_anything import sam_model_registry, SamPredictor
from plyfile import PlyData 
import pycolmap

BERT_FINAL_DIR = "/root/autodl-tmp/bert-base-uncased-local"

import transformers
transformers.utils.logging.set_verbosity_error()

# =======================================================================
# 路径与超参数配置
# =======================================================================
IMAGE_DIR = "/root/autodl-tmp/Grounded-Segment-Anything-main/plant/plant_019/images"
OUTPUT_LEAF_DIR = "/root/autodl-tmp/Grounded-Segment-Anything-main/plant/plant_019/leaf_masks"
PLANT_PLY_PATH = "/root/autodl-tmp/Grounded-Segment-Anything-main/pointcloud/plant19/plant_019_90_best.ply"
SPARSE_DIR = "/root/autodl-tmp/Grounded-Segment-Anything-main/plant/plant_019/sparse/0" 

# 参数
TEXT_PROMPT = "leaf. small dense leaf . separate flower bud"
BOX_THRESHOLD = 0.10  
TEXT_THRESHOLD = 0.20

os.makedirs(OUTPUT_LEAF_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("加载 3D 点云 与 COLMAP 相机参数...", flush=True)
plydata = PlyData.read(PLANT_PLY_PATH)
plant_xyz = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=1)
reconstruction = pycolmap.Reconstruction(SPARSE_DIR)

print("初始化模型（GroundingDINO & SAM）...", flush=True)
dino_config = "/root/autodl-tmp/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
dino_checkpoint = "/root/autodl-tmp/Grounded-Segment-Anything-main/weights/groundingdino_swint_ogc.pth"
sam_checkpoint = "/root/autodl-tmp/Grounded-Segment-Anything-main/weights/sam_vit_h_4b8939.pth"

import groundingdino.util.get_tokenlizer as gd_tokenizer
from transformers import AutoTokenizer, BertModel
def hacked_get_pretrained_language_model(*args, **kwargs): return BertModel.from_pretrained(BERT_FINAL_DIR)
def hacked_get_tokenlizer(*args, **kwargs): return AutoTokenizer.from_pretrained(BERT_FINAL_DIR)
gd_tokenizer.get_pretrained_language_model = hacked_get_pretrained_language_model
gd_tokenizer.get_tokenlizer = hacked_get_tokenlizer

model = load_model(dino_config, dino_checkpoint, device=DEVICE)
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device=DEVICE)
predictor = SamPredictor(sam)
print("模型加载成功")

# =======================================================================
# 循环遍历
# =======================================================================
img_names = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.bmp'))])
print(f"开始重新生成高精度 2D 实例掩码，共计 {len(img_names)} 张...")

for idx_img, img_name in enumerate(img_names):
    img_path = os.path.join(IMAGE_DIR, img_name)
    image_cv = cv2.imread(img_path)
    if image_cv is None: continue
    h, w, _ = image_cv.shape
    
    # 模型探测
    image_pillow, image_tensor = load_image(img_path)
    boxes, logits, phrases = predict(model=model, image=image_tensor, caption=TEXT_PROMPT, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
    
    # COLMAP 投影点云获取植物骨架
    colmap_img = None
    for im_id, im_obj in reconstruction.images.items():
        if im_obj.name == img_name:
            colmap_img = im_obj
            break
    if colmap_img is None: continue
        
    try:
        if hasattr(colmap_img, 'cam_from_world') and callable(getattr(colmap_img, 'cam_from_world')):
            pose = colmap_img.cam_from_world()
            R, T = pose.rotation.matrix(), pose.translation
        else:
            R, T = colmap_img.rotation_matrix(), colmap_img.translation if hasattr(colmap_img, 'translation') else colmap_img.tvec
    except Exception:
        R, T = colmap_img.rotmat(), colmap_img.tvec
        
    camera = reconstruction.cameras[colmap_img.camera_id]
    calib_res = camera.calibration_matrix()
    K = calib_res if (calib_res.ndim == 2 and calib_res.shape == (3, 3)) else np.array([[calib_res[0], 0, calib_res[2]], [0, calib_res[1], calib_res[3]], [0, 0, 1]])
    
    cam_coords = (R @ plant_xyz.T).T + T
    pixel_coords = (K @ cam_coords.T).T
    z_coords = pixel_coords[:, 2:3]
    z_coords[z_coords == 0] = 1e-5
    plant_pixels = (pixel_coords[:, :2] / z_coords).astype(np.int32)
    
    valid_mask = (plant_pixels[:, 0] >= 0) & (plant_pixels[:, 0] < w) & (plant_pixels[:, 1] >= 0) & (plant_pixels[:, 1] < h)
    pts_in_frame = plant_pixels[valid_mask]
    
    plant_skeleton_2d = np.zeros((h, w), dtype=np.uint8)
    if len(pts_in_frame) > 0:
        plant_skeleton_2d[pts_in_frame[:, 1], pts_in_frame[:, 0]] = 255

    instance_label_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_id_counter = 1
    
    # 获取排序
    sorted_indices = torch.argsort(logits, descending=True).cpu().numpy()
    predictor.set_image(image_cv)
    
    total_skeleton_pts = np.sum(plant_skeleton_2d > 0)
    
# =======================================================================

    # =======================================================================
    total_skeleton_pts = np.sum(plant_skeleton_2d > 0)
    
    for idx in sorted_indices:
        box = boxes[idx]
        box_np = box.cpu().numpy()
        
        cx, cy, bw, bh = box_np[0], box_np[1], box_np[2], box_np[3]
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        
        # 3D 点云骨架中心引导
        box_skeleton_mask = np.zeros((h, w), dtype=np.uint8)
        box_skeleton_mask[y1:y2, x1:x2] = plant_skeleton_2d[y1:y2, x1:x2]
        skel_y, skel_x = np.where(box_skeleton_mask > 0)
        
        if len(skel_x) > 0:
            input_point = np.array([[np.median(skel_x), np.median(skel_y)]])
            input_label = np.array([1])
        else:
            input_point = np.array([[int((x1 + x2) / 2), int((y1 + y2) / 2)]])
            input_label = np.array([1])

        masks, _, _ = predictor.predict(
            point_coords=input_point, 
            point_labels=input_label, 
            box=np.array([x1, y1, x2, y2]), 
            multimask_output=False
        )
        single_sam_mask = masks[0]
        
        sam_mask_area = np.sum(single_sam_mask == True)
        if sam_mask_area < 15: continue  # 过滤极小噪声
        
        overlap_count = np.sum(np.logical_and(single_sam_mask, plant_skeleton_2d > 0))
        
        
        # 门槛放低到 0.05 后，如果一个掩码面积占了全图 6% 以上，或者吞了 30% 以上的骨架则丢弃
        if sam_mask_area > (h * w * 0.06) or (total_skeleton_pts > 0 and (overlap_count / total_skeleton_pts) > 0.30):
            continue
            
        if overlap_count >= 5: 
            intersecting_pixels = instance_label_mask[single_sam_mask == True]
            existing_ids = intersecting_pixels[intersecting_pixels > 0]
            
            assigned_id = None
            
            if len(existing_ids) > 0:
                ids, counts = np.unique(existing_ids, return_counts=True)
                major_id_idx = np.argmax(counts)
                dominant_id = ids[major_id_idx]
                max_overlap_pixels = counts[major_id_idx]
                
                overlap_ratio = max_overlap_pixels / (sam_mask_area + 1e-6)
                
                # 如果这个新框有 35% 以上的面积和某个旧 ID 重叠，判定为同一个个体的延伸
                if overlap_ratio > 0.35:
                    assigned_id = dominant_id  # 记录要继承的旧 ID，不执行 continue
                    
                # 跨物体粘连阻断：如果一个框跨越了多个不同物体，且没有绝对大比例重叠则去除
                elif len(ids) >= 2 and overlap_ratio < 0.25:
                    continue
            
            if assigned_id is not None:
               
                instance_label_mask[single_sam_mask == True] = assigned_id
            else:
                # 如果是一个全新的独立框，开辟新 ID 涂色
                target_pixels = (single_sam_mask == True) & (instance_label_mask == 0)
                if np.sum(target_pixels) > 10:
                    instance_label_mask[target_pixels] = leaf_id_counter
                    leaf_id_counter += 1
                
    # 生成对齐的掩膜图
    out_mask_name = img_name.replace(".bmp", "_leaf.png").replace(".jpg", "_leaf.png").replace(".png", "_leaf.png")
    cv2.imwrite(os.path.join(OUTPUT_LEAF_DIR, out_mask_name), instance_label_mask)
    
    if (idx_img + 1) % 10 == 0 or (idx_img + 1) == len(img_names):
        print(f"📸 进度: [{idx_img + 1}/{len(img_names)}] 视图 {img_name} 已通过熔断技术提纯出 {leaf_id_counter - 1} 个精细隔离体")

print("\n🏁 [2D 阶段完美收官]：恶性粘连掩码已被彻底轰碎，精细麦穗特征现已成型！")