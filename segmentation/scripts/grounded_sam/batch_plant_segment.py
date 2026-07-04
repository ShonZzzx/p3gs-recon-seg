import os
import sys

# 禁止任何联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 修复某些环境下 OpenCV 多线程死锁或崩溃的问题
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("=== [DEBUG] 脚本已成功启动 ===", flush=True)
import cv2
import torch
import numpy as np
from pathlib import Path

import cv2
print("=== [DEBUG] cv2 导入成功 ===", flush=True)

import torch
print(f"=== [DEBUG] torch 导入成功，CUDA 是否可用: {torch.cuda.is_available()} ===", flush=True)

import numpy as np
from pathlib import Path

print("=== [DEBUG] 正在尝试从 GroundingDINO 导入组件... ===", flush=True)
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor
print("=== [DEBUG] 所有核心库导入成功！ ===", flush=True)

# ==================== 配置路径与参数 ====================
IMAGE_DIR = "/root/autodl-tmp/Grounded-Segment-Anything-main/plant/plant_020/images"     
OUTPUT_DIR = "/root/autodl-tmp/Grounded-Segment-Anything-main/plant/plant_020/masks"      

# 参数
TEXT_PROMPT = "plant . plant stems"
BOX_THRESHOLD = 0.10   
TEXT_THRESHOLD = 0.10           
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "weights/groundingdino_swint_ogc.pth"
SAM_WEIGHTS = "weights/sam_vit_h_4b8939.pth"
# =======================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 检查路径下有没有图片
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]
image_paths = [p for p in Path(IMAGE_DIR).glob("**/*") if p.suffix in valid_extensions]
print(f"=== [DEBUG] 目标路径: {IMAGE_DIR}，共找到图片数量: {len(image_paths)} ===", flush=True)

if len(image_paths) == 0:
    print(" 错误：在指定路径下没有找到任何有效格式的图片！请检查路径或是否成功解压！", flush=True)
    sys.exit()

print("正在加载 GroundingDINO 模型...", flush=True)

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

#  加载配置文件
args = SLConfig.fromfile(DINO_CONFIG)
args.device = DEVICE


args.text_encoder_type = "bert-base-uncased" 
args.bert_base_uncased_path = "/root/autodl-tmp/Grounded-Segment-Anything-main/weights/bert-base-uncased"

#  构建模型并加载权重
dino_model = build_model(args)
checkpoint = torch.load(DINO_WEIGHTS, map_location="cpu")
dino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
dino_model.eval()
dino_model = dino_model.to(DEVICE)

print("正在加载 SAM 模型...", flush=True)
sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

print("开始生成 2D Mask...", flush=True)


for img_path in image_paths:
    print(f"正在处理: {img_path.name}", flush=True)
    
    # 运行 GroundingDINO 抓取选区
    image_source, image_transformed = load_image(str(img_path))
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_transformed,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )
    
    # 如果完全没检测到边界框，直接输出纯黑掩码
    if boxes.shape[0] == 0:
        black_mask = np.zeros(image_source.shape[:2], dtype=np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_path.stem}_mask.png"), black_mask)
        continue
        
    H, W, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([W, H, W, H])
    boxes_xyxy = boxes_unnorm.clone()
    boxes_xyxy[:, :2] -= boxes_unnorm[:, 2:] / 2
    boxes_xyxy[:, 2:] += boxes_unnorm[:, 2:] / 2
    
    # 将选区送给 SAM
    sam_predictor.set_image(image_source)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    
    # 开启 multimask_output=True，让 SAM 吐出 3 种不同的抠图解法
    masks, scores, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True, 
    )
    
    # 绿色分析与掩码筛选
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    
    # 将原图转换到 HSV 色彩空间，用来定位“绿色”
    hsv = cv2.cvtColor(image_source, cv2.COLOR_BGR2HSV)
    # 定义通用的植物绿色范围
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([85, 255, 255])
    green_pixels_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    for i in range(masks.shape[0]):
        
        box_masks = masks[i] 
        
        best_mask_np = None
        max_green_ratio = -1
        
        # 依次检查 SAM 提供的 3 种颗粒度的抠图结果
        for m_idx in range(3):
            candidate_mask_np = box_masks[m_idx].cpu().numpy().astype(np.uint8)
            
           
            total_pixels = np.sum(candidate_mask_np)
            if total_pixels == 0:
                continue
                
           
            green_in_mask = np.sum(cv2.bitwise_and(candidate_mask_np, green_pixels_mask))
            green_ratio = green_in_mask / total_pixels  # 绿色纯度占比
            
           
            if green_ratio > max_green_ratio:
                max_green_ratio = green_ratio
                best_mask_np = candidate_mask_np
        
        # 如果通过颜色找到了最好的植物掩码，将其融合进来
        if best_mask_np is not None and max_green_ratio > 0.1: 
            combined_mask = cv2.bitwise_or(combined_mask, best_mask_np)
        else:
            # 如果用颜色过滤完发现都不达标（比如植物本身不是绿色的），则退回使用 SAM 分数最高的那层默认掩码
            best_score_idx = torch.argmax(scores[i]).item()
            fallback_mask = box_masks[best_score_idx].cpu().numpy().astype(np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, fallback_mask)
            
    # 保存最终完美的黑白 Mask 图
    output_mask_path = os.path.join(OUTPUT_DIR, f"{img_path.stem}_mask.png")
    cv2.imwrite(output_mask_path, combined_mask * 255)

print(f"所有 2D Mask 已保存在 {OUTPUT_DIR}", flush=True)