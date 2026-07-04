import os
import sys
import numpy as np
import cv2
import pycolmap
import pandas as pd
from plyfile import PlyData, PlyElement
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# =======================================================================
#  基础数据与路径加载
# =======================================================================
MASK_DIR = "/root/autodl-tmp/Grounded-Segment-Anything-main/plant/plant_002/leaf_masks"
PLY_PATH = "/root/autodl-tmp/2/segmentData_plant/plant_002.ply"
SPARSE_DIR = "/root/autodl-tmp/Grounded-Segment-Anything-main/plant/plant_002/sparse/0"
OUTPUT_PLY = "/root/autodl-tmp/Grounded-Segment-Anything-main/pointcloud/plant2/plant_002_INSTANCES_99.ply"

print("加载 3D 原始植物点云...", flush=True)
plydata = PlyData.read(PLY_PATH)
vertex = plydata["vertex"]
points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
N = len(points)

print("加载 COLMAP 相机内外参...", flush=True)
rec = pycolmap.Reconstruction(SPARSE_DIR)

# 稳健提取 mask 文件，兼容扩展名大小写
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
num_views = len(mask_files)
print(f"共发现 {num_views} 个 2D Mask 文件。", flush=True)

colmap_images = {im.name: im for im in rec.images.values()}
print(f" COLMAP 中共注册了 {len(colmap_images)} 个相机视角。", flush=True)

# =======================================================================

view_count = np.zeros(N, dtype=np.float32)
hit_count = np.zeros(N, dtype=np.float32)
point_view_labels = np.zeros((N, num_views), dtype=np.int32)

DEPTH_EPS = 0.012  

print(f" Z-Buffer ", flush=True)
valid_processed_views = 0

for vid, mask_name in enumerate(mask_files):
    mask_path = os.path.join(MASK_DIR, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: 
        print(f"无法读取图像 {mask_name}", flush=True)
        continue
    h, w = mask.shape

    # 统一名称
    mask_base = mask_name.lower().replace("_leaf", "").split(".")[0]
    
    img = None
    for colmap_name, im_obj in colmap_images.items():
        colmap_base = colmap_name.lower().split(".")[0]
        # 只要有一方包含另一方，或者基准名完全一致，直接握手
        if (mask_base in colmap_base) or (colmap_base in mask_base):
            img = im_obj
            break
            
    if img is None: 
        if vid < 3: 
            print(f"名字对齐失败！Mask名: '{mask_name}'(处理后:'{mask_base}') 找不到对应的 COLMAP 注册图像。", flush=True)
        continue

    valid_processed_views += 1

    try:
        pose = img.cam_from_world()
        R, T = pose.rotation.matrix(), pose.translation
    except:
        R, T = img.rotmat(), img.tvec

    cam = rec.cameras[img.camera_id]
    K = cam.calibration_matrix()

    cam_xyz = (R @ points.T).T + T
    z = cam_xyz[:, 2:3]
    z[z == 0] = 1e-6

    proj = (K @ cam_xyz.T).T
    proj_xy = proj[:, :2] / z
    px, py = proj_xy[:, 0].astype(np.int32), proj_xy[:, 1].astype(np.int32)

    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (cam_xyz[:, 2] > 0)
    idxs = np.where(valid)[0]
    if len(idxs) == 0: continue

    df = pd.DataFrame({'idx': idxs, 'x': px[idxs], 'y': py[idxs], 'd': cam_xyz[idxs, 2]})
    df_sorted = df.sort_values(by=['y', 'x', 'd'])
    df_min = df_sorted.drop_duplicates(subset=['y', 'x'], keep='first')
    
    depth_buffer = np.full((h, w), np.inf)
    depth_buffer[df_min['y'].values, df_min['x'].values] = df_min['d'].values

    view_count[idxs] += 1
    
    front_mask = abs(cam_xyz[idxs, 2] - depth_buffer[py[idxs], px[idxs]]) < DEPTH_EPS
    hit_pixels = mask[py[idxs], px[idxs]] > 0
    hit_idxs = idxs[front_mask & hit_pixels]
    hit_count[hit_idxs] += 1
    
    point_view_labels[idxs[front_mask], vid] = mask[py[idxs[front_mask]], px[idxs[front_mask]]]

print(f"视角扫描完成，共 {valid_processed_views} / {num_views} 个有效视角！", flush=True)

plant_mask = (hit_count >= 3) | (((hit_count / (view_count + 1e-6)) > 0.28) & (hit_count >= 1))
print(f"植物实体点共: {np.sum(plant_mask)} / {N}", flush=True)


# =======================================================================
# 空间几何超点切分
# =======================================================================
print("几何超点空间切分...", flush=True)
num_superpoints = 550  
kmeans = MiniBatchKMeans(n_clusters=num_superpoints, batch_size=4096, random_state=42, n_init=3)

superpoint_labels = np.full(N, -1, dtype=np.int32)
plant_idxs = np.where(plant_mask)[0]

if len(plant_idxs) == 0:
    print("错误：未识别到有效植物点。")
    sys.exit()

lebeled_sp_subset = kmeans.fit_predict(points[plant_idxs])
superpoint_labels[plant_idxs] = lebeled_sp_subset

sp_centroids = np.zeros((num_superpoints, 3))
active_sp_list = []
for i in range(num_superpoints):
    sp_pt_indices = np.where(superpoint_labels == i)[0]
    if len(sp_pt_indices) > 0:
        sp_centroids[i] = np.mean(points[sp_pt_indices], axis=0)
        active_sp_list.append(i)
    else:
        sp_centroids[i] = np.array([999.0, 999.0, 999.0])

# =======================================================================
# 提纯超点多视角标签
# =======================================================================
print("提纯超点多视角标签", flush=True)
superpoint_view_masks = np.zeros((num_superpoints, num_views), dtype=np.int32)
superpoint_confidence = np.zeros(num_superpoints, dtype=np.float32) 

for sp_idx in active_sp_list:
    sp_pt_indices = np.where(superpoint_labels == sp_idx)[0]
    # 计算当前超点块整体踩中前景的平均比率
    superpoint_confidence[sp_idx] = np.mean(hit_count[sp_pt_indices] / (view_count[sp_pt_indices] + 1e-6))
    
    for vid in range(num_views):
        labels_in_view = point_view_labels[sp_pt_indices, vid]
        non_zero_labels = labels_in_view[labels_in_view > 0]
        if len(non_zero_labels) > 0:
            superpoint_view_masks[sp_idx, vid] = np.argmax(np.bincount(non_zero_labels))

# =======================================================================
#  构建拓扑连通图
# =======================================================================
print("构建 3D 局部空间紧密拓扑连通图...", flush=True)
adj_matrix = np.zeros((num_superpoints, num_superpoints), dtype=np.int32)
# 维持在 4.0 厘米。既保证长叶身前后顺畅连通，又防止跨空腔粘连散点
for i in active_sp_list:
    for j in active_sp_list:
        if i >= j: continue
        if np.linalg.norm(sp_centroids[i] - sp_centroids[j]) < 0.040:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
connectivity = csr_matrix(adj_matrix)

# =======================================================================
# 动态平滑阻抗矩阵
# =======================================================================
distance_matrix = np.ones((num_superpoints, num_superpoints)) * 0.40

for i in active_sp_list:
    distance_matrix[i, i] = 0.0
    mask_i = superpoint_view_masks[i]
    valid_i = (mask_i > 0)
    
    for j in active_sp_list:
        if i >= j: continue
        mask_j = superpoint_view_masks[j]
        valid_j = (mask_j > 0)
        
        common_views = valid_i & valid_j
        sum_common = np.sum(common_views)
        
        if sum_common > 0:
            same_votes = np.sum(mask_i[common_views] == mask_j[common_views])
            agree_ratio = same_votes / sum_common
            
            if agree_ratio >= 0.65:
                dist = 0.02 * (1.0 - agree_ratio)
            else:
                dist = 50.0  # 有标签冲突
        else:
           
            if superpoint_confidence[i] < 0.25 or superpoint_confidence[j] < 0.25:
                dist = 2.0 
            else:
                dist = 0.25 # 只有两个都是高置信度实体叶片块，才允许盲区拓扑拼合
                
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

# =======================================================================
# 层次凝聚聚类器
# =======================================================================
try:
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.60, connectivity=connectivity)
    sp_clusters = clustering.fit_predict(distance_matrix)
except TypeError:
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.60, connectivity=connectivity)
    sp_clusters = clustering.fit_predict(distance_matrix)

final_labels = np.zeros(N, dtype=np.int32)
for sp_idx, cluster_id in enumerate(sp_clusters):
    if sp_idx not in active_sp_list: continue
    sp_pt_indices = np.where(superpoint_labels == sp_idx)[0]
    final_labels[sp_pt_indices] = cluster_id + 1

# =======================================================================
# 三维欧氏连通域分析
# =======================================================================
unique_clusters = np.unique(final_labels)
cleaned_labels = np.zeros(N, dtype=np.int32)
real_leaf_counter = 1

for c_id in unique_clusters:
    if c_id == 0: continue
    pt_indices = np.where(final_labels == c_id)[0]
    if len(pt_indices) < 150: continue  # 过滤过小块
    
    # 提取当前叶片实例内部所有 3D 点
    cluster_points = points[pt_indices]
    
    # 用半径为 3 厘米的近邻图分析内部物理连通性
    nbrs = NearestNeighbors(radius=0.03).fit(cluster_points)
    adj_graph = nbrs.radius_neighbors_graph(cluster_points, mode='connectivity')
    
    from scipy.sparse.csgraph import connected_components
    n_components, labels_components = connected_components(csgraph=adj_graph, directed=False)
    
    if n_components > 1:
        # 找出最大的那个连通体
        component_sizes = np.bincount(labels_components)
        largest_component_id = np.argmax(component_sizes)
        
        # 仅仅保留最大主体的点
        valid_sub_indices = np.where(labels_components == largest_component_id)[0]
        actual_global_indices = pt_indices[valid_sub_indices]
    else:
        actual_global_indices = pt_indices

    if len(actual_global_indices) > 150:
        cleaned_labels[actual_global_indices] = real_leaf_counter
        real_leaf_counter += 1

print(f"剔除背景散点，共 {real_leaf_counter - 1} 片完美叶片实体")

# =======================================================================
#  写入高色差 3D 实例 PLY 文件
# =======================================================================
print(" 正在写入 PLY 文件...")
colors = np.zeros((N, 3), dtype=np.uint8)
np.random.seed(1314) 
palette = np.random.randint(40, 255, (real_leaf_counter + 5, 3))
colors = palette[cleaned_labels]
colors[cleaned_labels == 0] = [105, 105, 105]  # 噪声和被剥离的空间散点全部强制灰色

vertex_data = np.zeros(N, dtype=[
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("label", "i4")
])

vertex_data["x"], vertex_data["y"], vertex_data["z"] = points[:, 0], points[:, 1], points[:, 2]
vertex_data["red"], vertex_data["green"], vertex_data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
vertex_data["label"] = cleaned_labels

el = PlyElement.describe(vertex_data, "vertex")
PlyData([el], text=False).write(OUTPUT_PLY)

print(f" PLY 已生成：\n {OUTPUT_PLY}\n")