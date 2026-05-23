import os
import subprocess
import sys
import shutil

def run_command(command, description="正在执行任务"):
    """安全运行系统命令并实时打印输出"""
    print(f"\n{description}...")
    print(f"运行命令: {command}")
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()
    
    if process.returncode != 0:
        print(f"错误: {description} 失败，退出码: {process.returncode}")
        sys.exit(1)
    print(f"{description} 完成！")

def preprocess_scene(plant_path):
    """
    使用3DGS标准默认参数生成 sparse 数据
    plant_path: 场景文件夹路径，里面必须包含 images/ 文件夹
    """
    # 1. 检查输入路径
    images_dir = os.path.join(plant_path, "images")
    if not os.path.exists(images_dir):
        print(f"错误: 在 {plant_path} 下找不到 images 文件夹，请检查路径是否正确！")
        return

    # 2. 创建工作区和标准 sparse/0 目录
    colmap_db = os.path.join(plant_path, "database.db")
    sparse_dir = os.path.join(plant_path, "sparse")
    output_model_dir = os.path.join(sparse_dir, "0")
    
    # 清理旧的残留文件
    if os.path.exists(colmap_db): os.remove(colmap_db)
    if os.path.exists(sparse_dir): shutil.rmtree(sparse_dir)
    os.makedirs(output_model_dir, exist_ok=True)

    print(f"=== 开始标准3DGS预处理场景: {os.path.basename(plant_path)} ===")

    # 步骤一：特征提取
    extract_cmd = (
        f"colmap feature_extractor "
        f"--database_path {colmap_db} "
        f"--image_path {images_dir}"
    )
    run_command(extract_cmd, "步骤 1/4: 提取图像特征点")

    # 步骤二：穷举匹配（ps：这里用顺序匹配也可以，因为images文件夹中的图片是顺序拍摄的，使用顺序匹配可以节省运行内存并且减少匹配时间）
    match_cmd = f"colmap exhaustive_matcher --database_path {colmap_db}"
    run_command(match_cmd, "步骤 2/4: 进行特征点匹配")

    # 步骤三：稀疏重建
    mapper_cmd = (
        f"colmap mapper "
        f"--database_path {colmap_db} "
        f"--image_path {images_dir} "
        f"--output_path {sparse_dir}"
    )
    run_command(mapper_cmd, "步骤 3/4: 三维稀疏重建")

    # 步骤四：整理目录结构归类到 sparse/0
    generated_model_dir = os.path.join(sparse_dir, "0")
    if not os.path.exists(os.path.join(generated_model_dir, "cameras.bin")):
        for file in ["cameras.bin", "images.bin", "points3D.bin"]:
            src = os.path.join(sparse_dir, file)
            dst = os.path.join(output_model_dir, file)
            if os.path.exists(src):
                shutil.move(src, dst)

    # 步骤五：清理临时数据库文件
    if os.path.exists(colmap_db):
        os.remove(colmap_db)

    print(f"\n预处理完成！标准3DGS格式已存入: {output_model_dir}\n")

if __name__ == "__main__":
    # 输入路径
    user_input = input("请输入植物场景文件夹的路径 (例如 datasets/plant_003): ").strip()
    
    if user_input:
        preprocess_scene(user_input)
    else:
        print("错误：路径不能为空！")
