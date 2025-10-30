# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import os.path as op
import torchvision.transforms as T
from torchvision.transforms.functional import normalize 
from PIL import Image, ImageOps
import time
import pandas as pd
from tqdm import tqdm
# --- 从 IRRA 项目导入必要的模块 ---
from utils.iotools import load_train_configs
# 根据你的数据集导入对应的类
from datasets.cuhkpedes import CUHKPEDES
# 如果需要支持其他数据集，取消注释或添加对应的导入
# from datasets.icfgpedes import ICFGPEDES
# from datasets.rstpreid import RSTPReid

# === 全局配置和路径 ===

# 1. 指定包含 attack_mask.csv 和 defend_mask.csv 的完整路径
ATTACK_MASK_CSV_PATH = '/home/sanmuzzzzz/IRRA/data/CUHK-PEDES/mask_bicubic/attack_mask/attack_mask.csv'
DEFEND_MASK_CSV_PATH = '~/IRRA/data/CUHK-PEDES/mask_bicubic/defend_mask/defend_mask.csv'

# 2. 指定存放 攻击扰动 图像文件的目录 
ATTACK_MASK_IMG_DIR = '~/IRRA/data/CUHK-PEDES/mask_bicubic/attack_mask/'

# 3. 指定存放 防御扰动 图像文件的目录
DEFEND_MASK_IMG_DIR = '~/IRRA/data/CUHK-PEDES/mask_bicubic/defend_mask/'

# 4. 指定 IRRA 训练配置文件路径 (用于加载数据集根目录等)
CONFIG_FILE = 'logs/CUHK-PEDES/irra_cuhk/configs.yaml' 

# 5. 指定保存【攻击后】图像的输出根目录
OUTPUT_ATTACK_DIR = '~/IRRA/data/CUHK-PEDES/perturbed_images/attack/' 

# 6. 指定保存【防御后】图像的输出根目录
OUTPUT_DEFEND_DIR = '~/IRRA/data/CUHK-PEDES/perturbed_images/defend/' 

# 7. 扰动范围 Epsilon (用于将 [0, 255] 的扰动图还原回 delta)
EPSILON = 16

# 8. 目标图像尺寸 (H, W)，应与 IRRA 模型输入一致
TARGET_SIZE = (384, 128)

# 9. 数据集名称，用于从绝对路径提取相对路径作为 key
DATASET_FOLDER_NAME = 'CUHK-PEDES' 


# === 标准化参数 (通常在保存为图像文件时不需要，但如果添加扰动逻辑依赖标准化则需要) ===
# IMG_MEAN = [0.48145466, 0.4578275, 0.40821073]
# IMG_STD = [0.26862954, 0.26130258, 0.27577711]

# === 定义 PyTorch 图像转换 ===
pil_to_tensor = T.ToTensor() # PIL [0, 255] -> Tensor [0, 1]
tensor_to_pil = T.ToPILImage() # Tensor [0, 1] -> PIL [0, 255]

# === 辅助函数：加载扰动映射  ===
def load_mask_map(csv_path):
    """
    从 CSV 加载图像相对路径 (file_path) 到扰动图像绝对路径 (mask_path) 的映射字典。
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Perturbation CSV file not found at {csv_path}. Returning empty map.")
        return {}
    try:
        df = pd.read_csv(csv_path)
        perturbation_map = {}
        missing_perturbation_files = 0

        required_columns = ['file_path', 'mask_path']
        if not all(col in df.columns for col in required_columns):
             print(f"Error: CSV file {csv_path} must contain columns: {required_columns}")
             return {}

        for index, row in df.iterrows():
            relative_img_path_key = row['file_path'].replace('\\', '/')
            perturbation_abs_path = row['mask_path']

            if os.path.exists(perturbation_abs_path):
                perturbation_map[relative_img_path_key] = perturbation_abs_path
            else:
                missing_perturbation_files += 1

        print(f"Loaded {len(perturbation_map)} unique perturbation mappings from {csv_path}.")
        if missing_perturbation_files > 0:
            print(f"Warning: Checked {len(df)} rows, {missing_perturbation_files} referenced perturbation files were not found on disk.")
        return perturbation_map
    except Exception as e:
        print(f"Error loading perturbation CSV {csv_path}: {e}")
        return {}

# === 辅助函数：从绝对路径获取相对路径 Key  ===
def get_relative_path_key(abs_path, dataset_folder_name):
    """
    尝试从绝对路径中提取相对于数据集文件夹的路径作为 key。
    """
    try:
        # 尝试基于 dataset_folder_name 分割
        # 移除可能存在的基础路径部分，直到找到 dataset_folder_name
        parts = abs_path.split(os.path.sep)
        try:
            base_index = parts.index(dataset_folder_name)
            start_index = base_index + 1
            if parts[start_index] == 'imgs': 
                 start_index += 1
            relative_path = os.path.join(*parts[start_index:])
            return relative_path.replace('\\', '/')
        except ValueError:
            print(f"Warning: '{dataset_folder_name}' not found in path '{abs_path}'. Using fallback key extraction.")
            parts_fallback = abs_path.replace('\\', '/').split('/')
            if len(parts_fallback) >= 2 and parts_fallback[-2] in ['Market', 'test_query', 'train_query', 'imgs']:
                # 尝试匹配 CSV 中的常见格式
                if parts_fallback[-2] == 'imgs' and len(parts_fallback) >= 3:
                     return '/'.join(parts_fallback[-2:]) 
                else:
                     return '/'.join(parts_fallback[-2:])
            elif len(parts_fallback) >= 3:
                 return '/'.join(parts_fallback[-3:])
            else:
                 return parts_fallback[-1] # Fallback to filename
    except Exception as e:
        print(f"Error extracting relative path key from {abs_path}: {e}")
        return None

# === 核心函数：处理单张图片 ===
def process_and_save_image(original_img_path, relative_key, perturbation_map, output_dir, eps, target_size):
    """
    加载原图，查找、加载并添加扰动，然后保存结果。
    """
    perturbation_path = perturbation_map.get(relative_key)

    if not perturbation_path:
        # print(f"Skipping: No perturbation found for key {relative_key}") # Optional log
        return False # Indicate skipped

    # 构建输出路径，保持原始的相对目录结构
    output_path = os.path.join(output_dir, relative_key)
    output_subdir = os.path.dirname(output_path)
    os.makedirs(output_subdir, exist_ok=True) # 确保子目录存在

    try:
        # 1. 加载原图和扰动图 (PIL)
        pil_target_size = (target_size[1], target_size[0]) # W, H for PIL resize
        img_original_pil = Image.open(original_img_path).convert('RGB').resize(pil_target_size)
        perturbation_pil = Image.open(perturbation_path).convert('RGB').resize(pil_target_size)

        # 2. 转为 Tensor [0, 1]
        img_original_tensor = pil_to_tensor(img_original_pil)
        perturbation_tensor = pil_to_tensor(perturbation_pil)

        # 3. 计算 delta 并添加
        # 将 [0, 1] 映射到 [-eps/255, eps/255]
        delta_tensor = (perturbation_tensor - 0.5) * (eps / 128.0)
        perturbed_tensor_unclamped = img_original_tensor + delta_tensor

        # 4. 裁剪到 [0, 1]
        perturbed_tensor_clipped = torch.clamp(perturbed_tensor_unclamped, 0.0, 1.0)

        # 5. 转回 PIL 图像
        perturbed_pil = tensor_to_pil(perturbed_tensor_clipped)

        # 6. 保存图像
        perturbed_pil.save(output_path)
        return True # Indicate success

    except FileNotFoundError:
         print(f"Error: Original image not found at {original_img_path} (referenced by key {relative_key})")
         return False
    except Exception as e:
        print(f"Error processing image {original_img_path} with perturbation {perturbation_path}: {e}")
        return False # Indicate failure

# === 主执行块 ===
if __name__ == '__main__':
    print("--- Starting Perturbed Image Generation ---")

    # --- 加载配置 ---
    print(f"Loading config file: {CONFIG_FILE}")
    if not os.path.exists(CONFIG_FILE):
         print(f"Error: Config file not found at {CONFIG_FILE}.")
         exit(1)
    args = load_train_configs(CONFIG_FILE)
    print("Config loaded.")

    # --- 加载数据集元数据 ---
    print(f"Loading dataset metadata: {args.dataset_name} from root: {args.root_dir}")
    try:
        if args.dataset_name == 'CUHK-PEDES':
            dataset = CUHKPEDES(root=args.root_dir, verbose=False)
        # elif args.dataset_name == 'ICFG-PEDES':
        #     dataset = ICFGPEDES(root=args.root_dir, verbose=False)
        # elif args.dataset_name == 'RSTPReid':
        #     dataset = RSTPReid(root=args.root_dir, verbose=False)
        else:
             print(f"Error: Unsupported dataset name '{args.dataset_name}' for metadata loading.")
             exit(1)

        # 获取测试集图像绝对路径列表
        if 'test' in dir(dataset) and isinstance(dataset.test, dict) and 'img_paths' in dataset.test:
            original_image_paths = dataset.test['img_paths']
            if not original_image_paths:
                 print("Error: Dataset's test img_paths list is empty.")
                 exit(1)
            print(f"Found {len(original_image_paths)} images in the test set.")
        else:
             print("Error: Could not retrieve 'img_paths' from dataset.test dictionary.")
             exit(1)

    except Exception as e:
        print(f"Error loading dataset object: {e}")
        exit(1)

    # --- 加载扰动映射 ---
    print("Loading perturbation mappings...")
    attack_perturbation_map = load_mask_map(ATTACK_MASK_CSV_PATH)
    defend_perturbation_map = load_mask_map(DEFEND_MASK_CSV_PATH)
    print(f"Loaded {len(attack_perturbation_map)} attack and {len(defend_perturbation_map)} defend perturbation mappings.")

    # --- 创建输出目录 ---
    print(f"Ensuring output directory exists: {OUTPUT_ATTACK_DIR}")
    os.makedirs(OUTPUT_ATTACK_DIR, exist_ok=True)
    print(f"Ensuring output directory exists: {OUTPUT_DEFEND_DIR}")
    os.makedirs(OUTPUT_DEFEND_DIR, exist_ok=True)

    # --- 遍历并处理图像 ---
    print("\nProcessing images...")
    attack_success_count = 0
    defend_success_count = 0
    skipped_count = 0
    error_count = 0

    # 使用 tqdm 显示进度条
    for original_path in tqdm(original_image_paths, desc="Generating perturbed images"):
        relative_key = get_relative_path_key(original_path, DATASET_FOLDER_NAME)

        if not relative_key:
            print(f"Warning: Could not get relative key for {original_path}. Skipping.")
            skipped_count += 1
            continue

        # 处理攻击扰动
        if relative_key in attack_perturbation_map:
            success = process_and_save_image(
                original_path, relative_key, attack_perturbation_map,
                OUTPUT_ATTACK_DIR, EPSILON, TARGET_SIZE
            )
            if success: attack_success_count += 1
            else: error_count += 1
        else:
            skipped_count += 1 # Count skips where no mapping exists

        # 处理防御扰动
        if relative_key in defend_perturbation_map:
            success = process_and_save_image(
                original_path, relative_key, defend_perturbation_map,
                OUTPUT_DEFEND_DIR, EPSILON, TARGET_SIZE
            )
            if success: defend_success_count += 1
            else: error_count += 1
        # No else needed here for skipping count, already counted if attack was skipped

    print("\n--- Generation Summary ---")
    print(f"Total original images processed: {len(original_image_paths)}")
    print(f"Successfully generated attack images: {attack_success_count}")
    print(f"Successfully generated defend images: {defend_success_count}")
    # Note: skipped_count might double count if an image lacks both attack and defend perturbations.
    # A more precise skip count would track unique keys skipped.
    print(f"Images skipped (no perturbation mapping found): approx. {skipped_count // 2 if skipped_count > 0 else 0}")
    print(f"Errors during processing: {error_count}")
    print(f"Attack images saved to: {OUTPUT_ATTACK_DIR}")
    print(f"Defend images saved to: {OUTPUT_DEFEND_DIR}")
    print("-------------------------")