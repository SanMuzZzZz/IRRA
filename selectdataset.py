# coding=utf-8
import pandas as pd
import shutil
from pathlib import Path
import os
import sys

# --- (1) 配置区域：请在此处修改您的参数 ---

CONFIG = {
    # 1. 原始完整掩膜CSV的路径
    "ORIGINAL_CSV_FILE": "/home/sanmuzzzzz/Hack/Research/IRRA/data/CUHK-PEDES/mask111/mask/defend_mask/updated_data2.csv",
    
    # 2. 【新】掩膜图像的【目标文件夹】路径 (脚本将自动创建)
    "TARGET_IMAGE_FOLDER": "/home/sanmuzzzzz/Hack/Research/IRRA/data/CUHK-PEDES/defend_mask_N200_images/",
    
    # 3. 【新】CSV文件的【输出路径】(包含文件名)
    "NEW_CSV_OUTPUT_PATH": "/home/sanmuzzzzz/Hack/Research/IRRA/data/CUHK-PEDES/defend_mask_N200_images/defend_mask_N200.csv",

    # 4. 'mask_path' 所在的【列索引】(0-based, 默认为 2，即第3列)
    # (参考: select.py)
    "MASK_PATH_COLUMN_INDEX": 2, 
    
    # 5. 选择行的方式 (二选一)
    
    # 方式A: 选择前 N 行
    #"SELECT_N_ROWS": 100, 
    
    # 方式B: 选择一个切片 (如果使用方式A，请忽略此项)
    "SELECT_START_ROW": 0,
    "SELECT_END_ROW": 200, 
}

# ----------------------------------------------------


def create_mask_subset(config):
    """
    非交互式脚本：
    1. 根据 CONFIG 读取原始CSV。
    2. 选择指定行 (例如前N行)。
    3. 复制掩膜图像到新的目标文件夹。
    4. 创建一个新的CSV，并更新其中的 'mask_path' 指向新位置。
    """
    try:
        # --- 1. 验证配置 ---
        original_csv = Path(config["ORIGINAL_CSV_FILE"])
        target_img_folder = Path(config["TARGET_IMAGE_FOLDER"])
        new_csv_output = Path(config["NEW_CSV_OUTPUT_PATH"])
        col_index = config["MASK_PATH_COLUMN_INDEX"]

        if not original_csv.exists():
            print(f"错误: 原始CSV文件未找到: {original_csv}")
            sys.exit(1)

        os.makedirs(target_img_folder, exist_ok=True)
        print(f"目标图像文件夹已确认: {target_img_folder}")

        # --- 2. 读取和选择行 ---
        print(f"正在读取原始CSV: {original_csv}...")
        try:
            df_full = pd.read_csv(original_csv, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 解码失败，尝试 GBK...")
            df_full = pd.read_csv(original_csv, encoding='gbk')

        mask_col_name = df_full.columns[col_index]
        print(f"使用列 '{mask_col_name}' (索引 {col_index}) 作为掩膜路径。")

        # 根据配置选择行
        if "SELECT_N_ROWS" in config:
            start_row = 0
            end_row = config["SELECT_N_ROWS"]
        elif "SELECT_START_ROW" in config and "SELECT_END_ROW" in config:
            start_row = config["SELECT_START_ROW"]
            end_row = config["SELECT_END_ROW"]
        else:
            print("错误: 未在 CONFIG 中指定 'SELECT_N_ROWS' 或 'SELECT_START_ROW/END_ROW'")
            sys.exit(1)
            
        df_subset = df_full.iloc[start_row:end_row].copy()
        
        print(f"\n已选择 {len(df_subset)} 行 (从 {start_row} 到 {end_row}) 进行处理...")

        # --- 3. 复制文件并更新路径 ---
        success_count = 0
        missing_count = 0
        error_count = 0
        new_rows_list = [] # 存储修改后的行

        for index, row in df_subset.iterrows():
            try:
                # 原始的绝对路径
                original_mask_path_str = str(row[mask_col_name])
                original_mask_path = Path(original_mask_path_str)
                
                # 提取文件名
                image_name = original_mask_path.name
                
                # 构建新的目标路径
                target_path = target_img_folder / image_name
                
                if original_mask_path.exists():
                    # 复制文件
                    shutil.copy2(original_mask_path, target_path)
                    
                    # (关键步骤) 更新行中的 mask_path 为新路径
                    # 使用 .as_posix() 确保路径使用 '/'
                    row[mask_col_name] = target_path.as_posix() 
                    
                    new_rows_list.append(row)
                    success_count += 1
                else:
                    missing_count += 1
                    if missing_count <= 10:
                        print(f"警告: 源文件不存在(跳过): {original_mask_path_str}")
                        
            except Exception as e:
                error_count += 1
                print(f"处理第 {index} 行时出错: {e}")

        # --- 4. 保存新的CSV文件 ---
        if new_rows_list:
            df_new = pd.DataFrame(new_rows_list, columns=df_subset.columns)
            df_new.to_csv(new_csv_output, index=False, encoding='utf-8')
            print(f"\n成功创建新的CSV文件: {new_csv_output}")
        else:
            print("\n警告: 没有成功处理任何文件，未创建新的CSV。")

        # --- 5. 打印总结 ---
        print(f"\n{'='*50}")
        print(f"子集创建完成！")
        print(f"成功处理和复制: {success_count} 个文件")
        print(f"源文件缺失: {missing_count} 个")
        print(f"处理错误: {error_count} 个")
        if missing_count > 10:
            print(f"(已隐藏 {missing_count-10} 个缺失文件提示)")
        print(f"{'='*50}")

    except Exception as e:
        print(f"发生严重错误: {e}")

# --- 主执行块 ---
if __name__ == '__main__':
    create_mask_subset(CONFIG)