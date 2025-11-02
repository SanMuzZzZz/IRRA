import pandas as pd

# --- 请根据您的需求修改以下变量 ---

# 1. 输入文件名
input_file = 'updated_data.csv'

# 2. 要修改的列名
column_to_modify = 'file_path' 

# 3. 您想添加的前缀
prefix_to_add = 'imgs/'

# 4. 输出文件名
output_file = 'updated_data1.csv'

# -------------------------------------

try:
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查列是否存在
    if column_to_modify in df.columns:
        # 添加前缀
        # (确保该列为字符串类型，以防万一)
        df[column_to_modify] = prefix_to_add + df[column_to_modify].astype(str)
        
        # 保存到新的CSV文件
        df.to_csv(output_file, index=False)
        
        print(f"处理完成！已将前缀 '{prefix_to_add}' 添加到列 '{column_to_modify}'。")
        print(f"结果已保存至: {output_file}")
        
    else:
        print(f"错误：在文件 '{input_file}' 中未找到列 '{column_to_modify}'。")

except FileNotFoundError:
    print(f"错误：未找到输入文件 '{input_file}'。")
except Exception as e:
    print(f"发生了一个错误: {e}")