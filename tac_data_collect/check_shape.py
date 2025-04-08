import numpy as np
import os

# 定义文件路径
file_path = "real_tac_data/0001/00000.npy"

# 检查文件是否存在
if os.path.exists(file_path):
    # 加载 .npy 文件
    data = np.load(file_path)
    
    # 获取数据的 shape
    print(f"The shape of the data in {file_path} is: {data.shape}")
else:
    print(f"Error: File '{file_path}' does not exist.")