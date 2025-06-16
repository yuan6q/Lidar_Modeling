import numpy as np
import os

# npz_file_path = "your_file.npz"  # 替换成您的npz文件路径
npz_file_path = "cube_pointclouds/0.npz"  # 示例

if not os.path.exists(npz_file_path):
    raise FileNotFoundError(f"File not found: {npz_file_path}")

# 读取npz文件
print(f"\nReading file: {npz_file_path}")
data = np.load(npz_file_path)

# 显示所有字段名和前5个数据
print("\nField names and first 5 values:")
for key in data.files:
    values = data[key]
    print(f"\nField: '{key}'")
    print("Shape:", values.shape)
    print("First 5 entries:")
    print(values[:5])
