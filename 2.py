import numpy as np

def count_points_in_npz(file_path, dataset_name='xyz'):
    """
    统计NPZ文件中点云数据点的数目
    
    参数:
    file_path (str): NPZ文件路径
    dataset_name (str): 包含点云坐标的数据集名称 (默认为'xyz')
    """
    try:
        # 加载NPZ文件
        data = np.load(file_path)
        
        # 检查数据集是否存在
        if dataset_name not in data:
            available_datasets = list(data.keys())
            print(f"错误: 数据集 '{dataset_name}' 不存在于文件中。")
            print(f"文件中可用的数据集: {available_datasets}")
            return
        
        # 获取点云数据集
        point_cloud = data[dataset_name]
        
        # 确保是点云数据（至少二维，第二维为2或3）
        if point_cloud.ndim < 2 or point_cloud.shape[1] not in [2, 3, 4]:
            print(f"警告: 数据集 '{dataset_name}' 的形状为 {point_cloud.shape}，可能不是标准点云数据")
        
        # 计算并输出点云数目
        num_points = point_cloud.shape[0]
        print(f"文件: {file_path}")
        print(f"数据集: {dataset_name}")
        print(f"点云数据点数目: {num_points}")
        
        return num_points
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None

if __name__ == "__main__":
    # 文件路径
    #file_path = "filtered_pointcloud.npz"
    file_path = "filtered_pointcloud.npz"
    # 统计点云数目（默认使用'xyz'数据集）
    count_points_in_npz(file_path)