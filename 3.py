import numpy as np

def count_cluster_points(file_path):
    """
    统计文件中两类点云的个数
    
    参数:
    file_path (str): NPZ文件路径
    """
    try:
        # 加载NPZ文件
        data = np.load(file_path)
        
        # 获取所有数据集名称
        datasets = list(data.keys())
        
        # 检查cluster1和cluster2是否存在
        cluster1_exists = any(d.startswith('cluster1_') for d in datasets)
        cluster2_exists = any(d.startswith('cluster2_') for d in datasets)
        
        if not cluster1_exists:
            print("错误: 文件中未找到cluster1数据集")
        if not cluster2_exists:
            print("错误: 文件中未找到cluster2数据集")
        if not cluster1_exists or not cluster2_exists:
            print(f"文件中可用的数据集: {datasets}")
            return
        
        # 统计cluster1的点云个数
        cluster1_points = 0
        for dataset in datasets:
            if dataset.startswith('cluster1_') and dataset.endswith('_xyz'):
                cluster1_points = data[dataset].shape[0]
                break
        
        # 统计cluster2的点云个数
        cluster2_points = 0
        for dataset in datasets:
            if dataset.startswith('cluster2_') and dataset.endswith('_xyz'):
                cluster2_points = data[dataset].shape[0]
                break
        
        # 输出结果
        print(f"文件: {file_path}")
        print(f"cluster1 点云个数: {cluster1_points}")
        print(f"cluster2 点云个数: {cluster2_points}")
        print(f"总点数: {cluster1_points + cluster2_points}")
        
        return cluster1_points, cluster2_points
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None, None

if __name__ == "__main__":
    # 文件路径
    file_path = "pm_cluster\pm_points_raw_1749545953.npz"
    
    # 统计两类点云个数
    count_cluster_points(file_path)