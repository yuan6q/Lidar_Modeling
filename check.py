import numpy as np

def list_npz_fields(file_path, verbose=False):
    """
    输出NPZ文件中的所有字段名称及其基本信息
    
    参数:
    file_path (str): NPZ文件路径
    verbose (bool): 是否显示每个字段的详细信息 (默认为False)
    """
    try:
        # 加载NPZ文件（不立即加载数据）
        with np.load(file_path) as data:
            # 获取所有字段名称
            fields = data.files
            
            # 输出基本信息
            print(f"文件: {file_path}")
            print(f"包含字段数量: {len(fields)}")
            print("所有字段名称:")
            print("-" * 50)
            
            # 输出字段列表
            for i, field in enumerate(fields, 1):
                print(f"{i}. {field}")
            
            print("-" * 50)
            
            # 如果要求详细输出
            if verbose:
                print("\n字段详细信息:")
                print("=" * 50)
                for field in fields:
                    # 获取数组对象
                    array = data[field]
                    
                    # 输出字段信息
                    print(f"字段名称: {field}")
                    print(f"  数据类型: {array.dtype}")
                    print(f"  数组形状: {array.shape}")
                    print(f"  数组维度: {array.ndim}D")
                    print(f"  元素总数: {array.size}")
                    print(f"  内存占用: {array.nbytes / 1024:.2f} KB")
                    
                    # 如果是结构化数组，显示字段信息
                    if array.dtype.names:
                        print(f"  结构化字段:")
                        for name in array.dtype.names:
                            print(f"    - {name}: {array.dtype[name]}")
                    
                    print("-" * 50)
    
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    # 文件路径
    file_path = "pm_cluster/filtered_points/pm_points_raw_1749545740_filtered_points.npz"
    
    # 仅列出字段名称
    print("=== 仅列出字段名称 ===")
    list_npz_fields(file_path)
    
    # 列出字段名称及详细信息
    print("\n=== 列出字段名称及详细信息 ===")
    list_npz_fields(file_path, verbose=True)