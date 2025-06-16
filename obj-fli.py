import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from scipy import stats

# ========= 1. 读取原始 npz 文件 =========
data = np.load('cube_pointclouds/0.npz')

# 提取所有字段
xyz = data['xyz']       # (N, 3)
ch = data['ch']         # (N,)
theta = data['theta']   # (N,)
intensity = data['intensity']  # (N,)
ts = data['ts']         # (N,)
r = data['r']           # (N,)

# ========= 2. 可视化滤波前点云 =========
def plot_pointcloud(xyz, color=None, title="Point Cloud", size=1):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, cmap='viridis', s=size)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

plot_pointcloud(xyz, color=ch, title="Original Point Cloud (colored by channel)", size=1)

# ========= 3. 精简但强大的滤波算法 =========

print(f"原始点数: {xyz.shape[0]}")

# 3.1 去除包含 NaN 的点
mask_valid = np.isfinite(xyz).all(axis=1)
xyz = xyz[mask_valid]
ch = ch[mask_valid]
theta = theta[mask_valid]
intensity = intensity[mask_valid]
ts = ts[mask_valid]
r = r[mask_valid]

print(f"去除NaN后点数: {xyz.shape[0]}")

# 3.2 基于统计的离群点去除 (Statistical Outlier Removal) - 去除强离群点
def statistical_outlier_removal(points, k=50, std_mult=2.5):
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k+1)  # 包含自身
    mean_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身
    
    # 计算距离分布的统计量
    mean = np.mean(mean_distances)
    std = np.std(mean_distances)
    
    # 设置阈值 - 超过平均距离+标准差倍数的点为离群点
    threshold = mean + std_mult * std
    return mean_distances < threshold

# 第一轮离群点去除
mask_sor = statistical_outlier_removal(xyz, k=30, std_mult=2.5)
xyz = xyz[mask_sor]
ch = ch[mask_sor]
theta = theta[mask_sor]
intensity = intensity[mask_sor]
ts = ts[mask_sor]
r = r[mask_sor]

print(f"离群点去除后点数: {xyz.shape[0]}")

# 3.3 基于局部平面拟合的残差分析 - 核心滤波方法
def plane_fitting_filter(points, k=20, residual_threshold=0.05):
    """
    通过局部平面拟合和残差分析去除较小误差的点
    """
    tree = KDTree(points)
    residuals = np.zeros(len(points))
    
    for i in range(len(points)):
        # 查找k个最近邻
        _, idx = tree.query(points[i], k=k+1)  # 包含自身
        
        # 获取邻域点
        neighborhood = points[idx]
        
        # 计算邻域质心
        centroid = np.mean(neighborhood, axis=0)
        
        # 计算协方差矩阵
        cov_matrix = np.cov((neighborhood - centroid).T)
        
        # 计算特征值和特征向量
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        
        # 最小特征值对应的特征向量是法线方向
        normal = eigvecs[:, 0]
        
        # 计算点到拟合平面的距离（残差）
        residuals[i] = np.abs(np.dot(points[i] - centroid, normal))
    
    # 计算残差的统计量
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # 设置残差阈值（更严格的阈值）
    threshold = mean_residual + 1.5 * std_residual
    
    # 保留残差小于阈值的点
    return residuals < threshold

# 应用局部平面拟合滤波
mask_plane = plane_fitting_filter(xyz, k=15, residual_threshold=0.03)
xyz = xyz[mask_plane]
ch = ch[mask_plane]
theta = theta[mask_plane]
intensity = intensity[mask_plane]
ts = ts[mask_plane]
r = r[mask_plane]

print(f"平面拟合滤波后点数: {xyz.shape[0]}")

# 3.4 多轮迭代的统计滤波 - 渐进式去除较小误差
def iterative_statistical_filter(points, iterations=2, k_values=[15, 10], std_mult_values=[1.8, 1.5]):
    """
    多轮迭代的统计滤波，逐步收紧参数
    """
    mask = np.ones(len(points), dtype=bool)
    
    for i in range(iterations):
        k = k_values[i]
        std_mult = std_mult_values[i]
        
        # 只对当前保留的点进行处理
        current_points = points[mask]
        
        # 如果没有点可处理，提前结束
        if len(current_points) == 0:
            break
        
        tree = KDTree(current_points)
        distances, _ = tree.query(current_points, k=k+1)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        mean = np.mean(mean_distances)
        std = np.std(mean_distances)
        threshold = mean + std_mult * std
        
        # 更新当前轮次的掩码
        current_mask = mean_distances < threshold
        
        # 更新总掩码
        mask_indices = np.where(mask)[0]
        mask[mask_indices] = current_mask
    
    return mask

# 应用多轮迭代统计滤波
mask_iterative = iterative_statistical_filter(xyz, iterations=2, 
                                            k_values=[15, 10], 
                                            std_mult_values=[1.8, 1.5])
xyz = xyz[mask_iterative]
ch = ch[mask_iterative]
theta = theta[mask_iterative]
intensity = intensity[mask_iterative]
ts = ts[mask_iterative]
r = r[mask_iterative]

print(f"迭代滤波后点数: {xyz.shape[0]}")

# ========= 4. 可视化滤波后点云 =========
plot_pointcloud(xyz, color=ch, title="Filtered Point Cloud (colored by channel)", size=2)

# ========= 5. 保存为新 npz 文件 =========
np.savez('filtered_pointcloud.npz',
         xyz=xyz,
         ch=ch,
         theta=theta,
         intensity=intensity,
         ts=ts,
         r=r)