import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import os

# ========= 1. 读取原始 npz 文件 =========
data = np.load('aligned_merged.npz')
output_path = 'finaldata.npz'
# 只提取 xyz 和 ch 字段
xyz = data['xyz']       # (N, 3)  

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

# ========= 3. 滤波算法 =========

print(f"原始点数: {xyz.shape[0]}")

# 3.1 去除包含 NaN 的点
mask_valid = np.isfinite(xyz).all(axis=1)
xyz = xyz[mask_valid]
print(f"去除NaN后点数: {xyz.shape[0]}")

# 3.2 基于统计的离群点去除
def statistical_outlier_removal(points, k=30, std_mult=2.5):
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k+1)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    threshold = np.mean(mean_distances) + std_mult * np.std(mean_distances)
    return mean_distances < threshold

mask_sor = statistical_outlier_removal(xyz)
xyz = xyz[mask_sor]
print(f"离群点去除后点数: {xyz.shape[0]}")

# 3.3 基于局部平面拟合的残差分析
def plane_fitting_filter(points, k=15):
    tree = KDTree(points)
    residuals = np.zeros(len(points))
    for i in range(len(points)):
        _, idx = tree.query(points[i], k=k+1)
        neighborhood = points[idx]
        centroid = np.mean(neighborhood, axis=0)
        cov_matrix = np.cov((neighborhood - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normal = eigvecs[:, 0]
        residuals[i] = np.abs(np.dot(points[i] - centroid, normal))
    threshold = np.mean(residuals) + 1.5 * np.std(residuals)
    return residuals < threshold

mask_plane = plane_fitting_filter(xyz)
xyz = xyz[mask_plane]
print(f"平面拟合滤波后点数: {xyz.shape[0]}")

# 3.4 迭代统计滤波
def iterative_statistical_filter(points, iterations=2, k_values=[15, 10], std_mult_values=[1.8, 1.5]):
    mask = np.ones(len(points), dtype=bool)
    for i in range(iterations):
        k = k_values[i]
        std_mult = std_mult_values[i]
        current_points = points[mask]
        if len(current_points) == 0:
            break
        tree = KDTree(current_points)
        distances, _ = tree.query(current_points, k=k+1)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        threshold = np.mean(mean_distances) + std_mult * np.std(mean_distances)
        current_mask = mean_distances < threshold
        mask_indices = np.where(mask)[0]
        mask[mask_indices] = current_mask
    return mask

mask_iter = iterative_statistical_filter(xyz)
xyz = xyz[mask_iter]
print(f"迭代滤波后点数: {xyz.shape[0]}")

# ========= 4. 可视化滤波后点云 =========
plot_pointcloud(xyz, title="Filtered Point Cloud", size=2)

# ========= 5. 保存为新 npz 文件 =========
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.savez(output_path, xyz=xyz)
print(f"已保存滤波后的点云到：{output_path}")
