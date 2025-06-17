import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def clean_bending_bottom_edges_numpy(xyz, z_range=0.025, fit_axis='x', max_dev=0.01):
    
    # 1. 提取底部点
    min_z = np.min(xyz[:, 2])
    mask_bottom = xyz[:, 2] < min_z + z_range
    bottom_points = xyz[mask_bottom]

    if bottom_points.shape[0] < 10:
        print("底部点过少，跳过清理")
        return xyz

    # 2. 拟合直线（最小二乘法）
    if fit_axis == 'x':
        x = bottom_points[:, 0]
        y = bottom_points[:, 1]
    else:
        x = bottom_points[:, 1]
        y = bottom_points[:, 0]

    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]  # y ≈ mx + b

    # 3. 计算点到拟合直线的距离
    y_fit = m * x + b
    residuals = np.abs(y - y_fit)

    # 4. 选取偏差较小的点
    inliers = residuals < max_dev
    cleaned_bottom = bottom_points[inliers]

    # 5. 其他点拼回
    remaining = xyz[~mask_bottom]
    cleaned_xyz = np.vstack([remaining, cleaned_bottom])

    print(f"[清理底部弯折] 总底部点: {len(bottom_points)}, 保留: {len(cleaned_bottom)}, 总剩余点: {len(cleaned_xyz)}")

    return cleaned_xyz


# ========== 1. 加载合并点云 ==========
input_path = 'aligned_merged.npz'
data = np.load(input_path)
xyz = data['xyz']
print(f"[原始点数] {xyz.shape[0]}")

# ========== 2. 去除 NaN ==========
mask_valid = np.isfinite(xyz).all(axis=1)
xyz = xyz[mask_valid]
print(f"[去除 NaN 后] {xyz.shape[0]}")


# ========== 3. 强化版统计离群点滤波 ==========
def statistical_outlier_removal(points, k=25, std_mult=2.5):
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k+1)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    mean = np.mean(mean_dists)
    std = np.std(mean_dists)
    threshold = mean + std_mult * std
    return mean_dists < threshold

# 第1轮
mask1 = statistical_outlier_removal(xyz, k=40, std_mult=1.5)
xyz = xyz[mask1]
print(f"[统计滤波 1] {xyz.shape[0]}")
xyz = clean_bending_bottom_edges_numpy(xyz, fit_axis='x')
xyz = clean_bending_bottom_edges_numpy(xyz, fit_axis='y')

# 第2轮更严格
mask2 = statistical_outlier_removal(xyz, k=30, std_mult=1.2)
xyz = xyz[mask2]
print(f"[统计滤波 2] {xyz.shape[0]}")

# ========== 4. 平面拟合残差滤波 ==========
def plane_fitting_filter(points, k=10,residual_threshold=0.01):
    tree = KDTree(points)
    residuals = np.zeros(len(points))
    for i in range(len(points)):
        _, idx = tree.query(points[i], k=k+1)
        neighborhood = points[idx]
        centroid = np.mean(neighborhood, axis=0)
        cov = np.cov((neighborhood - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        residuals[i] = np.abs(np.dot(points[i] - centroid, normal))
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    threshold = mean_res + 1.2 * std_res
    return residuals < threshold

mask_plane = plane_fitting_filter(xyz, k=15)
xyz = xyz[mask_plane]
print(f"[平面拟合滤波] {xyz.shape[0]}")
xyz = clean_bending_bottom_edges_numpy(xyz, z_range=0.025, fit_axis='x', max_dev=0.01)
mask_z = xyz[:, 2] >= -0.12
xyz = xyz[mask_z]
# ========== 5. 可视化结果 ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Filtered Merged Point Cloud")
plt.tight_layout()
plt.show()

# ========== 6. 保存结果 ==========
output_path = 'clusterdata/fil_pts/aligned_merged_filtered.npz'
np.savez(output_path, xyz=xyz)
print(f"[保存成功] 点云写入：{output_path}")
