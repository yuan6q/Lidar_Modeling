import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === 参数设置 ===
data_folder = 'clusterdata\\fil_pts'
file_template = '{}-fil.npz'
angles_deg = [i * 45 for i in range(8)]  # [0, 45, ..., 315]

# === 加载数据 ===
xyz_by_angle = {}
for i, angle in enumerate(angles_deg):
    file_path = os.path.join(data_folder, file_template.format(i))
    data = np.load(file_path)
    xyz = data['xyz']  # (N, 3)
    xyz_by_angle[angle] = xyz

# === 添加统一旋转函数（保持角度顺序和一致性） === 
def get_aligned_pointclouds(center_xy):
    aligned_points_by_angle = {}
    for angle in angles_deg:
        rotated = rotate_around_z_axis(xyz_by_angle[angle], center_xy, -angle)
        aligned_points_by_angle[angle] = rotated
    return aligned_points_by_angle

# === 点云旋转函数（绕任意 z 轴方向中心旋转） ===
def rotate_around_z_axis(points, center_xy, angle_deg):
    angle_rad = np.radians(angle_deg)
    cx, cy = center_xy
    translated = points[:, :2] - np.array([cx, cy])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.dot(translated, np.array([[c, -s], [s, c]])) + np.array([cx, cy])
    return np.hstack([rot, points[:, 2:3]])  # 保留z轴不变

# === 估算初始 转轴中心180°平面与雷达垂直距离相等，因此旋转中心和物体中心几乎重合） ===
def estimate_initial_d():
    y_90 = xyz_by_angle[90][:, 1]
    y_270 = xyz_by_angle[270][:, 1]
    return 0.25 * ((np.max(y_90) - np.min(y_90))+(np.max(y_270) - np.min(y_270)))
# 对准平面。计算转轴
dx=0
dy=0
for angle in range(0,271,90):
   dx+=(np.max(xyz_by_angle[angle][:, 0]) + np.min(xyz_by_angle[angle][:, 0]))*0.5
   dy+=(np.max(xyz_by_angle[angle][:, 1]) + np.min(xyz_by_angle[angle][:, 1]))*0.5
dx /= 4
dy /= 4
d_init = estimate_initial_d()
x0_init = d_init+dx  # 以 d 作为 x 初始值猜测
initial_guess = np.array([x0_init, dy])

estimated_center = initial_guess

print(f"估计的转轴位置：x = {estimated_center[0]:.3f}, y = {estimated_center[1]:.3f}")

def visualize_all_clouds(center_xy):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection': '3d'})
    axs = axs.flatten()

    # 获取对齐后的点云
    aligned_by_angle = get_aligned_pointclouds(center_xy)

    # 获取统一坐标范围
    all_points = np.vstack(list(aligned_by_angle.values()))
    pad = 0.1
    limits = {
        'x': [np.min(all_points[:, 0]) - pad, np.max(all_points[:, 0]) + pad],
        'y': [np.min(all_points[:, 1]) - pad, np.max(all_points[:, 1]) + pad],
        'z': [np.min(all_points[:, 2]) - pad, np.max(all_points[:, 2]) + pad],
    }

    # 使用 tab10 颜色
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(angles_deg))]

    # 单独展示每一帧
    for i, (angle, ax, color) in enumerate(zip(angles_deg, axs[:8], colors)):
        points = aligned_by_angle[angle]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=color)
        ax.set_title(f'{angle}°')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-60)
        ax.set_xlim(limits['x'])
        ax.set_ylim(limits['y'])
        ax.set_zlim(limits['z'])

    # 合并所有点云
    ax = axs[8]
    for i, angle in enumerate(angles_deg):
        points = aligned_by_angle[angle]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=colors[i], label=f'{angle}°')
    ax.set_title('All Aligned Point Clouds')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-60)
    ax.set_xlim(limits['x'])
    ax.set_ylim(limits['y'])
    ax.set_zlim(limits['z'])
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

# === 可视化所有配准结果 ===
visualize_all_clouds(estimated_center)
# === 合并所有点云为一个整体并保存 ===
aligned_all = np.vstack(list(get_aligned_pointclouds(estimated_center).values()))

# 保存为 npz 文件
output_path = 'aligned_merged.npz'
np.savez(output_path, xyz=aligned_all)
print(f"已保存配准后的合并点云到：{output_path}")