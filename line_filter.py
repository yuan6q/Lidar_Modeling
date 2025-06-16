import numpy as np
import os
import matplotlib.pyplot as plt

# === 文件路径、文件名（使用相对路径） ===
load_path = 'pm_cluster'
file_name = 'pm_points_raw_1749545953.npz'
filepath = os.path.join(load_path, file_name)

# === 设置输出路径 ===
save_base_dir = os.path.join(load_path, 'filtered_points')
os.makedirs(save_base_dir, exist_ok=True)

# === 生成带 "_filtered" 后缀的文件名前缀和图像输出文件夹 ===
file_prefix = os.path.splitext(file_name)[0] + "_filtered"
min_points_per_line = 5
plot_output_dir = os.path.join(save_base_dir, f"{file_prefix}_fit_plots")
os.makedirs(plot_output_dir, exist_ok=True)

# === 加载数据 ===
data = np.load(filepath)
clusters = []
for i in [1, 2]:
    try:
        xyz = data[f'cluster{i}_xyz']
        ch = data[f'cluster{i}_ch']
        theta = data[f'cluster{i}_theta']
        clusters.append((xyz, ch, theta))
    except KeyError as e:
        print(f"跳过 cluster{i}：缺失字段 {e}")
        continue

all_xyz = np.concatenate([c[0] for c in clusters], axis=0)
all_id = np.concatenate([c[1] for c in clusters], axis=0)
all_azimuth = np.concatenate([c[2] for c in clusters], axis=0)

unique_ids = np.unique(all_id)
fit_results_filtered = {}
global_errors_filtered = {}

def fit_line(x, y):
    return np.polyfit(x, y, 1)

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

final_fit_lines = []  # 保存最终表达式

# === 逐个雷达编号处理并保存图像 ===
for lid in unique_ids:
    mask = all_id == lid
    if np.sum(mask) < min_points_per_line:
        continue

    x = all_xyz[mask, 0]
    y = all_xyz[mask, 1]

    # 原始拟合
    m1, b1 = fit_line(x, y)
    y_pred1 = m1 * x + b1
    residuals = y - y_pred1
    rmse1 = compute_rmse(y, y_pred1)

    # 基于 MAD 的滤波
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad < 1e-6:
        mask_filtered = np.abs(residuals) < 0.01
    else:
        mask_filtered = np.abs(residuals - med) < 2 * mad

    x_filt = x[mask_filtered]
    y_filt = y[mask_filtered]

    if len(x_filt) >= min_points_per_line:
        m2, b2 = fit_line(x_filt, y_filt)
        y_pred2 = m2 * x_filt + b2
        rmse2 = compute_rmse(y_filt, y_pred2)

        fit_results_filtered[int(lid)] = (m2, b2)
        global_errors_filtered[int(lid)] = rmse2
        final_fit_lines.append(f"Laser {int(lid)}: y = {m2:.4f}x + {b2:.4f}  (RMSE: {rmse2:.4f})")

        # 绘图与保存
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, s=5, label='Raw', alpha=0.5)
        plt.plot(np.sort(x), m1 * np.sort(x) + b1, 'r--', label='Raw Fit')

        plt.scatter(x_filt, y_filt, s=5, label='Filtered', alpha=0.7)
        plt.plot(np.sort(x_filt), m2 * np.sort(x_filt) + b2, 'g-', label='Filtered Fit')

        plt.title(f'Laser {lid} Fit (XY Plane)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(plot_output_dir, f"laser_{int(lid)}_fit.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        print(f"Laser {int(lid)}: 过滤后点数不足，跳过拟合图像保存")

# === 输出表达式到控制台与文件 ===
equation_path = os.path.join(save_base_dir, f"{file_prefix}_fit_equations.txt")
with open(equation_path, "w") as f:
    for line in final_fit_lines:
        print(line)
        f.write(line + "\n")

print(f"\n所有拟合图像已保存到: {plot_output_dir}")
print(f"所有拟合表达式已保存到: {equation_path}")

# === 汇总所有滤波后的点云索引并保存完整信息 ===
filtered_indices = []

for lid in unique_ids:
    mask = all_id == lid
    if np.sum(mask) < min_points_per_line:
        continue

    x = all_xyz[mask, 0]
    y = all_xyz[mask, 1]
    m1, b1 = fit_line(x, y)
    residuals = y - (m1 * x + b1)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad < 1e-6:
        mask_filtered = np.abs(residuals) < 0.01
    else:
        mask_filtered = np.abs(residuals - med) < 2 * mad

    # 获取全局索引
    idx_laser = np.where(mask)[0]
    idx_filtered = idx_laser[mask_filtered]
    filtered_indices.append(idx_filtered)

# === 拼接所有通过滤波的点索引 ===
filtered_indices = np.concatenate(filtered_indices)
filtered_indices = np.sort(filtered_indices)

# === 按索引保留所有字段数据 ===
filtered_xyz = all_xyz[filtered_indices]
filtered_ch = all_id[filtered_indices]
filtered_theta = all_azimuth[filtered_indices]

# 原始数据字段
field_map = {
    'xyz': filtered_xyz,
    'ch': filtered_ch,
    'theta': filtered_theta
}
# 若数据中有其他字段，也一并保留
optional_fields = ['intensity', 'ts', 'r']
for field in optional_fields:
    for i in [1, 2]:
        key = f'cluster{i}_{field}'
        if key in data:
            arr = data[key]
            break
    else:
        continue  # 如果两个聚类都没有这个字段就跳过
    all_field_data = np.concatenate([data[f'cluster{j}_{field}'] for j in [1, 2]], axis=0)
    field_map[field] = all_field_data[filtered_indices]

# === 保存保留字段的点云 ===
filtered_save_path = os.path.join(save_base_dir, f"{file_prefix}_points.npz")
np.savez(filtered_save_path, **field_map)

print(f"滤波后的完整点云数据已保存到: {filtered_save_path}")
