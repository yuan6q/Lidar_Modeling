import numpy as np
import os
import matplotlib.pyplot as plt

# === 文件路径设置 ===
load_path = 'clusterdata'
file_name = '0_points_raw_1749546812.npz'
filepath = os.path.join(load_path, file_name)

# === 输出路径 ===
save_base_dir = os.path.join(load_path, 'filtered_points')
os.makedirs(save_base_dir, exist_ok=True)
file_prefix = os.path.splitext(file_name)[0] + "_filtered"
plot_output_dir = os.path.join(save_base_dir, f"{file_prefix}_fit_plots")
os.makedirs(plot_output_dir, exist_ok=True)

# === 加载数据并识别字段 ===
data = np.load(filepath)
field_suffixes = set()
for key in data.files:
    if key.startswith('cluster1_') or key.startswith('cluster2_'):
        suffix = key.split('_', 1)[1]
        field_suffixes.add(suffix)

data_fields = {}
for suffix in field_suffixes:
    arrays = []
    for i in [1, 2]:
        key = f'cluster{i}_{suffix}'
        if key in data:
            arrays.append(data[key])
    if arrays:
        try:
            combined = np.concatenate(arrays, axis=0)
            data_fields[suffix] = combined
        except Exception as e:
            print(f"字段 {suffix} 拼接失败：{e}")

# === 拟合准备 ===
all_xyz = data_fields['xyz']
all_id = data_fields['ch']
all_theta = data_fields.get('theta', np.zeros(len(all_id)))

unique_ids = np.unique(all_id)
min_points_per_line = 5
fit_results_filtered = {}
global_errors_filtered = {}
final_fit_lines = []
filtered_indices = []

# === 拟合 + 过滤 + 图像输出 ===
for lid in unique_ids:
    mask = all_id == lid
    if np.sum(mask) < min_points_per_line:
        continue

    x = all_xyz[mask, 0]
    y = all_xyz[mask, 1]

    m1, b1 = np.polyfit(x, y, 1)
    y_pred1 = m1 * x + b1
    residuals = y - y_pred1
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad < 1e-6:
        mask_filtered = np.abs(residuals) < 0.02
    else:
        mask_filtered = np.abs(residuals - med) < 0.7 * mad

    x_filt = x[mask_filtered]
    y_filt = y[mask_filtered]

    if len(x_filt) >= min_points_per_line:
        m2, b2 = np.polyfit(x_filt, y_filt, 1)
        y_pred2 = m2 * x_filt + b2
        rmse2 = np.sqrt(np.mean((y_filt - y_pred2) ** 2))

        fit_results_filtered[int(lid)] = (m2, b2)
        global_errors_filtered[int(lid)] = rmse2
        final_fit_lines.append(f"Laser {int(lid)}: y = {m2:.4f}x + {b2:.4f}  (RMSE: {rmse2:.4f})")

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

    idx_laser = np.where(mask)[0]
    idx_filtered = idx_laser[mask_filtered]
    filtered_indices.append(idx_filtered)

# === 索引合并与数据过滤 ===
filtered_indices = np.concatenate(filtered_indices)
filtered_indices = np.sort(filtered_indices)

filtered_data = {}
for field, arr in data_fields.items():
    try:
        filtered_data[field] = arr[filtered_indices]
    except Exception as e:
        print(f"字段 {field} 筛选失败: {e}")

# === 保存拟合表达式 ===
equation_path = os.path.join(save_base_dir, f"{file_prefix}_fit_equations.txt")
with open(equation_path, "w") as f:
    for line in final_fit_lines:
        print(line)
        f.write(line + "\n")

# === 保存筛选后的完整数据 ===
filtered_save_path = os.path.join(save_base_dir, f"{file_prefix}_points.npz")
np.savez(filtered_save_path, **filtered_data)

# === 总结输出 ===
print(f"\n所有拟合图像已保存到: {plot_output_dir}")
print(f"所有拟合表达式已保存到: {equation_path}")
print(f"滤波后的完整点云数据已保存到: {filtered_save_path}")