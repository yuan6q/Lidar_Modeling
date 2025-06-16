import numpy as np
import open3d as o3d
import time

# 1. 正方体参数设置
cube_center = np.array([0, 0, 0])  # 正方体中心
cube_size = 2.0                    # 边长
d = 5.0                            # 观测距离
num_angles = 8                     # 观测方向数量
angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)  # 8个方位角 (0-315°)

# 2. 生成高密度正方体表面点云（全局坐标系）
def generate_dense_cube_points(center, size, points_per_face=2500):
    points = []
    half = size / 2
    
    # 生成立方体6个面的点
    for axis in range(3):
        for sign in [-1, 1]:
            # 在面上生成密集网格点
            axes = [i for i in range(3) if i != axis]
            u = np.linspace(-half, half, int(np.sqrt(points_per_face)))
            v = np.linspace(-half, half, int(np.sqrt(points_per_face)))
            u, v = np.meshgrid(u, v)
            
            # 创建面上的点
            face_points = np.zeros((u.size, 3))
            face_points[:, axes[0]] = u.flatten() + center[axes[0]]
            face_points[:, axes[1]] = v.flatten() + center[axes[1]]
            face_points[:, axis] = center[axis] + sign * half
            
            points.append(face_points)
    
    # 添加边缘增强点
    edge_points = []
    for axis in range(3):
        for sign in [-1, 1]:
            for other_axis in range(3):
                if other_axis != axis:
                    # 沿着边缘生成点
                    edge_line = np.linspace(-half, half, int(np.sqrt(points_per_face)))
                    edge_pts = np.zeros((len(edge_line), 3))
                    edge_pts[:, axis] = center[axis] + sign * half
                    edge_pts[:, other_axis] = edge_line + center[other_axis]
                    edge_pts[:, 3-axis-other_axis] = center[3-axis-other_axis] + half
                    edge_points.append(edge_pts)
                    
                    edge_pts2 = edge_pts.copy()
                    edge_pts2[:, 3-axis-other_axis] = center[3-axis-other_axis] - half
                    edge_points.append(edge_pts2)
    
    all_points = np.vstack(points + edge_points)
    
    # 添加随机噪声使点云更真实
    noise = np.random.normal(0, 0.005, all_points.shape)
    return all_points + noise

# 生成高密度正方体点云 (每个面2500点 + 边缘增强)
print("生成高密度正方体点云...")
start_time = time.time()
cube_points = generate_dense_cube_points(cube_center, cube_size)
print(f"生成完成! 总点数: {len(cube_points)}，耗时: {time.time()-start_time:.2f}秒")

# 3. 计算雷达位置和坐标系方向
def calculate_lidar_position(angle, distance):
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    return np.array([x, y, 0])

def calculate_coordinate_system(angle):
    # Z轴指向正方体中心 (0,0,0)
    z_axis = np.array([-np.cos(angle), -np.sin(angle), 0])
    z_axis /= np.linalg.norm(z_axis)
    
    # Y轴向上 (全局Z方向)
    y_axis = np.array([0, 0, 1])
    
    # X轴由Y和Z叉积确定
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    return x_axis, y_axis, z_axis

# 4. 将全局点云转换到雷达坐标系
def global_to_local(point, lidar_pos, x_axis, y_axis, z_axis):
    # 计算相对位置
    relative_pos = point - lidar_pos
    
    # 计算在局部坐标系的坐标
    x = np.dot(relative_pos, x_axis)
    y = np.dot(relative_pos, y_axis)
    z = np.dot(relative_pos, z_axis)
    
    return np.array([x, y, z])

# 5. 从局部坐标系到绝对坐标系的变换
def local_to_absolute(local_point, angle_j, ref_angle=0):
    # 计算相对角度
    theta_j = angle_j - ref_angle
    
    # 平移向量 (根据推导公式)
    T = np.array([
        -d * np.sin(theta_j),
        0,
        -d * (np.cos(theta_j) - 1)
    ])
    
    # 旋转矩阵 (绕Y轴旋转-theta_j)
    rotation_matrix = np.array([
        [np.cos(theta_j), 0, np.sin(theta_j)],
        [0, 1, 0],
        [-np.sin(theta_j), 0, np.cos(theta_j)]
    ])
    
    # 应用变换
    return T + np.dot(rotation_matrix, local_point)

# 6. 主处理流程
# 选择参考坐标系 (0°方向)
ref_angle = 0
ref_idx = np.where(angles == ref_angle)[0][0]

# 存储所有变换后的点云
all_transformed_points = []

print("\n开始处理各视角点云...")
for i, angle in enumerate(angles):
    print(f"\n处理视角 {i+1}/8 (角度: {np.degrees(angle):.0f}°)")
    start_time = time.time()
    
    # 计算雷达位置和坐标系
    lidar_pos = calculate_lidar_position(angle, d)
    x_axis, y_axis, z_axis = calculate_coordinate_system(angle)
    
    # 将全局点云转换到当前雷达坐标系
    print("转换到局部坐标系...")
    local_points = []
    
    # 向量化计算提高效率
    direction_to_lidar = lidar_pos - cube_points
    norms = np.linalg.norm(direction_to_lidar, axis=1)
    direction_to_lidar_normalized = direction_to_lidar / norms[:, np.newaxis]
    
    # 可见性判断: 点与立方体中心的向量与到雷达方向的夹角
    center_to_point = cube_points - cube_center
    dot_products = np.sum(center_to_point * direction_to_lidar_normalized, axis=1)
    visible_mask = dot_products > 0
    
    print(f"可见点数量: {np.sum(visible_mask)}/{len(cube_points)}")
    
    # 只处理可见点
    visible_points = cube_points[visible_mask]
    
    # 计算局部坐标
    relative_pos = visible_points - lidar_pos
    x_coords = np.dot(relative_pos, x_axis)
    y_coords = np.dot(relative_pos, y_axis)
    z_coords = np.dot(relative_pos, z_axis)
    local_points = np.column_stack((x_coords, y_coords, z_coords))
    
    # 变换到绝对坐标系
    print("变换到绝对坐标系...")
    if i == ref_idx:
        transformed_points = local_points
    else:
        # 向量化变换
        theta_j = angle - ref_angle
        T = np.array([-d * np.sin(theta_j), 0, -d * (np.cos(theta_j) - 1)])
        rot_matrix = np.array([
            [np.cos(theta_j), 0, np.sin(theta_j)],
            [0, 1, 0],
            [-np.sin(theta_j), 0, np.cos(theta_j)]
        ])
        
        transformed_points = T + np.dot(local_points, rot_matrix.T)
    
    all_transformed_points.append(transformed_points)
    print(f"处理完成! 耗时: {time.time()-start_time:.2f}秒")

# 7. 合并所有点云
print("\n合并点云...")
merged_points = np.vstack(all_transformed_points)
print(f"总点数: {len(merged_points)}")

# 8. 使用Open3D进行可视化
print("\n使用Open3D可视化结果...")

# 创建点云对象
pcd_merged = o3d.geometry.PointCloud()
pcd_merged.points = o3d.utility.Vector3dVector(merged_points)
pcd_merged.paint_uniform_color([0.8, 0.2, 0.2])  # 红色表示合并点云

# 创建坐标系
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# 创建各视角点云对象
pcd_views = []
colors = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
    [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0], [0, 0.5, 0.5]
]

for i, points in enumerate(all_transformed_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(colors[i % len(colors)])
    pcd_views.append(pcd)

# 可视化所有视角点云和合并结果
print("显示所有视角点云和合并结果...")
o3d.visualization.draw_geometries([coordinate_frame] + pcd_views + [pcd_merged],
                                  window_name="多视角点云与合并结果",
                                  width=1200, height=800)

# 可视化合并结果（单独）
print("显示合并后的点云...")
o3d.visualization.draw_geometries([coordinate_frame, pcd_merged],
                                  window_name="合并后的点云",
                                  width=800, height=600)

# 保存点云为PLY文件
print("\n保存点云文件...")
o3d.io.write_point_cloud("high_density_merged_point_cloud.ply", pcd_merged)
for i, pcd in enumerate(pcd_views):
    o3d.io.write_point_cloud(f"high_density_view_{i}_point_cloud.ply", pcd)

print("处理完成! 高密度点云已保存为PLY文件。")