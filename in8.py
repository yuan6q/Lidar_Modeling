import numpy as np
import os

def generate_occluded_cube_pointcloud(angle_deg=45, num_points=12000, 
                                     normal_noise_std=0.01, tangent_noise_std=0.02, 
                                     global_noise_std=0.005, outlier_ratio=0.05, 
                                     outlier_std=0.2, surface_roughness=0.01, 
                                     output_name="input_pointcloud", seed=42):
    """
    生成带遮挡的立方体点云（误差较小版本）
    
    参数:
    angle_deg: 观察角度(度)
    num_points: 总点数
    normal_noise_std: 法线方向噪声标准差（减小）
    tangent_noise_std: 切向方向噪声标准差（减小）
    global_noise_std: 全局随机噪声标准差（减小）
    outlier_ratio: 离群点比例（减小）
    outlier_std: 离群点噪声标准差（减小）
    surface_roughness: 表面粗糙度（减小）
    output_name: 输出文件名前缀
    seed: 随机种子
    """
    np.random.seed(seed)
    side = 1.0
    half = side / 2.0
    points = []
    normals_list = []
    
    # 定义立方体六个面
    face_choices = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
    points_per_face = num_points // len(face_choices)

    for face in face_choices:
        for _ in range(points_per_face):
            if face == 'x+':
                x, y, z = half, np.random.uniform(-half, half), np.random.uniform(-half, half)
                normal = [1, 0, 0]
            elif face == 'x-':
                x, y, z = -half, np.random.uniform(-half, half), np.random.uniform(-half, half)
                normal = [-1, 0, 0]
            elif face == 'y+':
                y, x, z = half, np.random.uniform(-half, half), np.random.uniform(-half, half)
                normal = [0, 1, 0]
            elif face == 'y-':
                y, x, z = -half, np.random.uniform(-half, half), np.random.uniform(-half, half)
                normal = [0, -1, 0]
            elif face == 'z+':
                z, x, y = half, np.random.uniform(-half, half), np.random.uniform(-half, half)
                normal = [0, 0, 1]
            elif face == 'z-':
                z, x, y = -half, np.random.uniform(-half, half), np.random.uniform(-half, half)
                normal = [0, 0, -1]
            points.append([x, y, z])
            normals_list.append(normal)

    xyz = np.array(points)
    normals = np.array(normals_list)

    # 旋转立方体
    angle_rad = np.radians(angle_deg)
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    xyz_rot = xyz @ Rz.T
    normals_rot = normals @ Rz.T  # 法向量随点云一起旋转

    # 视线方向（沿X轴）
    view_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
    
    # 可见性判断（点积大于0表示面向相机）
    visible_mask = (normals_rot @ view_dir) > 0
    xyz_visible = xyz_rot[visible_mask]
    normals_visible = normals_rot[visible_mask]
    
    num_visible_points = xyz_visible.shape[0]
    
    # ================== 噪声处理（减小噪声参数） ==================
    # 1. 法线方向噪声（垂直于表面）
    normal_noise = np.random.normal(scale=normal_noise_std, size=xyz_visible.shape)
    for i in range(num_visible_points):
        normal_dir = normals_visible[i]
        # 投影到法线方向
        projection = np.dot(normal_noise[i], normal_dir) * normal_dir
        normal_noise[i] = projection
    
    # 2. 切向方向噪声（平行于表面）
    tangent_noise = np.zeros_like(xyz_visible)
    for i in range(num_visible_points):
        normal_dir = normals_visible[i]
        
        # 创建两个垂直于法线的切向量
        if abs(normal_dir[0]) > 0.5:  # 如果法线主要是x方向
            tangent1 = np.array([0, 1, 0])
        else:
            tangent1 = np.array([1, 0, 0])
        
        # 确保切向量垂直于法线
        tangent1 = tangent1 - np.dot(tangent1, normal_dir) * normal_dir
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(normal_dir, tangent1)
        
        # 在切平面内添加随机噪声
        coeff1 = np.random.normal(scale=tangent_noise_std)
        coeff2 = np.random.normal(scale=tangent_noise_std)
        tangent_noise[i] = coeff1 * tangent1 + coeff2 * tangent2
    
    # 3. 全局随机噪声
    global_noise = np.random.normal(scale=global_noise_std, size=xyz_visible.shape)
    
    # 4. 组合所有噪声
    noisy_points = xyz_visible + normal_noise + tangent_noise + global_noise
    
    # 5. 添加少量离群点
    num_outliers = int(num_visible_points * outlier_ratio)
    # 5.1 完全随机的离群点
    random_outliers = np.random.uniform(low=-1.2, high=1.2, size=(num_outliers//2, 3))
    # 5.2 从原始点偏移的离群点
    outlier_indices = np.random.choice(num_visible_points, num_outliers - num_outliers//2, replace=False)
    offset_outliers = noisy_points[outlier_indices] + np.random.normal(scale=outlier_std, size=(len(outlier_indices), 3))
    
    # 6. 组合所有点
    all_points = np.vstack([noisy_points, random_outliers, offset_outliers])
    # ========================================================
    
    # 生成其他属性
    N = all_points.shape[0]
    ch = np.random.randint(0, 64, size=N)
    theta = np.full(N, angle_rad)
    intensity = np.random.uniform(0.7, 1.0, size=N)  # 提高强度值使点更明显
    ts = np.linspace(0, 1, N)
    r = np.linalg.norm(all_points, axis=1)

    # 保存结果
    output_path = f"{output_name}.npz"
    np.savez(output_path,
             xyz=all_points,
             ch=ch,
             theta=theta,
             intensity=intensity,
             ts=ts,
             r=r)
    
    print(f"生成角度 {angle_deg}° 点云完成! 总点数: {N}")
    print(f"- 可见表面点数: {num_visible_points}")
    print(f"- 添加离群点数: {num_outliers}")
    print(f"文件已保存: {output_path}")
    return all_points

# 生成8个角度的点云数据 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
if __name__ == "__main__":
    output_dir = "cube_pointclouds"
    os.makedirs(output_dir, exist_ok=True)
    
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for i, angle in enumerate(angles):
        print(f"\n正在生成角度 {angle}° 的数据 ({i+1}/8)")
        generate_occluded_cube_pointcloud(
            angle_deg=angle,
            num_points=12000,          # 增加初始点数补偿遮挡损失
            normal_noise_std=0.01,     # 减小法线噪声
            tangent_noise_std=0.015,   # 减小切向噪声
            global_noise_std=0.005,    # 减小全局噪声
            outlier_ratio=0.05,       # 减小离群点比例
            outlier_std=0.15,         # 减小离群点噪声
            surface_roughness=0.008,  # 减小表面粗糙度
            output_name=os.path.join(output_dir, str(i)),  # 输出文件名 0.npz, 1.npz, ...
            seed=42 + i  # 不同角度使用不同种子
        )
    
    print("\n所有角度点云生成完成！文件保存在", output_dir)