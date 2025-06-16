import numpy as np

def generate_occluded_cube_pointcloud(angle_deg=45, num_points=9000, 
                                     normal_noise_std=0.05, tangent_noise_std=0.03, 
                                     global_noise_std=0.01, outlier_ratio=0.15, 
                                     outlier_std=0.8, surface_roughness=0.02, seed=42):
    """
    生成带遮挡的立方体点云，添加多方向噪声和离群点
    
    参数:
    angle_deg: 观察角度(度)
    num_points: 总点数
    normal_noise_std: 法线方向噪声标准差
    tangent_noise_std: 切向方向噪声标准差
    global_noise_std: 全局随机噪声标准差
    outlier_ratio: 离群点比例
    outlier_std: 离群点噪声标准差
    surface_roughness: 表面粗糙度（沿表面随机偏移）
    seed: 随机种子
    """
    np.random.seed(seed)
    side = 1.0
    half = side / 2.0
    points = []
    normals_list = []  # 保存每个点的法线方向
    
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

    # 旋转
    angle_rad = np.radians(angle_deg)
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    xyz_rot = xyz @ Rz.T
    
    # 法向量随点云一起旋转
    normals_rot = normals @ Rz.T

    # 视线方向
    view_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
    
    # 可见性判断
    visible_mask = (normals_rot @ view_dir) > 0
    xyz_visible = xyz_rot[visible_mask]
    normals_visible = normals_rot[visible_mask]
    
    # 计算可见点数
    num_visible_points = xyz_visible.shape[0]
    
    # ================== 多方向噪声增强 ==================
    # 1. 法线方向噪声（垂直于表面）
    normal_noise = np.zeros_like(xyz_visible)
    for i in range(num_visible_points):
        normal_dir = normals_visible[i]
        noise_magnitude = np.random.normal(scale=normal_noise_std)
        normal_noise[i] = normal_dir * noise_magnitude
    
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
        
        # 第二个切向量是法线和第一个切向量的叉积
        tangent2 = np.cross(normal_dir, tangent1)
        
        # 在切平面内添加随机噪声
        noise_dir = np.random.randn(2)  # 两个切向量的随机系数
        noise_dir /= np.linalg.norm(noise_dir)  # 归一化
        noise_vector = tangent_noise_std * (noise_dir[0] * tangent1 + noise_dir[1] * tangent2)
        
        # 添加表面粗糙度（沿表面的随机偏移）
        surface_offset = surface_roughness * np.random.randn()
        tangent_noise[i] = noise_vector * np.random.randn() + surface_offset * tangent1
    
    # 3. 全局随机噪声
    global_noise = np.random.normal(scale=global_noise_std, size=xyz_visible.shape)
    
    # 4. 组合所有噪声
    noisy_points = xyz_visible + normal_noise + tangent_noise + global_noise
    
    # 5. 添加离群点
    num_outliers = int(num_visible_points * outlier_ratio)
    
    # 5.1 完全随机的离群点
    random_outliers = np.random.uniform(low=-1.5, high=1.5, size=(num_outliers//2, 3))
    
    # 5.2 从原始点偏移的离群点
    outlier_indices = np.random.choice(num_visible_points, num_outliers - num_outliers//2, replace=False)
    offset_outliers = noisy_points[outlier_indices] + np.random.normal(scale=outlier_std, 
                                                                    size=(len(outlier_indices), 3))
    
    # 6. 组合所有点
    all_points = np.vstack([noisy_points, random_outliers, offset_outliers])
    # =============================================
    
    # 生成其他属性
    N = all_points.shape[0]
    ch = np.random.randint(0, 64, size=N)
    theta = np.full(N, angle_rad)
    intensity = np.random.uniform(0, 1, size=N)
    ts = np.linspace(0, 1, N)
    r = np.linalg.norm(all_points, axis=1)

    # 保存结果
    np.savez('input_pointcloud.npz',
             xyz=all_points,
             ch=ch,
             theta=theta,
             intensity=intensity,
             ts=ts,
             r=r)
    
    print(f"生成点云完成! 总点数: {N}")
    print(f"- 可见表面点数: {num_visible_points}")
    print(f"- 添加离群点数: {num_outliers}")
    print(f"- 法线噪声标准差: {normal_noise_std}")
    print(f"- 切向噪声标准差: {tangent_noise_std}")
    print(f"- 全局噪声标准差: {global_noise_std}")
    print(f"- 表面粗糙度: {surface_roughness}")
    print(f"- 离群点比例: {outlier_ratio*100}%")
    print(f"- 离群点噪声标准差: {outlier_std}")
    print(f"文件已保存: input_pointcloud.npz")

# 测试调用
if __name__ == "__main__":
    # 使用多方向噪声和离群点
    generate_occluded_cube_pointcloud(
        angle_deg=45,
        num_points=12000,
        normal_noise_std=0.03,      # 垂直于表面的噪声
        tangent_noise_std=0.05,     # 平行于表面的噪声 - 增加此值使XY方向更不规则
        global_noise_std=0.01,      # 全局随机噪声
        surface_roughness=0.04,     # 沿表面的随机偏移
        outlier_ratio=0.15,         # 离群点比例
        outlier_std=0.8             # 离群点噪声
    )