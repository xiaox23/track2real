import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

def apply_transformation(points, transform_matrix):
    """应用4x4变换矩阵到点云"""
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = (transform_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]

def compute_rmse(source, target):
    """计算源点云到目标点云的RMSE"""
    tree = KDTree(target)
    distances, _ = tree.query(source)
    return np.sqrt(np.mean(distances**2))

def visualize_point_clouds(source, target, source_color=[1, 0, 0], target_color=[0, 0, 1]):
    """可视化点云"""
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)
    pcd_source.paint_uniform_color(source_color)

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)
    pcd_target.paint_uniform_color(target_color)

    o3d.visualization.draw_geometries([pcd_source, pcd_target])

# import numpy as np
# import open3d as o3d
# from scipy.spatial import KDTree

# def load_pcd_as_numpy(pcd_path):
#     """读取PCD文件并转换为NumPy数组"""
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     return np.asarray(pcd.points)

# def apply_transformation_open3d(pcd, transform_matrix):
#     """使用Open3D内置方法应用变换"""
#     transformed_pcd = pcd.transform(transform_matrix)
#     return transformed_pcd

# def compute_rmse(source_points, target_points):
#     """计算点云配准误差"""
#     tree = KDTree(target_points)
#     distances, _ = tree.query(source_points)
#     return np.sqrt(np.mean(distances**2))

if __name__ == "__main__":
    # # 1. 读取PCD文件
    # source_pcd = o3d.io.read_point_cloud(r"/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/real_point_cloud.pcd")  # 替换为实际路径
    # target_pcd = o3d.io.read_point_cloud(r"/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/simu_point_cloud.pcd")  # 替换为实际路径
    # source_pcd_tran = o3d.io.read_point_cloud(r"/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/real_point_cloud.pcd")  # 替换为实际路径
    
    # # 2. 定义变换矩阵（示例矩阵，需替换为实际矩阵）
    # T1 = np.array([[1.000,-0.005,0.003,0.012],[0.005,1.000,0.018,-0.022],[-0.003,-0.018,1.000,0.007],[0.000,0.000,0.000,1.000]])  # 单位矩阵，无变换
    # T2 = np.array([[1.000,-0.018,-0.025,0.012],[0.017,0.999,-0.042,0.020],[0.026,0.042,0.999,-0.007],[0.000,0.000,0.000,1.000]])  # 单位矩阵，无变换


    # # 3. 应用变换（两种方法任选其一）
    # # 方法一：使用Open3D直接变换
    # transformed_pcd_1 = source_pcd_tran.transform(np.linalg.inv(T1))
    # transformed_pcd_2 = transformed_pcd_1.transform(np.linalg.inv(T2))

    # tran_matrix = np.linalg.inv(T1) @ np.linalg.inv(T2)
    # print(tran_matrix)
    
    # # # 方法二：转换为NumPy处理（适用于自定义处理）
    # # source_points = np.asarray(source_pcd.points)
    # # transformed_points = (transform_matrix[:3, :3] @ source_points.T).T + transform_matrix[:3, 3]
    # # transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    # # 4. 计算配准误差
    # source_pcd_tran_points = np.asarray(source_pcd_tran.points)
    # target_points = np.asarray(target_pcd.points)
    # rmse = compute_rmse(target_points, source_pcd_tran_points)
    # print(f"配准误差RMSE: {rmse:.6f} meters")

    # # # 5. 可视化对比
    # source_pcd.paint_uniform_color([1, 0, 0])  # 红色为原始点云
    # transformed_pcd_2.paint_uniform_color([0, 1, 0])  # 绿色为变换后点云
    # target_pcd.paint_uniform_color([0, 0, 1])  # 蓝色为目标点云
    # o3d.visualization.draw_geometries([source_pcd, transformed_pcd_2, target_pcd], window_name="Point Cloud Registration")

    # # 6. 保存结果（可选）
    # o3d.io.write_point_cloud("transformed.pcd", transformed_pcd)



    # 示例数据（替换为实际点云数据）
    # A = np.loadtxt('cloud_a.txt')  # 源点云
    # B = np.loadtxt('cloud_b.txt')  # 目标点云
    
    pcd_real = np.load("/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/real_point_cloud.npy")
    pcd_sim = np.load("/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/simu_point_cloud.npy")

    # 示例变换矩阵（替换为实际矩阵）
    T1 = np.array([[1.000,-0.005,0.003,0.012],[0.005,1.000,0.018,-0.022],[-0.003,-0.018,1.000,0.007],[0.000,0.000,0.000,1.000]])  # 单位矩阵，无变换
    T2 =  np.array([[1.000,-0.018,-0.025,0.012],[0.017,0.999,-0.042,0.0202],[0.026,0.042,0.999,-0.007],[0.000,0.000,0.000,1.000]])  # 单位矩阵，无变换
    T3 = np.array([[ 0.99899855,  0.02198923,  0.02283245, -0.02413714],
 [-0.02264921 , 0.9992579   ,0.02344749 , 0.00262957],
 [-0.02265247 ,-0.02438476 , 0.99907572 , 0.00111667],
 [ 0.       ,   0.  ,        0.   ,       1.        ]])
    # 应用变换
    # A1 = apply_transformation(pcd_real, T1)
    # A2 = apply_transformation(A1, T2)
    A = apply_transformation(pcd_real, T3)

    # 计算RMSE
    rmse_t1 = compute_rmse(A, pcd_sim)
    # rmse_t2 = compute_rmse(A2, B)
    print(f"RMSE after T1: {rmse_t1:.4f}")
    # print(f"RMSE after T2: {rmse_t2:.4f}")

    # # 可视化结果
    visualize_point_clouds(A, pcd_real, [1, 0, 0], [0, 0, 1])  # 红色为变换后的A1，蓝色为B
    visualize_point_clouds(A, pcd_sim, [1, 0, 0], [0, 0, 1])  # 红色为变换后的A2，蓝色为B