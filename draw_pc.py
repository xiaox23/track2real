

import numpy as np
import open3d as o3d

# 假设点云数据和 mask 数据已经加载到 numpy 数组中
# point_cloud: 480x640x3 的点云数据
# mask: 480x640 的 mask 数据，值为 0, 1, 2
point_cloud = np.load(r"/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_22-37-33.325/Episode_1/obs_and_info_step_0/raw_point_cloud_1_0.npy") # 示例点云数据
mask = np.load(r"/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_22-37-33.325/Episode_1/obs_and_info_step_0/mask_1_0.npy")  # 示例 mask 数据

# 提取 mask 中值为 1 和 2 的点
indices_1 = np.where(mask == 1)  # 获取 mask 值为 1 的点的索引
indices_2 = np.where(mask == 2)  # 获取 mask 值为 2 的点的索引

# 合并索引
aa = np.concatenate((indices_1, indices_2), axis=1)
indices = aa.T

# 提取对应的点云
selected_points = point_cloud[indices[:, 0], indices[:, 1]]

colors = np.zeros_like(selected_points)
colors[:, 0] = 1.0  # 红色通道

# 将点云转换为 Open3D 的 PointCloud 格式
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(selected_points)
pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置颜色


# 新增绿色点云
point_cloud_peg = np.load(r"/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/point_peg_1_0.npy") # 示例绿色点云数据
point_cloud_hole = np.load(r"/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/point_hole_1_0.npy") # 示例绿色点云数据
point_cloud_green = np.concatenate((point_cloud_peg, point_cloud_hole), axis=0)
point_cloud_green_new = point_cloud_green.reshape(-1, 3)

# 提取绿色点云
colors_green = np.zeros_like(point_cloud_green_new)
colors_green[:, 1] = 1.0  # 设置为绿色

# 创建绿色点云
pcd_green = o3d.geometry.PointCloud()
pcd_green.points = o3d.utility.Vector3dVector(point_cloud_green_new)
pcd_green.colors = o3d.utility.Vector3dVector(colors_green)

# np_pcd_red = np.asarray(pcd.points)
# np_pcd_green = np.asarray(pcd_green.points)
# np.save("/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/real_point_cloud.npy", np_pcd_red)
# np.save("/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/simu_point_cloud.npy", np_pcd_green)


# o3d.io.write_point_cloud("/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/real_point_cloud.pcd", pcd)
# o3d.io.write_point_cloud("/home/xulab10/lcy/challenge_2025_real_track2_auxloss/Memo/2025-02-24_15-59-16.205/simu_point_cloud.pcd", pcd_green)


# 合并并可视化两个点云
o3d.visualization.draw_geometries([pcd, pcd_green], window_name="Combined Point Clouds")

