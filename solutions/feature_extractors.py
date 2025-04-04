import os, sys

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from networks import PointNetFeatureExtractor

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import gymnasium as gym
from solutions.networks import PointNetFeatureExtractor

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision.transforms as transforms
import open3d as o3d
import cv2
"""
Feature Extractors for different environments
by default, the feature extractors are for actor network
unless it starts with "CriticFeatureExtractor"
"""


class CriticFeatureExtractor(BaseFeaturesExtractor):
    """general critic feature extractor for peg-in-hole env. the input for critic network is the gt_offset."""

    def __init__(self, observation_space: gym.spaces):
        super(CriticFeatureExtractor, self).__init__(observation_space, features_dim=3)
        self._features_dim = 3

    def forward(self, observations) -> torch.Tensor:
        return observations["gt_offset"]


class FeatureExtractorForPointFlowEnv(BaseFeaturesExtractor):
    """
    feature extractor for point flow env. the input for actor network is the point flow.
    so this 'feature extractor' actually only extracts point flow from the original observation dictionary.
    the actor network contains a pointnet module to extract latent features from point flow.
    """

    def __init__(self, observation_space: gym.spaces):
        super(FeatureExtractorForPointFlowEnv, self).__init__(
            observation_space, features_dim=512
        )
        self._features_dim = 512

    def forward(self, observations) -> torch.Tensor:
        original_obs = observations["marker_flow"]
        if original_obs.ndim == 4:
            original_obs = torch.unsqueeze(original_obs, 0)
        # (batch_num, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v))
        fea = torch.cat(
            [original_obs[:, :, 0, ...], original_obs[:, :, 1, ...]], dim=-1
        )
        return fea


class FeatureExtractorState(BaseFeaturesExtractor):
    """
    General critic feature extractor for PegInsertion env (v2).
    The input for critic network is the gt_offset + relative_motion + direction.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        super(FeatureExtractorState, self).__init__(observation_space, features_dim=9)

    def forward(self, observations) -> torch.Tensor:
        gt_offset = observations["gt_offset"]  # 4
        relative_motion = observations["relative_motion"]  # 4
        gt_direction = observations["gt_direction"]  # 1
        return torch.cat([gt_offset, relative_motion, gt_direction], dim=-1)


class FeaturesExtractorPointCloud(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        vision_kwargs: dict = None,
        tac_kwargs: dict = None,
    ):
        super().__init__(observation_space, features_dim=1)

        # PointCloud
        vision_dim = vision_kwargs.get("dim", 3)
        vision_out_dim = vision_kwargs.get("out_dim", 64)
        self.vision_scale = vision_kwargs.get("scale", 1.0)
        vision_batchnorm = vision_kwargs.get("batchnorm", False)
        self.point_net_vision1 = PointNetFeatureExtractor(
            dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm
        )
        self.point_net_vision2 = PointNetFeatureExtractor(
            dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm
        )
        self.vision_feature_dim = vision_out_dim * 2
        # Tactile
        tac_dim = tac_kwargs.get("dim", 4)
        tac_out_dim = tac_kwargs.get("out_dim", 32)
        tac_batchnorm = tac_kwargs.get("batchnorm", False)
        self.point_net_tac = PointNetFeatureExtractor(
            dim=tac_dim, out_dim=tac_out_dim, batchnorm=tac_batchnorm
        )
        self.tac_feature_dim = tac_out_dim * 2

        self.layernorm_vision = nn.LayerNorm(self.vision_feature_dim)
        self.layernorm_tac = nn.LayerNorm(self.tac_feature_dim)

        self._features_dim = self.vision_feature_dim + self.tac_feature_dim

    def parse_obs(self, obs: dict):
        obs = obs.copy()
        marker_flow = obs["marker_flow"]
        point_cloud = torch.Tensor(obs["object_point_cloud"]) # [2, 2, 128, 3]

        unsqueezed = False
        if marker_flow.ndim == 4:
            assert point_cloud.ndim == 3
            marker_flow = torch.unsqueeze(marker_flow, 0)
            point_cloud = point_cloud.unsqueeze(0)
            unsqueezed = True

        fea = torch.cat([marker_flow[:, :, 0, ...], marker_flow[:, :, 1, ...]], dim=-1)
        tactile_left, tactile_right = (
            fea[:, 0],
            fea[:, 1],
        )  # (batch_size, marker_num, 4[u0,v0,u1,v1])

        point_cloud = point_cloud * self.vision_scale

        return tactile_left, tactile_right, point_cloud, unsqueezed

    def forward(self, obs):
        tactile_left, tactile_right, point_cloud, unsqueezed = self.parse_obs(obs)

        # the gripper is ignored here.
        vision_feature_1 = self.point_net_vision1(point_cloud[:, 0])  # object 1 # [2,128]
        vision_feature_2 = self.point_net_vision2(point_cloud[:, 1])  # object 2 # [2,128]
        vision_feature = torch.cat([vision_feature_1, vision_feature_2], dim=-1) # [2,256]

        tactile_left_feature = self.point_net_tac(tactile_left) # [2,64]
        tactile_right_feature = self.point_net_tac(tactile_right) # [2,64]
        tactile_feature = torch.cat(
            [tactile_left_feature, tactile_right_feature], dim=-1
        ) # [2,128]

        features = torch.cat(
            [
                self.layernorm_vision(vision_feature),
                self.layernorm_tac(tactile_feature),
            ],
            dim=-1,
        ) # [2,384]
        if unsqueezed:
            features = features.squeeze(0)
        return features
    
class FeaturesExtractorColoredPointCloud(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        vision_kwargs: dict = None,
        tac_kwargs: dict = None,
    ):
        super().__init__(observation_space, features_dim=1)

        # PointCloud
        vision_dim = vision_kwargs.get("dim", 6)  # 3D坐标 + RGB颜色
        vision_out_dim = vision_kwargs.get("out_dim", 64)
        self.vision_scale = vision_kwargs.get("scale", 1.0)
        vision_batchnorm = vision_kwargs.get("batchnorm", False)
        self.point_net_vision = PointNetFeatureExtractor(
            dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm
        )
        self.vision_feature_dim = vision_out_dim
        
        # 红外相机内参 - 转换为tensor
        self.fx = torch.tensor(595.8051147460938)
        self.fy = torch.tensor(595.8051147460938)
        self.cx = torch.tensor(315.040283203125)
        self.cy = torch.tensor(246.26866149902344)
        
        # Tactile
        tac_dim = tac_kwargs.get("dim", 4)
        tac_out_dim = tac_kwargs.get("out_dim", 32)
        tac_batchnorm = tac_kwargs.get("batchnorm", False)
        self.point_net_tac = PointNetFeatureExtractor(
            dim=tac_dim, out_dim=tac_out_dim, batchnorm=tac_batchnorm
        )
        self.tac_feature_dim = tac_out_dim * 2

        self.layernorm_vision = nn.LayerNorm(self.vision_feature_dim)
        self.layernorm_tac = nn.LayerNorm(self.tac_feature_dim)

        self._features_dim = self.vision_feature_dim + self.tac_feature_dim

        self.target_size = 8192

    def depth_image_to_point_cloud(self, depth_image, color_image, mask=None):
        """将深度图和RGB图转换为彩色点云，全部使用tensor运算"""
        # 确保输入是tensor
        if not isinstance(depth_image, torch.Tensor):
            depth_image = torch.tensor(depth_image, dtype=torch.float32)
        if not isinstance(color_image, torch.Tensor):
            color_image = torch.tensor(color_image, dtype=torch.float32)
            if color_image.dtype == torch.uint8:
                color_image = color_image.float() / 255.0
        
        # 检查并调整颜色图像的格式
        if color_image.ndim == 3 and color_image.shape[0] == 3:
            # 如果颜色图像是[3, H, W]格式，转换为[H, W, 3]
            color_image = color_image.permute(1, 2, 0)
        
        height, width = depth_image.shape
        device = depth_image.device
        
        # 创建网格坐标
        v, u = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        Z = depth_image
        X = (u - self.cx.to(device)) * Z / self.fx.to(device)
        Y = (v - self.cy.to(device)) * Z / self.fy.to(device)
        
        # 创建点云
        point_cloud = torch.stack((X, Y, Z), dim=-1)
        
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, device=device)
            valid_points = (mask == 0) & (Z > 0)
            point_cloud = point_cloud[valid_points]
            color_points = color_image[valid_points]
            colored_point_cloud = torch.cat((point_cloud, color_points), dim=-1)
        else:
            valid_points = (Z > 0)
            point_cloud = point_cloud[valid_points]
            color_points = color_image[valid_points]
            colored_point_cloud = torch.cat((point_cloud, color_points), dim=-1)
            
        return colored_point_cloud

    def parse_obs(self, obs: dict):
        obs = obs.copy()
        marker_flow = obs["marker_flow"]
        
        # 从观测中获取RGB和深度图像
        rgb_images = obs["rgb_picture"]  # RGB图像 [batch_size, 3, 480, 640]
        depth_images = obs["depth_picture"]  # 深度图像 [batch_size, 480, 640]
        
        # 确保RGB和深度图像是tensor
        if not isinstance(rgb_images, torch.Tensor):
            rgb_images = torch.tensor(rgb_images, dtype=torch.float32)
            if rgb_images.dtype == torch.uint8:
                rgb_images = rgb_images.float() / 255.0
        
        if not isinstance(depth_images, torch.Tensor):
            depth_images = torch.tensor(depth_images, dtype=torch.float32)
        
        unsqueezed = False
        if marker_flow.ndim == 4:
            marker_flow = torch.unsqueeze(marker_flow, 0)
            if rgb_images.ndim == 3:  # [3, 480, 640]
                rgb_images = rgb_images.unsqueeze(0)  # [1, 3, 480, 640]
            if depth_images.ndim == 2:  # [480, 640]
                depth_images = depth_images.unsqueeze(0)  # [1, 480, 640]
            unsqueezed = True
            
        fea = torch.cat([marker_flow[:, :, 0, ...], marker_flow[:, :, 1, ...]], dim=-1)
        tactile_left, tactile_right = (
            fea[:, 0],
            fea[:, 1],
        )  # (batch_size, marker_num, 4[u0,v0,u1,v1])
        
        device = marker_flow.device
        batch_size = marker_flow.shape[0]
        
        # 批量处理所有图像，生成点云
        # 将RGB图像调整为正确的格式 [B, H, W, 3]
        if rgb_images.shape[1] == 3:  # [B, 3, H, W]
            rgb_images = rgb_images.permute(0, 2, 3, 1)  # 转为 [B, H, W, 3]
            
        # 获取图像尺寸
        B, H, W = depth_images.shape if depth_images.ndim == 3 else (1, depth_images.shape[0], depth_images.shape[1])
        
        # 创建批量网格坐标
        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        # 扩展坐标到批次维度 [B, H, W]
        u = u.expand(B, -1, -1)
        v = v.expand(B, -1, -1)
        
        # 计算3D坐标
        Z = depth_images
        X = (u - self.cx.to(device)) * Z / self.fx.to(device)
        Y = (v - self.cy.to(device)) * Z / self.fy.to(device)
        
        # 创建点云 [B, H, W, 3]
        point_cloud_xyz = torch.stack((X, Y, Z), dim=-1)
        
        # 创建有效点掩码 [B, H, W]
        valid_depth_mask = (Z > 0)
        
        # 创建颜色掩码来过滤背景（白色）
        # 将RGB图像转换为适合处理的格式
        rgb_for_mask = rgb_images
        if not isinstance(rgb_for_mask, torch.Tensor):
            rgb_for_mask = torch.tensor(rgb_for_mask, device=device)

        # 可视化mask - Debug
        if False:
            mask_vis = rgb_for_mask[0].cpu().numpy()
            cv2.imshow("mask", mask_vis)
            cv2.waitKey(0)
        
        # 创建白色背景掩码 (接近白色的像素)
        # 检测RGB值都接近255的像素
        white_threshold = 0.9 if rgb_for_mask.max() <= 1.0 else 230  # 根据RGB范围调整阈值
        color_mask = (rgb_for_mask[..., 0] > white_threshold) & \
                     (rgb_for_mask[..., 1] > white_threshold) & \
                     (rgb_for_mask[..., 2] > white_threshold)
        
        # 初始化点云列表
        all_point_clouds = []
        
        # 对每个批次处理点云
        for b in range(B):
            # 结合深度有效性和颜色掩码
            # 保留深度有效且不是白色背景的点
            valid_b = valid_depth_mask[b] & (~color_mask[b])
            
            # 提取当前批次的有效点
            xyz_b = point_cloud_xyz[b][valid_b]
            rgb_b = rgb_images[b][valid_b]
            
            # 检查是否有足够的有效点
            if xyz_b.shape[0] < 10:  # 设置一个最小阈值
                # 如果有效点太少，使用所有深度有效的点（忽略颜色掩码）
                valid_b = valid_depth_mask[b]
                xyz_b = point_cloud_xyz[b][valid_b]
                rgb_b = rgb_images[b][valid_b]
                
                # 如果仍然太少，创建一个占位符点云
                if xyz_b.shape[0] < 10:
                    placeholder = torch.zeros((self.target_size, 6), device=device)
                    all_point_clouds.append(placeholder)
                    continue
            
            # 合并XYZ和RGB
            colored_points = torch.cat([xyz_b, rgb_b], dim=-1)
            
            # 归一化XYZ坐标
            xyz_min = xyz_b.min(dim=0)[0]
            xyz_max = xyz_b.max(dim=0)[0]
            xyz_range = xyz_max - xyz_min
            xyz_range[xyz_range == 0] = 1.0
            
            # 归一化到[-1, 1]范围
            normalized_xyz = 2 * (xyz_b - xyz_min) / xyz_range - 1
            
            # 更新点云中的XYZ坐标
            colored_points[:, :3] = normalized_xyz
            
            # 采样固定数量的点
            if colored_points.shape[0] > self.target_size:
                indices = torch.randperm(colored_points.shape[0], device=device)[:self.target_size]
                colored_points = colored_points[indices]
            elif colored_points.shape[0] < self.target_size:
                # 如果点太少，通过重复来填充
                indices = torch.randint(0, colored_points.shape[0], (self.target_size - colored_points.shape[0],), device=device)
                colored_points = torch.cat([colored_points, colored_points[indices]], dim=0)
            
            all_point_clouds.append(colored_points)

            # 可视化点云 - Debug
            if False:
                colored_points = colored_points.cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(colored_points[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(colored_points[:, 3:])
                o3d.visualization.draw_geometries([pcd])
        
        # 堆叠所有批次的点云
        point_clouds = torch.stack(all_point_clouds)
        
        return tactile_left, tactile_right, point_clouds, unsqueezed

    def forward(self, obs):
        tactile_left, tactile_right, point_clouds, unsqueezed = self.parse_obs(obs)

        # 批量处理所有点云
        batch_size = point_clouds.shape[0]
        
        # 处理触觉特征 - 直接使用批量处理
        tactile_left_feature = self.point_net_tac(tactile_left)  # [batch_size, out_dim]
        tactile_right_feature = self.point_net_tac(tactile_right)  # [batch_size, out_dim]
        tactile_feature = torch.cat(
            [tactile_left_feature, tactile_right_feature], dim=-1
        )  # [batch_size, out_dim*2]
        
        # 处理视觉特征 - 使用批量处理
        # 注意：PointNet可能需要修改以支持批量处理
        # 如果PointNet不支持批量处理，可以使用以下替代方法
        # all_vision_features = []
        # for i in range(batch_size):
        #     vision_feature_i = self.point_net_vision(point_clouds[i])
        #     all_vision_features.append(vision_feature_i)
        # vision_feature = torch.stack(all_vision_features) # [batch_size, 1, 128]
        vision_feature = self.point_net_vision(point_clouds)
        
        # 合并特征
        features = torch.cat(
            [
                self.layernorm_vision(vision_feature),
                self.layernorm_tac(tactile_feature),
            ],
            dim=-1,
        )  # [batch_size, vision_dim + tac_dim*2]
        
        if unsqueezed:
            features = features.squeeze(0)
            
        return features
    
class FeaturesExtractorProjectedPointCloud(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        vision_kwargs: dict = None,
        tac_kwargs: dict = None,
    ):
        super().__init__(observation_space, features_dim=1)

        # PointCloud - 只使用xyz坐标
        vision_dim = vision_kwargs.get("dim", 3)  # 只有3D坐标，没有RGB
        vision_out_dim = vision_kwargs.get("out_dim", 64)
        vision_batchnorm = vision_kwargs.get("batchnorm", False)
        self.point_net_vision = PointNetFeatureExtractor(
            dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm
        )
        self.vision_feature_dim = vision_out_dim
        
        # 红外相机内参
        self.fx = 595.8051147460938
        self.fy = 595.8051147460938
        self.cx = 315.040283203125
        self.cy = 246.26866149902344
        
        # Tactile
        tac_dim = tac_kwargs.get("dim", 4)
        tac_out_dim = tac_kwargs.get("out_dim", 32)
        tac_batchnorm = tac_kwargs.get("batchnorm", False)
        self.point_net_tac = PointNetFeatureExtractor(
            dim=tac_dim, out_dim=tac_out_dim, batchnorm=tac_batchnorm
        )
        self.tac_feature_dim = tac_out_dim * 2

        self.layernorm_vision = nn.LayerNorm(self.vision_feature_dim)
        self.layernorm_tac = nn.LayerNorm(self.tac_feature_dim)

        self._features_dim = self.vision_feature_dim + self.tac_feature_dim

        self.scale = 0.2
        
        # 固定点云点数
        self.target_size = 8192

    def get_pcd_from_rgbd(self, depth, rgb, mask):
        """从深度图和RGB图像生成点云，但只提取XYZ坐标"""
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        
        height, width = depth.shape 
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        # 从深度图创建点云 (X, Y, Z)
        point_cloud = np.dstack((X, Y, Z))
        point_cloud = point_cloud[mask]
        
        return point_cloud
    
    def process_point_cloud(self, point_cloud, target_size):
        """处理点云以确保固定大小，类似于policies.py中的process_point_cloud"""
        original_size = point_cloud.shape[0]
        
        if original_size > target_size:
            # 下采样
            index = np.random.choice(np.arange(0, point_cloud.shape[0]), size=target_size, replace=False)
            processed_point_cloud = point_cloud[index]
        elif original_size == 0:
            # 如果没有点，创建全零点云
            processed_point_cloud = np.zeros((target_size, point_cloud.shape[1]))
        else:
            # 上采样 - 使用最近邻插值
            original_indices = np.arange(original_size)
            target_indices = np.linspace(0, original_size - 1, target_size)
            interpolated_points = []
            for i in range(point_cloud.shape[1]):
                interp_func = np.interp(target_indices, original_indices, point_cloud[:, i])
                interpolated_points.append(interp_func)
                
            processed_point_cloud = np.vstack(interpolated_points).T
            
        return processed_point_cloud

    def parse_obs(self, obs: dict):
        obs = obs.copy()
        marker_flow = obs["marker_flow"]
        rgb_images = obs["rgb_picture"]
        
        # 获取深度图像
        depth_images = obs["depth_picture"]
        
        unsqueezed = False
        if marker_flow.ndim == 4:
            marker_flow = torch.unsqueeze(marker_flow, 0)
            if isinstance(depth_images, torch.Tensor) and depth_images.ndim == 2:
                depth_images = depth_images.unsqueeze(0)
            elif not isinstance(depth_images, torch.Tensor):
                depth_images = torch.tensor(depth_images, dtype=torch.float32).unsqueeze(0)
            # 确保RGB图像也有批次维度
            if isinstance(rgb_images, torch.Tensor) and len(rgb_images.shape) == 3:  # [C,H,W]
                rgb_images = rgb_images.unsqueeze(0)  # [1,C,H,W]
            unsqueezed = True
            
        fea = torch.cat([marker_flow[:, :, 0, ...], marker_flow[:, :, 1, ...]], dim=-1)
        tactile_left, tactile_right = (
            fea[:, 0],
            fea[:, 1],
        )  # (batch_size, marker_num, 4[u0,v0,u1,v1])
        
        device = marker_flow.device
        batch_size = marker_flow.shape[0]
        
        # 初始化点云列表
        all_point_clouds = []
        
        # 对每个批次处理点云
        for b in range(batch_size):
            # 提取当前批次的深度图
            if isinstance(depth_images, torch.Tensor):
                if depth_images.ndim > 2:  # 有批次维度
                    depth_image = depth_images[b].cpu().numpy()
                else:  # 没有批次维度
                    depth_image = depth_images.cpu().numpy()
            else:
                # 非tensor类型
                if batch_size == 1:
                    depth_image = depth_images
                else:
                    depth_image = depth_images[b]
            
            # 提取当前批次的RGB图像
            if isinstance(rgb_images, torch.Tensor):
                if rgb_images.ndim > 3:  # 有批次维度 [B,C,H,W]
                    rgb_image = rgb_images[b].cpu().numpy()
                else:  # 没有批次维度 [C,H,W]
                    rgb_image = rgb_images.cpu().numpy()
            else:
                # 非tensor类型
                if batch_size == 1:
                    rgb_image = rgb_images
                else:
                    rgb_image = rgb_images[b]

            # 确保RGB形状正确
            if len(rgb_image.shape) == 3:  # 有通道维度
                if rgb_image.shape[0] == 3:  # [C,H,W]格式
                    rgb_image = np.transpose(rgb_image, (1, 2, 0))  # 转为[H,W,C]

            mask = cv2.inRange(rgb_image, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
            mask = (mask == 0) & (depth_image > 0)
            
            # 使用RGB和深度图生成点云（只有XYZ坐标）
            pcd = self.get_pcd_from_rgbd(depth_image, rgb_image, mask)
            pcd_raw = pcd.copy()
            
            # 如果点云为空，创建占位符
            if pcd.shape[0] < 10:
                placeholder = np.zeros((self.target_size, 3))
                all_point_clouds.append(torch.tensor(placeholder, device=device))
                continue
                
            # 尺度和中心化处理
            xyz = pcd / self.scale
            xyz_mean = np.mean(xyz, axis=0)
            
            # 计算到中心的距离并过滤掉远点
            distances = np.linalg.norm(xyz - xyz_mean, axis=1)
            xyz = xyz[distances <= 0.5]
            
            # 如果过滤后点云为空，创建占位符
            if xyz.shape[0] < 10:
                placeholder = np.zeros((self.target_size, 3))
                all_point_clouds.append(torch.tensor(placeholder, device=device))
                continue
                
            # 重新计算中心并中心化
            xyz_mean = np.mean(xyz, axis=0)
            xyz = xyz - xyz_mean
            
            # 处理点云大小
            xyz = self.process_point_cloud(xyz, self.target_size)

            # 可视化点云 - Debug
            if False:
                raw_pcd_open3d = o3d.geometry.PointCloud()
                raw_pcd_open3d.points = o3d.utility.Vector3dVector(pcd_raw[:, :3])
                o3d.visualization.draw_geometries([raw_pcd_open3d])
                processed_pcd_open3d = o3d.geometry.PointCloud()
                processed_pcd_open3d.points = o3d.utility.Vector3dVector(xyz[:, :3])
                o3d.visualization.draw_geometries([processed_pcd_open3d])
                # 保存点云
                o3d.io.write_point_cloud(f"raw_pcd_policy.ply", raw_pcd_open3d)
                o3d.io.write_point_cloud(f"processed_pcd_policy.ply", processed_pcd_open3d)
                exit()
            
            # 转换为tensor
            pcd_tensor = torch.tensor(xyz, dtype=torch.float32, device=device)
            
            all_point_clouds.append(pcd_tensor)

        # 堆叠所有批次的点云
        point_clouds = torch.stack(all_point_clouds)
        
        return tactile_left, tactile_right, point_clouds, unsqueezed

    def forward(self, obs):
        tactile_left, tactile_right, point_clouds, unsqueezed = self.parse_obs(obs)

        # 处理触觉特征
        tactile_left_feature = self.point_net_tac(tactile_left)  # [batch_size, out_dim]
        tactile_right_feature = self.point_net_tac(tactile_right)  # [batch_size, out_dim]
        tactile_feature = torch.cat(
            [tactile_left_feature, tactile_right_feature], dim=-1
        )  # [batch_size, out_dim*2]
        
        # 处理视觉特征
        vision_feature = self.point_net_vision(point_clouds)
        
        # 合并特征
        features = torch.cat(
            [
                self.layernorm_vision(vision_feature),
                self.layernorm_tac(tactile_feature),
            ],
            dim=-1,
        )  # [batch_size, vision_dim + tac_dim*2]
        
        if unsqueezed:
            features = features.squeeze(0)
            
        return features

class FeaturesExtractorXYZ(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        vision_kwargs: dict = None,
        tac_kwargs: dict = None,
    ):
        super().__init__(observation_space, features_dim=1)

        # PointCloud - 只使用xyz坐标
        vision_dim = vision_kwargs.get("dim", 3)  # 只有3D坐标，没有RGB
        vision_out_dim = vision_kwargs.get("out_dim", 64)
        vision_batchnorm = vision_kwargs.get("batchnorm", False)
        self.point_net_vision = PointNetFeatureExtractor(
            dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm
        )
        self.vision_feature_dim = vision_out_dim
        
        # 红外相机内参
        self.fx = 595.8051147460938
        self.fy = 595.8051147460938
        self.cx = 315.040283203125
        self.cy = 246.26866149902344
        
        self.layernorm_vision = nn.LayerNorm(self.vision_feature_dim)

        self._features_dim = self.vision_feature_dim

        self.scale = 0.2
        
        # 固定点云点数
        self.target_size = 8192

    def get_pcd_from_rgbd(self, depth, rgb, mask):
        """从深度图和RGB图像生成点云，但只提取XYZ坐标"""
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        
        height, width = depth.shape 
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        # 从深度图创建点云 (X, Y, Z)
        point_cloud = np.dstack((X, Y, Z))
        point_cloud = point_cloud[mask]
        
        return point_cloud
    
    def process_point_cloud(self, point_cloud, target_size):
        """处理点云以确保固定大小，类似于policies.py中的process_point_cloud"""
        original_size = point_cloud.shape[0]
        
        if original_size > target_size:
            # 下采样
            index = np.random.choice(np.arange(0, point_cloud.shape[0]), size=target_size, replace=False)
            processed_point_cloud = point_cloud[index]
        elif original_size == 0:
            # 如果没有点，创建全零点云
            processed_point_cloud = np.zeros((target_size, point_cloud.shape[1]))
        else:
            # 上采样 - 使用最近邻插值
            original_indices = np.arange(original_size)
            target_indices = np.linspace(0, original_size - 1, target_size)
            interpolated_points = []
            for i in range(point_cloud.shape[1]):
                interp_func = np.interp(target_indices, original_indices, point_cloud[:, i])
                interpolated_points.append(interp_func)
                
            processed_point_cloud = np.vstack(interpolated_points).T
            
        return processed_point_cloud

    def parse_obs(self, obs: dict):
        obs = obs.copy()
        rgb_images = obs["rgb_picture"]
        
        # 获取深度图像
        depth_images = obs["depth_picture"]
        
        unsqueezed = False
        # 首先确定RGB图像的格式和维度
        if isinstance(rgb_images, torch.Tensor):
            # 处理PyTorch张量格式的RGB图像
            if rgb_images.ndim == 3:  # 形状为(C,H,W)，需要增加批次维度
                rgb_images = rgb_images.unsqueeze(0)  # (1,C,H,W)
                unsqueezed = True
            # 此时rgb_images应该是(batch_size,C,H,W)格式
        else:
            # 处理numpy数组格式的RGB图像
            rgb_images = np.array(rgb_images)
            if rgb_images.ndim == 3:  # 形状为(C,H,W)或(H,W,C)
                rgb_images = np.expand_dims(rgb_images, axis=0)  # 增加批次维度
                unsqueezed = True
        
        # 确保深度图像也有相同的批次维度
        if isinstance(depth_images, torch.Tensor):
            if depth_images.ndim == 2 and unsqueezed:  # 单个深度图
                depth_images = depth_images.unsqueeze(0)
        elif unsqueezed:  # 深度图是numpy数组
            depth_images = np.array(depth_images)
            if depth_images.ndim == 2:  # 单个深度图
                depth_images = np.expand_dims(depth_images, axis=0)
        
        # 确定设备和批次大小
        device = rgb_images.device if isinstance(rgb_images, torch.Tensor) else torch.device('cpu')
        batch_size = rgb_images.shape[0]
        
        # 初始化点云列表
        all_point_clouds = []
        
        # 对每个批次处理点云
        for b in range(batch_size):
            # 提取当前批次的深度图
            if isinstance(depth_images, torch.Tensor):
                depth_image = depth_images[b].cpu().numpy()
            else:
                depth_image = depth_images[b]
            
            # 提取当前批次的RGB图像
            if isinstance(rgb_images, torch.Tensor):
                rgb_image = rgb_images[b].cpu().numpy()
            else:
                rgb_image = rgb_images[b]

            # 确保RGB形状正确 (H,W,C) 适合OpenCV处理
            if len(rgb_image.shape) == 3:
                if rgb_image.shape[0] == 3:  # (C,H,W)格式
                    rgb_image = np.transpose(rgb_image, (1, 2, 0))  # 转为(H,W,C)
            
            # 确保RGB图像格式适合cv2.inRange处理
            if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                if rgb_image.max() <= 1.0:  # [0,1]范围
                    mask = cv2.inRange(rgb_image, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
                else:  # [0,255]范围
                    mask = cv2.inRange(rgb_image, (255, 255, 255), (255, 255, 255))
            else:  # 可能是uint8类型
                mask = cv2.inRange(rgb_image, (255, 255, 255), (255, 255, 255))
            
            mask = (mask == 0) & (depth_image > 0)
            
            # 使用RGB和深度图生成点云（只有XYZ坐标）
            pcd = self.get_pcd_from_rgbd(depth_image, rgb_image, mask)
            pcd_raw = pcd.copy()
            
            # 如果点云为空，创建占位符
            if pcd.shape[0] < 10:
                placeholder = np.zeros((self.target_size, 3))
                all_point_clouds.append(torch.tensor(placeholder, device=device))
                continue
                
            # 尺度和中心化处理
            xyz = pcd / self.scale
            xyz_mean = np.mean(xyz, axis=0)
            
            # 计算到中心的距离并过滤掉远点
            distances = np.linalg.norm(xyz - xyz_mean, axis=1)
            xyz = xyz[distances <= 0.5]
            
            # 如果过滤后点云为空，创建占位符
            if xyz.shape[0] < 10:
                placeholder = np.zeros((self.target_size, 3))
                all_point_clouds.append(torch.tensor(placeholder, device=device))
                continue
                
            # 重新计算中心并中心化
            xyz_mean = np.mean(xyz, axis=0)
            xyz = xyz - xyz_mean
            
            # 处理点云大小
            xyz = self.process_point_cloud(xyz, self.target_size)

            # 可视化点云 - Debug
            if False:
                raw_pcd_open3d = o3d.geometry.PointCloud()
                raw_pcd_open3d.points = o3d.utility.Vector3dVector(pcd_raw[:, :3])
                o3d.visualization.draw_geometries([raw_pcd_open3d])
                processed_pcd_open3d = o3d.geometry.PointCloud()
                processed_pcd_open3d.points = o3d.utility.Vector3dVector(xyz[:, :3])
                o3d.visualization.draw_geometries([processed_pcd_open3d])
                # 保存点云
                o3d.io.write_point_cloud(f"raw_pcd_policy.ply", raw_pcd_open3d)
                o3d.io.write_point_cloud(f"processed_pcd_policy.ply", processed_pcd_open3d)
                exit()
            
            # 转换为tensor
            pcd_tensor = torch.tensor(xyz, dtype=torch.float32, device=device)
            
            all_point_clouds.append(pcd_tensor)

        # 堆叠所有批次的点云
        point_clouds = torch.stack(all_point_clouds)
        
        return point_clouds, unsqueezed

    def forward(self, obs):
        point_clouds, unsqueezed = self.parse_obs(obs)

        # 处理视觉特征
        vision_feature = self.point_net_vision(point_clouds)
        
        # 合并特征
        features = torch.cat(
            [
                self.layernorm_vision(vision_feature),
            ],
            dim=-1,
        )  # [batch_size, vision_dim]
        
        if unsqueezed:
            features = features.squeeze(0)
            
        return features
