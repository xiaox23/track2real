from typing import Optional, Dict, Any

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy
import torch
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_action_dim
from solutions.networks import OffsetPrediction, PointCloudOffsetPrediction
from torchvision import transforms
from PIL import Image
import numpy as np
from solutions.actor_and_critics import CustomCritic
from solutions.feature_extractors import FeatureExtractorState, FeaturesExtractorColoredPointCloud
from PIL import Image
import cv2
from scipy.interpolate import interp1d
from loguru import logger as log

class TD3PolicyForPegInsertionV2(TD3Policy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(TD3PolicyForPegInsertionV2, self).__init__(*args, **kwargs)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, FeatureExtractorState(self.observation_space)
        )
        return CustomCritic(**critic_kwargs).to(self.device)

    
class TD3PolicyForPegInsertionV2ColoredPoints(TD3Policy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(TD3PolicyForPegInsertionV2ColoredPoints, self).__init__(*args, **kwargs)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, FeatureExtractorState(self.observation_space)
        )
        return CustomCritic(**critic_kwargs).to(self.device)
    
class TD3PolicyWithColoredPointsOffsetPrediction(nn.Module):
    """A wrapper policy that combines offset prediction using colored points with TD3"""
    
    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 net_arch=None,
                 activation_fn=nn.ReLU,
                 features_extractor_class=None,
                 features_extractor_kwargs=None,
                 normalize_images=True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_kwargs=None,
                 n_critics=2,
                 share_features_extractor=True,
                 offset_model_path="/home/tars/workspace/xx/tactile/STEIIA-PENTAC-3rd-commit/stg2_2nd/models/ofst_predictor.pth",
                 device='cuda:0',
                 target_size=8192):
        super().__init__()
        
        self.logger = log.bind(name="main")
        
        self.offset_predictor = PointCloudOffsetPrediction(num_class=2, vision_dim=3, embedding_dim=8, vision_out_dim=64, out_dim=3)
        if offset_model_path:
            self.offset_predictor.load_state_dict(torch.load(offset_model_path))
        self.offset_predictor.eval()
        
        self.target_size = target_size
        self.scale = 0.2
        
        td3_kwargs = dict(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )
        
        self.td3_policy = TD3PolicyForPegInsertionV2ColoredPoints(**td3_kwargs)
        
        self.first_step = True
        self.device = torch.device(device)
        
        # 添加新的状态变量
        self.cnt = 1
        self.offset_pred = None
        self.theta_process_times = 0
        self.x_process_times = 0
        self.y_process_times = 0
        self.theta_done = False
        self.x_done = False
        self.y_done = False
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.td3_policy.load_state_dict(state_dict)
        
    def to(self, device):
        self.device = device
        self.offset_predictor = self.offset_predictor.to(device)
        self.td3_policy = self.td3_policy.to(device)
        return self
        
    def forward(self, observation):
        action, _ = self.predict(observation)
        return action
    
    def get_pcd_from_rgbd(self, depth, rgb, mask):
        fx = 595.8051147460938
        fy = 595.8051147460938
        cx = 315.040283203125
        cy = 246.26866149902344
        height, width = depth.shape 
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        point_cloud = np.dstack((X, Y, Z))
        point_cloud = point_cloud[mask]
        rgb = rgb[mask]
        pcd = np.concatenate((point_cloud.reshape(-1, 3), rgb.reshape(-1, 3)), axis=-1)

        return pcd[:, :3]
    
    def process_point_cloud(self, point_cloud):
        original_size = point_cloud.shape[0]
        
        if original_size > self.target_size:
            index = np.random.choice(np.arange(0, point_cloud.shape[0]), size=self.target_size, replace=False)
            downsampled_point_cloud = point_cloud[index]
        else:
            original_indices = np.arange(original_size)
            target_indices = np.linspace(0, original_size - 1, self.target_size)
            interpolated_points = []
            for i in range(point_cloud.shape[1]):
                interp_func = interp1d(original_indices, point_cloud[:, i], kind='nearest', fill_value='extrapolate')
                interpolated_points.append(interp_func(target_indices))

            downsampled_point_cloud = np.vstack(interpolated_points).T

        return downsampled_point_cloud
        
    def _predict(self, observation, deterministic: bool = False):
        relative_motion = observation.get("relative_motion", None)
        if relative_motion is not None and torch.all(relative_motion == 0):
            self.first_step = True
            # 重置计数器和状态
            self.cnt = 1
            self.theta_process_times = 0
            self.x_process_times = 0
            self.y_process_times = 0
            self.theta_done = False
            self.x_done = False
            self.y_done = False
        
        if self.first_step:
            with torch.no_grad():
                image = observation.get("rgb_picture", None)
                depth = observation.get("depth_picture", None)
                
                if image is not None and depth is not None:
                    if isinstance(image, torch.Tensor):
                        image_np = image.cpu().numpy()
                    else:
                        image_np = image
                        
                    if isinstance(depth, torch.Tensor):
                        depth_np = depth.cpu().numpy()
                    else:
                        depth_np = depth

                    mask = cv2.inRange(image_np, (255, 255, 255), (255, 255, 255))
                    mask = (mask == 0) & (depth_np > 0)

                    masked_pcd = self.get_pcd_from_rgbd(depth_np, image_np, mask)
                    xyz = masked_pcd[:,:3]
                    xyz /= self.scale
                    xyz_mean = np.mean(xyz, axis=0)
                    distances = np.linalg.norm(xyz - xyz_mean, axis=1)
                    filter = distances <= 0.5
                    xyz = xyz[filter]
                    masked_pcd = masked_pcd[filter]
                    xyz_mean = np.mean(xyz, axis=0)
                    masked_pcd[:,:3] -= xyz_mean
                    processed_pcd = self.process_point_cloud(masked_pcd)

                    # 可视化点云 - Debug
                    if False:
                        import open3d as o3d
                        print("masked_pcd.shape: ", masked_pcd.shape)
                        print("processed_pcd.shape: ", processed_pcd.shape)
                        raw_pcd_open3d = o3d.geometry.PointCloud()
                        raw_pcd_open3d.points = o3d.utility.Vector3dVector(masked_pcd[:, :3])
                        raw_pcd_open3d.colors = o3d.utility.Vector3dVector(masked_pcd[:, 3:])
                        o3d.visualization.draw_geometries([raw_pcd_open3d])
                        processed_pcd_open3d = o3d.geometry.PointCloud()
                        processed_pcd_open3d.points = o3d.utility.Vector3dVector(processed_pcd[:, :3])
                        processed_pcd_open3d.colors = o3d.utility.Vector3dVector(processed_pcd[:, 3:])
                        o3d.visualization.draw_geometries([processed_pcd_open3d])
                        # 保存点云
                        o3d.io.write_point_cloud(f"pcd_raw_pred_input.ply", raw_pcd_open3d)
                        o3d.io.write_point_cloud(f"pcd_processed_pred_input.ply", processed_pcd_open3d)
                        exit()

                    
                    pcd_tensor = torch.tensor(processed_pcd, dtype=torch.float32).to(self.device)
                    
                    offset_pred, gt_direction = self.offset_predictor(pcd_tensor)
                    self.logger.info(f"offset_pred: {offset_pred.cpu().numpy()}")
                    self.logger.info(f"gt_direction: {gt_direction.cpu().numpy()}")
                    
                    # 保存offset_pred用于后续步骤
                    self.offset_pred = offset_pred.flatten()[:3]
                    
                    # 计算需要处理的次数
                    self.x_process_times = abs(int(self.offset_pred[0]))
                    self.y_process_times = abs(int(self.offset_pred[1]))
                    self.theta_process_times = abs(int(self.offset_pred[2]))
                    
                    # 第一步：处理theta
                    action = torch.zeros(4, device=self.device)
                    action[2] = -torch.sign(self.offset_pred[2])
                    action = action / torch.tensor([6.0, 30.0, 10.0, 2.0], device=self.device)
                    
                    self.first_step = False
                    self.cnt += 1
                    return action
                
                self.first_step = False
        else:
            # 处理后续步骤
            action = torch.zeros(4, device=self.device)
            
            if self.theta_process_times >= self.cnt:
                # 继续处理theta
                action[2] = -torch.sign(self.offset_pred[2])
            else:
                self.theta_done = True
                
            if self.theta_done:
                if self.x_process_times >= self.cnt - self.theta_process_times:
                    # 处理x
                    action[0] = -torch.sign(self.offset_pred[0])
                else:
                    self.x_done = True
                    
                if self.y_process_times >= self.cnt - self.theta_process_times:
                    # 处理y
                    action[1] = -torch.sign(self.offset_pred[1])
                else:
                    self.y_done = True
            
            if self.x_done and self.y_done and self.theta_done:
                # 所有偏移都已处理完，切换到TD3策略
                td3_action = self.td3_policy._predict(observation, deterministic)
                return td3_action / torch.tensor([1.0, 1.0, 1.0, 1.0], device=td3_action.device)
            
            self.cnt += 1
            return action / torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return self._predict(observation, deterministic), state
    

