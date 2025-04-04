from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import numpy as np


class PointNetFeaNew(nn.Module):
    def __init__(self, point_dim, net_layers: List, batchnorm=False):
        super(PointNetFeaNew, self).__init__()
        self.layer_num = len(net_layers)
        self.conv0 = nn.Conv1d(point_dim, net_layers[0], 1)
        self.bn0 = nn.BatchNorm1d(net_layers[0]) if batchnorm else nn.Identity()
        for i in range(0, self.layer_num - 1):
            self.__setattr__(
                f"conv{i + 1}", nn.Conv1d(net_layers[i], net_layers[i + 1], 1)
            )
            self.__setattr__(
                f"bn{i + 1}",
                nn.BatchNorm1d(net_layers[i + 1]) if batchnorm else nn.Identity(),
            )

        self.output_dim = net_layers[-1]

    def forward(self, x):
        for i in range(0, self.layer_num - 1):
            x = F.relu(self.__getattr__(f"bn{i}")(self.__getattr__(f"conv{i}")(x)))
        x = self.__getattr__(f"bn{self.layer_num - 1}")(
            self.__getattr__(f"conv{self.layer_num - 1}")(x)
        )
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)
        return x


class PointNetFeatureExtractor(nn.Module):
    """
    this is a latent feature extractor for point cloud data
    need to distinguish this from other modules defined in feature_extractors.py
    those modules are only used to extract the corresponding input (e.g. point flow, manual feature, etc.) from original observations
    """

    def __init__(self, dim, out_dim, batchnorm=False):
        super(PointNetFeatureExtractor, self).__init__()
        self.dim = dim

        self.pointnet_local_feature_num = 64
        self.pointnet_global_feature_num = 512

        self.pointnet_local_fea = nn.Sequential(
            nn.Conv1d(dim, self.pointnet_local_feature_num, 1),
            (
                nn.BatchNorm1d(self.pointnet_local_feature_num)
                if batchnorm
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.pointnet_local_feature_num, self.pointnet_local_feature_num, 1
            ),
            (
                nn.BatchNorm1d(self.pointnet_local_feature_num)
                if batchnorm
                else nn.Identity()
            ),
            nn.ReLU(),
        )
        self.pointnet_global_fea = PointNetFeaNew(
            self.pointnet_local_feature_num,
            [64, 128, self.pointnet_global_feature_num],
            batchnorm=batchnorm,
        )

        self.mlp_output = nn.Sequential(
            nn.Linear(self.pointnet_global_feature_num, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, marker_pos):
        """
        :param marker_pos: Tensor, size (batch, num_points, 4)
        :return:
        """
        if marker_pos.ndim == 2:
            marker_pos = torch.unsqueeze(marker_pos, dim=0)

        marker_pos = torch.transpose(marker_pos, 1, 2)
        local_feature = self.pointnet_local_fea(
            marker_pos
        )  # (batch_num, self.pointnet_local_feature_num, point_num)
        # shape: (batch, step * 2, num_points)
        global_feature = self.pointnet_global_fea(local_feature).view(
            -1, self.pointnet_global_feature_num
        )  # (batch_num, self.pointnet_global_feature_num)

        pred = self.mlp_output(global_feature)
        # pred shape: (batch_num, out_dim)
        return pred


class RGBPegClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(RGBPegClassifier, self).__init__()
        # 使用预训练的权重
        self.resnet_r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 冻结所有参数
        for param in self.resnet_r.parameters():
            param.requires_grad = False
            
        # 修改最后的全连接层
        f1 = self.resnet_r.fc.in_features
        self.resnet_r.fc = nn.Identity()  # 移除原始的fc层

        # 使用预训练的权重
        self.resnet_d = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 冻结所有参数 
        for param in self.resnet_d.parameters():
            param.requires_grad = False
            
        # 修改最后的全连接层
        f2 = self.resnet_d.fc.in_features
        self.resnet_d.fc = nn.Identity()  # 移除原始的fc层
        
        self.num_features = f1 + f2
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # 冻结分类器参数
        for param in self.classifier.parameters():
            param.requires_grad = False

    def forward(self, image, depth):
        rgb_features = self.resnet_r(image)
        depth_features = self.resnet_d(depth)
        features = torch.cat((rgb_features, depth_features), dim=1)
        return self.classifier(features)


class OffsetPrediction(nn.Module):
    def __init__(self, num_classes=2, embedding_dim=8, out_dim=4):
        super(OffsetPrediction, self).__init__()

        # 初始化分类器
        self.classifier = RGBPegClassifier(num_classes=num_classes)
        for param in list(self.classifier.parameters()):
            param.requires_grad = False
        
        self.embedding_layer = nn.Embedding(num_classes, embedding_dim)

        # 确保ResNet在创建时就加载到正确的设备
        self.resnet_r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in list(self.resnet_r.parameters())[:-4]:
            param.requires_grad = False
        f1 = self.resnet_r.fc.in_features
        self.resnet_r.fc = nn.Identity()

        self.resnet_d = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in list(self.resnet_d.parameters())[:-4]:
            param.requires_grad = False
        f2 = self.resnet_d.fc.in_features
        self.resnet_d.fc = nn.Identity()
        
        self.num_features = f1 + f2 + embedding_dim
        self.offset_head = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )
        
        # 确保所有子模块都在同一设备上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, image, depth):
        prediction = self.classifier(image, depth)
        _, peg_id = torch.max(prediction.data, 1)

        class_embedding = self.embedding_layer(peg_id.long())

        rgb_features = self.resnet_r(image)
        depth_features = self.resnet_d(depth)

        features = torch.cat((rgb_features, depth_features), dim=1)
        features = torch.cat((features, class_embedding), dim=1)

        output = self.offset_head(features)
        
        return output, peg_id


class PointNet_Classifier(nn.Module):
    def __init__(
        self, vision_dim=6, vision_out_dim=64, vision_batchnorm=False, num_class=2):
        super().__init__()

        self.point_net_vision = PointNetFeatureExtractor(
            dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm
        )
        self.vision_feature_dim = vision_out_dim

        self.fc = nn.Sequential(
            nn.Linear(self.vision_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_class)
        )
  
    def forward(self, point_clouds):
        vision_feature = self.point_net_vision(point_clouds)  
        logits = self.fc(vision_feature)
            
        return logits


class PointCloudOffsetPrediction(nn.Module):
    def __init__(self, num_class=2, vision_dim=6, embedding_dim=8, vision_out_dim=64, out_dim=3):
        super(PointCloudOffsetPrediction, self).__init__()

        self.classifier = PointNet_Classifier(vision_dim=vision_dim, num_class=num_class)
        for param in list(self.classifier.parameters()):
            param.requires_grad = False
        
        self.embedding_layer = nn.Embedding(num_class, embedding_dim)

        self.point_net = PointNetFeatureExtractor(dim=vision_dim, out_dim=vision_out_dim)
        self.vision_feature_dim = vision_out_dim
        
        self.fea_dim = self.vision_feature_dim + embedding_dim
        self.offset_head = nn.Sequential(
            nn.Linear(self.fea_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )
        
        # 确保所有子模块都在同一设备上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, pcd):
        prediction = self.classifier(pcd)
        _, peg_id = torch.max(prediction.data, 1)

        class_embedding = self.embedding_layer(peg_id.long())

        pcd_fea = self.point_net(pcd)
        features = torch.cat((pcd_fea, class_embedding), dim=1)

        output = self.offset_head(features)
        
        return output, peg_id
