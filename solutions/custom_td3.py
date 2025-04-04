from typing import Dict, Type, TypeVar, ClassVar

import torch as th
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import (
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
)

from solutions.policies import TD3PolicyForPegInsertionV2_test
from solutions.actor_and_critics import PointNetActorTest, CustomCritic

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class CustomTD3(TD3):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "TD3Policy_test": TD3PolicyForPegInsertionV2_test,
    }

    actor: PointNetActorTest
    actor_target: PointNetActorTest
    critic: CustomCritic
    critic_target: CustomCritic

    def __init__(
        self,
        *args,
        gamma: float = 0.99,  # 折扣因子
        binary_flag: bool = True,  # 是否添加二元分类损失
        spatial_flag: bool = True,  # 是否添加目标点关系损失
        max_steps: int = 100000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.binary_flag = binary_flag
        self.spatial_flag = spatial_flag
        self.max_steps = max_steps

    def get_binary_weight(
        self,
        step: int,
        base_weight: float = 100.0,
        decay_rate: float = 20000.0,
        gamma: float = 0.99,
    ) -> float:
        """
        根据训练步骤动态调整二元分类损失的权重。

        :param step: 当前训练步骤
        :param base_weight: 基础权重
        :param decay_rate: 衰减率
        :param gamma: 折扣因子
        :return: 动态权重
        """
        return base_weight * (gamma ** (step // decay_rate))

    def get_spatial_weight(
        self,
        step: int,
        base_weight: float = 100.0,
        decay_rate: float = 20000.0,
        gamma: float = 0.99,
    ) -> float:
        """
        根据训练步骤动态调整目标点关系损失的权重。

        :param step: 当前训练步骤
        :param base_weight: 基础权重
        :param decay_rate: 衰减率
        :param gamma: 折扣因子
        :return: 动态权重
        """
        return base_weight * (gamma ** (step // decay_rate))

    def get_linear_weight(
        self,
        step: int,
        max_steps: int,
        base_weight: float = 1.0,
    ) -> float:
        """
        根据训练步骤线性衰减权重。

        :param step: 当前训练步骤
        :param max_steps: 最大训练步骤
        :param base_weight: 基础权重
        :return: 线性衰减后的权重
        """
        return base_weight * (1 - step / max_steps)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        actor_aux_binary_losses, actor_aux_spatial_losses = [], []
        critic_aux_binary_losses, critic_aux_spatial_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(
                    0, self.target_policy_noise
                )
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions)[0],
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            current_q_values, binary_pred, spatial_pred = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )

            # Compute auxiliary losses
            binary_target = replay_data.observations[
                "gt_direction"
            ]  # 假设观测值中包含二元分类目标
            spatial_target = replay_data.observations[
                "gt_offset"
            ]  # 假设观测值中包含目标点关系目标
            binary_loss = F.binary_cross_entropy(binary_pred, binary_target)
            spatial_loss = F.mse_loss(spatial_pred, spatial_target)

            # Dynamic weights for auxiliary losses
            # binary_weight = self.get_binary_weight(
            #     self._n_updates, base_weight=100.0, decay_rate=20000.0, gamma=self.gamma
            # )
            # spatial_weight = self.get_spatial_weight(
            #     self._n_updates, base_weight=100.0, decay_rate=20000.0, gamma=self.gamma
            # )
            offset_weight = self.get_linear_weight(
                self._n_updates, self.max_steps, base_weight=3
            )

            # Add auxiliary losses based on flags
            if self.binary_flag:
                critic_loss += offset_weight * binary_loss
            if self.spatial_flag:
                critic_loss += offset_weight * spatial_loss

            critic_losses.append(critic_loss.item())
            if self.binary_flag:
                critic_aux_binary_losses.append(binary_loss.item())
            if self.spatial_flag:
                critic_aux_spatial_losses.append(spatial_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actions_pi = self.actor(replay_data.observations)
                binary_pred = self.actor.forward_binary()
                spatial_pred = self.actor.forward_spatial()

                q_values = self.critic.q1_forward(replay_data.observations, actions_pi)
                actor_loss = -q_values.mean()

                # Compute auxiliary losses
                binary_loss = F.binary_cross_entropy(binary_pred, binary_target)
                spatial_loss = F.mse_loss(spatial_pred, spatial_target)

                # Add auxiliary losses based on flags
                if self.binary_flag:
                    actor_loss += offset_weight * binary_loss
                if self.spatial_flag:
                    actor_loss += offset_weight * spatial_loss

                actor_losses.append(actor_loss.item())
                if self.binary_flag:
                    actor_aux_binary_losses.append(binary_loss.item())
                if self.spatial_flag:
                    actor_aux_spatial_losses.append(spatial_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau
                )
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0
                )

        # Log losses
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            if self.binary_flag:
                self.logger.record(
                    "train/actor_binary_loss", np.mean(actor_aux_binary_losses)
                )
            if self.spatial_flag:
                self.logger.record(
                    "train/actor_spatial_loss", np.mean(actor_aux_spatial_losses)
                )
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if self.binary_flag:
            self.logger.record(
                "train/critic_binary_loss", np.mean(critic_aux_binary_losses)
            )
        if self.spatial_flag:
            self.logger.record(
                "train/critic_spatial_loss", np.mean(critic_aux_spatial_losses)
            )
