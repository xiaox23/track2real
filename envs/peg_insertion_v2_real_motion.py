#! /usr/bin/python
import csv
import os
import sys
import time
from math import cos, sin
import ipdb

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)
import cv2

# import gym
import gymnasium as gym
import numpy as np
import rospy
from cv_bridge import CvBridge

# from gym.spaces import Box  # 按需合并导入
from gymnasium.spaces import Box  # 按需合并导入
from sensor_msgs.msg import Image
import tkinter as tk
from tkinter import messagebox
from utils.RL_data_save_load_helper import RlDataSaveLoadHelper
from gelsight_mini_ros.msg import judging_msg, tracking_msg
from gelsight_mini_ros.srv import ResetMarkerTracker
from envs.motion_manager_stage_v2 import MotionManagerStageV2
from utils.real_sense_help import RealSenseHelp
from utils.utils import ThreadSafeContainer
from utils.data_process_utils import adapt_marker_seq_to_unified_size
from utils.common import get_time
from utils.RL_common_utils import convert_observation_to_space
# from sri_force_sensor.msg import ForceAndTorque
import pyrealsense2 as rs
from loguru import logger as log
from path import Path
from utils.RL_common_utils import CustomMessageBox


bridge = CvBridge()

left_marker_flow_container = ThreadSafeContainer(max_size=30)
right_marker_flow_container = ThreadSafeContainer(max_size=30)
left_marker_image_container = ThreadSafeContainer(max_size=30)
right_marker_image_container = ThreadSafeContainer(max_size=30)
force_container = ThreadSafeContainer(max_size=30)
is_overforced = False
is_contact = False

force_data = None
FORCE_THRESHOLD = 100  # N    # 一般情况下peg环境不会出现问题，force sensor；是为了lock加的，所以这里可以把 threshold 调整大一点
TORQUE_THRESHOLD = 30  # Nm


def callback_sensor(data: judging_msg):
    global is_contact
    global is_overforced
    contact_msg = data
    is_overforced = contact_msg.is_overforced
    is_contact = contact_msg.is_contact


def callback_marker_flow_left(data: tracking_msg):
    global left_marker_flow_container
    marker_init_pos = np.stack([data.marker_x, data.marker_y]).transpose()
    marker_displacement = np.stack(
        [data.marker_displacement_x, data.marker_displacement_y]
    ).transpose()
    marker_cur_pos = marker_init_pos + marker_displacement
    marker_observation = np.stack([marker_init_pos, marker_cur_pos])
    left_marker_flow_container.put(marker_observation)


def callback_marker_flow_right(data: tracking_msg):
    global right_marker_flow_container
    marker_init_pos = np.stack([data.marker_x, data.marker_y]).transpose()
    marker_displacement = np.stack(
        [data.marker_displacement_x, data.marker_displacement_y]
    ).transpose()
    marker_cur_pos = marker_init_pos + marker_displacement
    marker_observation = np.stack([marker_init_pos, marker_cur_pos])
    right_marker_flow_container.put(marker_observation)


def callback_right_image(data: Image):
    global right_marker_image_container
    # bridge = CvBridge()
    right_image = bridge.imgmsg_to_cv2(data, "bgr8")
    right_marker_image_container.put(right_image)


def callback_left_image(data: Image):
    global left_marker_image_container
    # bridge = CvBridge()
    left_image = bridge.imgmsg_to_cv2(data, "bgr8")
    left_marker_image_container.put(left_image)


# def callback_force_sensor(data: ForceAndTorque):
#     # global force_data,
#     global force_container
#     # force_data = data
#     force_container.put(data)


class PegInsertionRealEnvV2(gym.Env):
    def __init__(
        self,
        motion_manager: MotionManagerStageV2,
        # max_error,
        # step_penalty,
        # final_reward,
        max_action_mm_deg,
        peg,  # hex
        max_steps,
        normalize,
        grasp_height_offset,
        # peg position
        # peg_x_max_offset_mm: float ,
        # peg_y_max_offset_mm: float ,
        # peg_theta_max_offset_deg: float ,
        # peg_dist_z_mm: float ,
        # peg_dist_z_diff_mm: float ,
        insertion_depth_mm: float,
        # for logging
        log_path=None,
        logger=None,
        # for vision
        vision_params: dict = None,
        # render_rgb
        render_rgb=False,
    ):
        # for logging
        time.sleep(np.random.rand(1)[0])
        if logger is None:
            self.logger = log
            self.logger.remove()
        else:
            self.logger = logger

        self.log_time = get_time()
        self.pid = os.getpid()
        if log_path is None:
            self.log_folder = track_path + "/envs/" + self.log_time
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)

        elif os.path.isdir(log_path):
            self.log_folder = log_path
        else:
            self.log_folder = Path(log_path).dirname()
        self.log_path = Path(
            os.path.join(
                self.log_folder,
                f"{self.log_time}_PegInsertionEnvV2.log",
            )
        )
        print(self.log_path)
        self.logger.add(
            self.log_path,
            filter=lambda record: record["extra"]["name"] == self.log_time,
        )
        self.unique_logger = self.logger.bind(name=self.log_time)
        # if not os.path.exists(self.log_folder + "/tracker_img/"):
        #     os.makedirs(self.log_folder + "/tracker_img/")

        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        self.msg_box = CustomMessageBox(root)

        # environment parameters
        self.current_episode = 0
        self.max_action_mm_deg = np.array(max_action_mm_deg)
        # self.step_penalty = step_penalty
        # self.final_reward = final_reward
        self.max_steps = max_steps
        self.current_episode_elapsed_steps = 0
        self.error_too_large = False
        self.too_many_steps = False
        self.tactile_movement_too_large = False
        self.current_episode_max_tactile_diff = 0
        self.current_episode_over = False

        self.render_rgb = render_rgb

        self.peg = peg
        self.target_insertion_depth_mm = insertion_depth_mm

        self.normalize_flow = normalize
        self.grasp_height_offset = grasp_height_offset

        # ROS communication
        rospy.init_node("peg_insertion_environment", anonymous=True)
        left_sensor_topic_name = "Marker_Tracking_Left"
        right_sensor_topic_name = "Marker_Tracking_Right"
        left_sensor_img_topic_name = "Tactile_Image_Left"
        right_sensor_img_topic_name = "Tactile_Image_Right"

        self.sub_tactile_fb = rospy.Subscriber(
            "Marker_Tracking_Contact",
            judging_msg,
            callback=callback_sensor,
            queue_size=10,
        )
        self.sub_tactile_marker_flow_L = rospy.Subscriber(
            left_sensor_topic_name,
            tracking_msg,
            callback=callback_marker_flow_left,
            queue_size=10,
        )
        self.sub_tactile_marker_flow_R = rospy.Subscriber(
            right_sensor_topic_name,
            tracking_msg,
            callback=callback_marker_flow_right,
            queue_size=10,
        )
        if self.render_rgb:
            self.sub_tactile_image_L = rospy.Subscriber(
                left_sensor_img_topic_name,
                Image,
                callback=callback_left_image,
                queue_size=10,
            )
            self.sub_tactile_image_R = rospy.Subscriber(
                right_sensor_img_topic_name,
                Image,
                callback=callback_right_image,
                queue_size=10,
            )
            global right_marker_image_container
            global left_marker_image_container

        self.left_sensor_init_marker_tracker_call = rospy.ServiceProxy(
            "Marker_Tracking_Srv_Left", ResetMarkerTracker
        )
        self.right_sensor_init_marker_tracker_call = rospy.ServiceProxy(
            "Marker_Tracking_Srv_Right", ResetMarkerTracker
        )
        global left_marker_flow_container
        global right_marker_flow_container
        global is_contact

        # vision related
        self.realsense = RealSenseHelp(self.peg)
        self.camera_size = (480, 640)
        self.vision_params = vision_params
        self.max_points = self.vision_params["max_points"]

        # motion stage related
        # self.msg_box.show_message("Please make sure the peg is in the save position.")
        self.motion_manager = motion_manager
        self.motion_manager.zero2cubhome()
        # self.motion_manager.go_to_safe_height()
        # self.motion_manager.open_gripper()
        # self.motion_manager.go_to_garage_xytheta()

        self.peg_in_gripper = False
        if not self.motion_manager.gripper.is_active():
            self.motion_manager.reset_gripper()

        self.reset_marker_tracker()

        # Initialize tactile marker noise thresholds
        self._initialize_tactile_noise_thresholds()

        # Add force sensor subscriber after other subscribers
        # self.sub_force = rospy.Subscriber(
        #     "sri_force_sensor",
        #     ForceAndTorque,
        #     callback=callback_force_sensor,
        #     queue_size=10,
        # )

        # Initialize baseline force readings
        # self.baseline_force = None
        # self.get_force_baseline()

        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.marker_flow_size = 128
        self.default_observation = self.__get_sensor_default_observation__()
        self.observation_space = convert_observation_to_space(self.default_observation)

    def __get_sensor_default_observation__(self) -> dict:

        obs = {
            "gt_direction": np.zeros((1,), dtype=np.float32),
            "gt_offset": np.zeros((4,), dtype=np.float32),
            "relative_motion": np.zeros((4,), dtype=np.float32),
            "marker_flow": np.zeros((2, 2, self.marker_flow_size, 2), dtype=np.float32),
            "rgb_picture": np.zeros(
                (self.camera_size[0], self.camera_size[1], 3), dtype=np.uint8
            ),
            "depth_picture": np.zeros(
                (self.camera_size[0], self.camera_size[1]), dtype=np.float32
            ),
            "point_cloud": np.zeros(
                (self.camera_size[0], self.camera_size[1], 3), dtype=np.float32
            ),
            "object_point_cloud": np.zeros((2, self.max_points, 3), dtype=np.float32),
        }

        # for refrerence method

        if self.render_rgb:
            obs["rgb_images"] = np.stack(
                [
                    np.zeros((240, 320, 3), dtype=np.uint8),
                    np.zeros((240, 320, 3), dtype=np.uint8),
                ],
                axis=0,
            )

        return obs

    def set_start_episode(self, episode):
        self.current_episode = episode

    def _initialize_tactile_noise_thresholds(self):
        # Clear containers
        left_marker_flow_container.clear()
        right_marker_flow_container.clear()

        # time.sleep(5)

        # Wait until containers have enough data
        while (
            left_marker_flow_container.current_size < 30
            or right_marker_flow_container.current_size < 30
            or not left_marker_flow_container.check_shape()
            or not right_marker_flow_container.check_shape()
        ):
            time.sleep(0.1)

        # Calculate noise thresholds
        left_marker_seq = left_marker_flow_container.get(list(range(30)))
        right_marker_seq = right_marker_flow_container.get(list(range(30)))

        self.left_change_threshold = (
            np.mean(
                [
                    np.mean(
                        np.linalg.norm(
                            left_marker_seq[i][1] - left_marker_seq[i][0], axis=-1
                        )
                    )
                    for i in range(30)
                ]
            )
            * 1.25
        )
        self.right_change_threshold = (
            np.mean(
                [
                    np.mean(
                        np.linalg.norm(
                            right_marker_seq[i][1] - right_marker_seq[i][0], axis=-1
                        )
                    )
                    for i in range(30)
                ]
            )
            * 1.25
        )

        # Clamp thresholds to minimum values
        self.left_change_threshold = max(self.left_change_threshold, 0.15)
        self.right_change_threshold = max(self.right_change_threshold, 0.15)

        self.unique_logger.info(
            f"Tactile marker displacement thresholds: {self.left_change_threshold:.2f} (left), "
            f"{self.right_change_threshold:.2f} (right)"
        )

    def get_force_baseline(self):
        """Get average of first 30 force readings as baseline"""
        # global force_data
        global force_container
        # Wait until containers have enough data
        while (
            force_container.current_size
            < 30
            # or not force_container.check_shape()
        ):
            time.sleep(0.1)
        force_data = force_container.get(list(range(30)))

        self.unique_logger.info("Getting force baseline readings...")

        if force_data is None:
            raise ValueError("No force data received yet.")
        force_readings = []

        for i in range(30):
            fx = force_data[i].force_x
            fy = force_data[i].force_y
            fz = force_data[i].force_z
            mx = force_data[i].torque_x
            my = force_data[i].torque_y
            mz = force_data[i].torque_z
            force_readings.append([fx, fy, fz, mx, my, mz])
        self.baseline_force = np.mean(force_readings, axis=0)
        self.unique_logger.info(f"Baseline force: {self.baseline_force[:3]} N")
        self.unique_logger.info(f"Baseline torque: {self.baseline_force[3:]} Nm")

    """
    Motion related methods
    """

    def set_peg(self, peg_type):

        if not self.peg_in_gripper:
            self.peg = peg_type
            self.switch_peg_flag = True
            self.motion_manager.set_peg(self.peg)
            self.realsense.set_peg(self.peg)
            self.motion_manager.open_gripper()
            self.motion_manager.close_gripper()
            # self.motion_manager.go_to_safe_height()
            # self.motion_manager.go_to_garage_xytheta()

        else:
            if self.peg == peg_type:
                self.switch_peg_flag = False
                return
            self.switch_peg_flag = True
            self.motion_manager.move_peg_from_anywhere_to_garage(True)
            self.motion_manager.move_peg_from_garage_to_save()
            self.peg_in_gripper = False
            self.peg = peg_type
            self.motion_manager.set_peg(self.peg)
            self.realsense.set_peg(self.peg)
            self._update_peg_status()

        while not self.peg_in_gripper:
            # self.msg_box.show_message(
            #     "Please make sure the peg is in the save position."
            # )

            # self.motion_manager.move_peg_from_save_to_garage()
            # self.motion_manager.move_peg_from_garage_to_origin(True)            
            self.motion_manager.open_gripper()
            self.motion_manager.close_gripper()
            self._update_peg_status()

    def reset_marker_tracker(self):
        if not (
            self.left_sensor_init_marker_tracker_call()
            and self.right_sensor_init_marker_tracker_call()
        ):
            raise Exception("Failed to reset marker trackers.")
        time.sleep(0.5)

    def reset_peg_pose(self):
        self._update_peg_status()
        reset_tracker = False
        if self.current_episode % 1 == 0 or self.switch_peg_flag:
            reset_tracker = True
        # if self.switch_peg_flag:
        #     reset_tracker = True

        if self.peg_in_gripper:
            # self.motion_manager.move_peg_from_anywhere_to_garage(reset_tracker)
            pass

        else:
            self.msg_box.show_message("Please put the peg in the garage.")
            self.motion_manager.open_gripper()
            self.motion_manager.go_to_safe_height()
            self.motion_manager.go_to_garage_xytheta()

        if reset_tracker:
            self.first_color_image_mask = self.realsense.reset_tracker()
            if self.switch_peg_flag:
                self.switch_peg_flag = False

        # self.motion_manager.move_peg_from_garage_to_origin(reset_tracker)
        self.motion_manager.cubhome2origin()
        self._update_peg_status()

        return self.first_color_image_mask

    def _update_peg_status(self):
        global is_contact
        time.sleep(1)
        self.peg_in_gripper = is_contact

    @staticmethod
    def get_tactile_marker_flow():
        global left_marker_flow_container
        global right_marker_flow_container
        left_marker_flow = left_marker_flow_container.get(-1)
        right_marker_flow = right_marker_flow_container.get(-1)
        return left_marker_flow, right_marker_flow

    @staticmethod
    def get_tactile_image_flow():
        global right_marker_image_container
        global left_marker_image_container
        left_tactile_image_flow = left_marker_image_container.get(-1)
        right_tactile_image_flow = right_marker_image_container.get(-1)
        return left_tactile_image_flow, right_tactile_image_flow

    @staticmethod
    def get_marker_flow_difference(left_marker_flow, right_marker_flow):
        l_diff = np.mean(
            np.linalg.norm(left_marker_flow[1] - left_marker_flow[0], axis=-1)
        )
        r_diff = np.mean(
            np.linalg.norm(right_marker_flow[1] - right_marker_flow[0], axis=-1)
        )
        return l_diff, r_diff

    """
    RL env methods
    """

    def reset(self, offset_mm_deg):
        self.unique_logger.info(
            "*************************************************************"
        )
        self.unique_logger.info("reset once")
        self.current_episode += 1

        # reset peg pose
        first_color_image_mask = self.reset_peg_pose()

        # Reset episodic variables

        self.current_episode_elapsed_steps = 0
        self.error_too_large = False
        self.too_many_steps = False
        self.tactile_movement_too_large = False
        self.current_episode_max_tactile_diff = 0

        # # Reset force baseline
        # self.get_force_baseline()

        # Initialize offset
        offset_mm_deg = np.array(offset_mm_deg)
        offset_x, offset_y, offset_theta, offset_z = offset_mm_deg
        self.z_target_mm = offset_z + self.target_insertion_depth_mm
        y_start_mm = 45.25 / 2  # mm, based on assets/peg_insertion/hole_2.5mm.STL
        if self.peg == "hexagon":
            self.y_target_mm = 45.25
        else:
            self.y_target_mm = 0
        self.motion_manager.go_relative_offset(
            x=float(offset_x),
            y=float(offset_y),
            theta=float(offset_theta),
            z=float(offset_z),
            vel=15000,
        )
        self.reset_marker_tracker()
        # self._initialize_tactile_noise_thresholds()
        self.current_episode_initial_z_pos = self.motion_manager.get_position('Z')

        # Get initial observations
        self.current_episode_initial_left_observation = (
            adapt_marker_seq_to_unified_size(left_marker_flow_container.get(-1), 128)
        )
        self.current_episode_initial_right_observation = (
            adapt_marker_seq_to_unified_size(right_marker_flow_container.get(-1), 128)
        )
        if self.normalize_flow:
            self.current_episode_initial_left_observation = (
                self.current_episode_initial_left_observation / 160.0 - 1.0
            )
            self.current_episode_initial_right_observation = (
                self.current_episode_initial_right_observation / 160.0 - 1.0
            )
        self.sensor_grasp_center_init_mm_deg = offset_mm_deg.copy().astype(np.float32)
        self.sensor_grasp_center_current_mm_deg = offset_mm_deg.copy().astype(
            np.float32
        )

        offset_mm_deg[1] = offset_mm_deg[1] + y_start_mm - self.y_target_mm
        offset_mm_deg[3] = self.z_target_mm + 0.5
        self.current_offset_of_current_episode_mm_deg = offset_mm_deg.copy().astype(
            np.float32
        )

        self.error_evaluation_list = []
        info = self.get_info()
        info["action"] = np.array([0, 0, 0, 0])
        info["action_mm_deg"] = np.array([0, 0, 0, 0])
        self.error_evaluation_list = []
        self.error_evaluation_list.append(self.evaluate_error_v2(info))
        obs = self.get_obs(info)

        return obs, info, first_color_image_mask

    def step(self, action):
        self.current_episode_elapsed_steps += 1
        self.unique_logger.info(
            "#############################################################"
        )
        self.unique_logger.info(
            f"current_episode_elapsed_steps: {self.current_episode_elapsed_steps}"
        )
        self.unique_logger.info(f"action: {action}")
        action_mm_deg = np.array(action).flatten() * self.max_action_mm_deg
        self.unique_logger.info(f"action_mm_deg: {action_mm_deg}")
        self._real_step(action_mm_deg)
        time.sleep(0.1)

        info = self.get_info()
        info["action"] = action
        info["action_mm_deg"] = action_mm_deg
        self.unique_logger.info(f"info: {info}")
        obs = self.get_obs(info)
        # reward = self.get_reward(info=info)
        terminated = self.get_terminated(info)
        truncated = self.get_truncated(info)
        # done = self.get_done(info=info)
        self.unique_logger.info(
            "#############################################################"
        )
        return obs, terminated, truncated, info

    def _real_step(self, action_mm_deg):
        """
        :param action: numpy array; action[0]: delta_x, mm; action[1]: delta_y, mm; action[2]: delta_theta, degree.
        :return: observation, reward, done
        """

        # Convert action to world coordinate
        action_mm_deg = np.clip(
            action_mm_deg, -self.max_action_mm_deg, self.max_action_mm_deg
        )
        current_theta_rad = (
            self.current_offset_of_current_episode_mm_deg[2] * np.pi / 180
        )
        action_x_mm, action_y_mm = action_mm_deg[:2] @ [
            cos(current_theta_rad),
            -sin(current_theta_rad),
        ], action_mm_deg[:2] @ [
            sin(current_theta_rad),
            cos(current_theta_rad),
        ]
        action_theta_deg = action_mm_deg[2]
        action_z_mm = action_mm_deg[3]

        self.current_offset_of_current_episode_mm_deg[0] += action_x_mm
        self.current_offset_of_current_episode_mm_deg[1] += action_y_mm
        self.current_offset_of_current_episode_mm_deg[2] += action_theta_deg
        self.current_offset_of_current_episode_mm_deg[3] += action_z_mm
        self.sensor_grasp_center_current_mm_deg[0] += action_x_mm
        self.sensor_grasp_center_current_mm_deg[1] += action_y_mm
        self.sensor_grasp_center_current_mm_deg[2] += action_theta_deg
        self.sensor_grasp_center_current_mm_deg[3] += action_z_mm

        if (
            np.abs(self.current_offset_of_current_episode_mm_deg[0]) > 12 + 1e-5
            or np.abs(self.current_offset_of_current_episode_mm_deg[1]) > 30 + 1e-5
            or np.abs(self.current_offset_of_current_episode_mm_deg[2]) > 20 + 1e-5
            or np.abs(self.current_offset_of_current_episode_mm_deg[3]) > 15 + 1e-5
        ):
            self.error_too_large = True
        elif self.current_episode_elapsed_steps > self.max_steps:
            self.too_many_steps = True
        else:

            # Perform motion steps
            self.motion_manager.go_relative_offset(
                x=action_x_mm,
                y=action_y_mm,
                theta=action_theta_deg,
                z=action_z_mm,
                vel=15000,
            )
            time.sleep(0.1)
            # if is_overforced or self.check_force_limits():
            #     self.tactile_movement_too_large = True

    def get_info(self):

        info = {"steps": self.current_episode_elapsed_steps}
        info["is_success"] = False
        info["error_too_large"] = False
        info["too_many_steps"] = False
        info["tactile_movement_too_large"] = False
        info["message"] = "Normal step"

        current_z_pos = self.motion_manager.get_position('Z')
        insertion_depth_now = self.current_episode_initial_z_pos - current_z_pos

        if self.error_too_large:
            info["error_too_large"] = True
            info["message"] = "Error too large, insertion attempt failed"
        elif self.too_many_steps:
            info["too_many_steps"] = True
            info["message"] = "Too many steps, insertion attempt failed"
        elif self.tactile_movement_too_large:
            info["tactile_movement_too_large"] = True
            info["message"] = "Tactile movement too large, insertion attempt failed"
        else:
            self.obs_marker_flow = self.get_tactile_marker_flow()
            l_diff, r_diff = self.get_marker_flow_difference(*self.obs_marker_flow)

            if self.render_rgb:
                left_tactile_img, right_tactile_img = self.get_tactile_image_flow()
                self.tactile_img = np.stack(
                    [
                        left_tactile_img,
                        right_tactile_img,
                    ],
                    axis=0,
                )
            info["surface_diff"] = np.array([l_diff, r_diff])
            self.current_episode_max_tactile_diff = max(
                self.current_episode_max_tactile_diff, l_diff, r_diff
            )
            if (
                np.abs(self.current_offset_of_current_episode_mm_deg[0]) < 6.0
                and np.abs(self.current_offset_of_current_episode_mm_deg[1]) < 6.0
                and np.abs(self.current_offset_of_current_episode_mm_deg[2]) < 10.0
                and insertion_depth_now > self.z_target_mm
            ):
                relax_ratio = max(self.current_episode_max_tactile_diff / 1.5, 1)
                if (
                    l_diff < self.left_change_threshold * relax_ratio * 3
                    and r_diff < self.right_change_threshold * relax_ratio * 3
                ):
                    double_check_success = self._success_check(z_distance=3)
                    if double_check_success:
                        info["is_success"] = True
                        info["message"] = "Insertion succeed！"

        info["relative_motion_mm_deg"] = (
            self.sensor_grasp_center_current_mm_deg
            - self.sensor_grasp_center_init_mm_deg
        )

        info["gt_offset_mm_deg"] = self.current_offset_of_current_episode_mm_deg

        return info

    def _success_check(self, z_distance=1):
        current_z = self.motion_manager.get_position('Z')
        initial_zdiff = z_distance
        self.motion_manager.relative_move("z", -initial_zdiff, vel=1.5, wait=False)

        while self.motion_manager.is_moving():
            left_marker_flow, right_marker_flow = self.get_tactile_marker_flow()
            l_diff, r_diff = self.get_marker_flow_difference(
                left_marker_flow, right_marker_flow
            )
            if (
                l_diff >= self.left_change_threshold * 6
                or r_diff >= self.right_change_threshold * 6
            ):
                self.motion_manager.stop()
                self.motion_manager.absolute_move("z", current_z, vel=3, wait=True)
                return False

        self.motion_manager.absolute_move("z", current_z, vel=3, wait=True)
        return True

    def get_obs(self, info=None):

        obs_dict = dict()

        if self.peg == "hexagon":
            obs_dict["gt_direction"] = np.ones((1,), dtype=np.int8)
        else:
            obs_dict["gt_direction"] = -np.ones((1,), dtype=np.int8)

        obs_dict["gt_offset"] = info["gt_offset_mm_deg"].copy().astype(np.float32)

        obs_dict["relative_motion"] = np.array(info["relative_motion_mm_deg"]).astype(
            np.float32
        )

        l_flow, r_flow = self.obs_marker_flow
        l_flow = adapt_marker_seq_to_unified_size(l_flow, 128)
        r_flow = adapt_marker_seq_to_unified_size(r_flow, 128)
        if self.normalize_flow:
            l_flow = l_flow / 160.0 - 1.0
            r_flow = r_flow / 160.0 - 1.0

        obs_dict["marker_flow"] = np.stack(
            [
                l_flow,
                r_flow,
            ],
            axis=0,
        ).astype(np.float32)

        vision_results = self.realsense.get_vision_result(max_points=self.max_points)
        obs_dict["raw_color_image"] = vision_results["raw_color_image"]
        obs_dict["raw_depth_image"] = vision_results["raw_depth_image"]
        obs_dict["raw_point_cloud"] = vision_results["raw_point_cloud"]
        obs_dict["rgb_picture"] = vision_results["color_image"]
        obs_dict["depth_picture"] = vision_results["depth_image"]
        obs_dict["point_cloud"] = vision_results["point_cloud"]
        obs_dict["object_point_cloud"] = vision_results["object_point_cloud"]
        obs_dict["mask"] = vision_results["mask"]
        obs_dict["raw_color_image_masked"] = vision_results["raw_color_image_masked"]
        temp_path = f"{self.log_folder}/tracker_img/{self.current_episode}_{self.current_episode_elapsed_steps}.jpg"
        self.realsense.save_images_together(
            obs_dict["rgb_picture"], obs_dict["depth_picture"], temp_path
        )
        if self.render_rgb:
            obs_dict["rgb_images"] = self.tactile_img.copy()
        return obs_dict

    def evaluate_error_v2(self, info):
        return np.linalg.norm(info["gt_offset_mm_deg"], ord=2)

    # def get_reward(self, info):
    #     self.error_evaluation_list.append(self.evaluate_error_v2(info))

    #     reward_part_1 = self.error_evaluation_list[-2] - self.error_evaluation_list[-1]
    #     reward_part_2 = -self.step_penalty
    #     reward_part_3 = 0

    #     if info["error_too_large"] or info["tactile_movement_too_large"]:
    #         reward_part_3 += (
    #             -2
    #             * self.step_penalty
    #             * (self.max_steps - self.current_episode_elapsed_steps)
    #             + self.step_penalty
    #         )
    #     elif info["is_success"]:
    #         reward_part_3 += self.final_reward

    #     reward = reward_part_1 + reward_part_2 + reward_part_3
    #     self.unique_logger.info(
    #         f"reward: {reward}, reward_part_1: {reward_part_1}, reward_part_2: {reward_part_2}, reward_part_3: {reward_part_3}"
    #     )

    #     return reward

    def get_truncated(self, info) -> bool:
        return (
            info["too_many_steps"]
            or info["tactile_movement_too_large"]
            or info["error_too_large"]
        )

    def get_terminated(self, info) -> bool:
        return info["is_success"]

    def close(self):
        if self.peg_in_gripper:
            self.motion_manager.move_peg_from_anywhere_to_garage(True)
            self.motion_manager.move_peg_from_garage_to_save()
            self.peg_in_gripper = False
        self.reset_marker_tracker()
        self.motion_manager.close()
        self.sub_force.unregister()

    def check_force_limits(self):
        # global force_data
        global force_container
        force_data = force_container.get(-1)

        if force_data is None or self.baseline_force is None:
            return False

        current_force = np.array(
            [
                force_data.force_x,
                force_data.force_y,
                force_data.force_z,
                force_data.torque_x,
                force_data.torque_y,
                force_data.torque_z,
            ]
        )

        force_diff = current_force[:3] - self.baseline_force[:3]
        torque_diff = current_force[3:] - self.baseline_force[3:]

        total_force_diff = np.linalg.norm(force_diff)
        total_torque_diff = np.linalg.norm(torque_diff)

        if total_force_diff > FORCE_THRESHOLD or total_torque_diff > TORQUE_THRESHOLD:
            self.unique_logger.info(
                f"Force/torque limits exceeded (Force: {total_force_diff} N, Torque: {total_torque_diff} Nm)."
            )
            self.motion_manager.stop()
            self.motion_manager.wait_for_move_stop()
            return True
        return False

    # def show_message(self, message: str):
    #     # 创建一个隐藏的根窗口
    #     root = tk.Tk()
    #     root.withdraw()  # 隐藏根窗口
    #
    #     # 显示消息框
    #     messagebox.showinfo("Warning!", message)


def normal_test(env: PegInsertionRealEnvV2, test_log):
    start_time = time.time()
    test_result = []
    offset_list = []
    offset_list.append([5, 5, -10, 9])
    # offset_list.append([5, 5, -10, 9])
    # offset_list.append([5, -5, 10, 9])
    action_list = []
    action_list.append(
        [[-0.5, -1, 1, -0.2]] * 10
        + [[0.05, -1, 0, 0]] * 18
        + [[0.0, 0, 0, -1]] * 7
        + [[0.0, 0, 0, -1]] * 6
    )
    # action_list.append(
    #     [[-0.5, -1, 1, -0.2]] * 10
    #     + [[0.05, -1, 0, 0]] * 18
    #     + [[0.0, 0, 0, -1]] * 7
    #     + [[0.0, 0, 0, -1]] * 6
    # )
    # action_list.append(
    #     [[-0.5, 1, -1, -0.2]] * 10
    #     + [[0.05, 1, 0, 0]] * 18
    #     + [[0.0, 0, 0, -1]] * 7
    #     + [[0.0, 0, 0, -1]] * 6
    # )
    for ii in range(len(offset_list)):
        if offset_list[ii][1] == -5:
            env.set_peg("hexagon")
        else:
            env.set_peg("cuboid")
        offset = offset_list[ii]
        actions = action_list[ii]
        test_log.opt(colors=True).info(
            f"<blue>#### Test No. {len(test_result) + 1} ####</blue>"
        )
        obs, info, first_color_image_mask = env.reset(offset)
        save_data_helper.save_one_time(
            env.current_episode,
            env.current_episode_elapsed_steps,
            obs,
            info,
            first_color_image_mask,
            is_mask=True,
        )
        for k, v in obs.items():
            test_log.info(f"{k} : {v.shape}")

        test_log.info(f"info : {info}\n")
        done, ep_len = False, 0
        while not done:
            ep_len += 1
            action = actions[ep_len - 1]
            test_log.info(f"Step {ep_len} Action: {action}")
            obs, terminated, truncated, info = env.step(action)
            save_data_helper.save_one_time(
                env.current_episode,
                env.current_episode_elapsed_steps,
                obs,
                info,
                obs["raw_color_image_masked"],
            )
            done = terminated or truncated
            if "gt_offset" in obs.keys():
                test_log.info(f"Offset: {obs['gt_offset']}")
            test_log.info(f"info: {info}")
        if info["is_success"]:
            test_result.append([True, ep_len])
            test_log.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
        else:
            test_result.append([False, ep_len])
            test_log.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    end_time = time.time()
    test_log.info(f"cost time: {end_time - start_time}")
    env.close()


def for_PC_registration(env: PegInsertionRealEnvV2, test_log):
    start_time = time.time()
    test_result = []
    offset_list = []
    offset_list.append([0, 0, -10, 1])
    offset_list.append([5, 27.625, 5, 3]) 
    offset_list.append([5, -27.625, 0, 5])
    offset_list.append([-5, 27.625, 10, 7])
    offset_list.append([-5, -27.625, -5, 9])

    for ii in range(len(offset_list)):
        env.set_peg("cuboid")
        offset = offset_list[ii]
        test_log.opt(colors=True).info(
            f"<blue>#### Test No. {len(test_result) + 1} ####</blue>"
        )
        obs, info, first_color_image_mask = env.reset(offset)
        save_data_helper.save_one_time(
            env.current_episode,
            env.current_episode_elapsed_steps,
            obs,
            info,
            first_color_image_mask,
            is_mask=True,
        )
        for k, v in obs.items():
            test_log.info(f"{k} : {v.shape}")
        test_log.info(f"info : {info}\n")

    end_time = time.time()
    test_log.info(f"cost time: {end_time - start_time}")
    env.close()


if __name__ == "__main__":

    motion_manager = MotionManagerStageV2(
        "/dev/ttyUSB0", "/dev/ttyUSB1", "cuboid", 50
    )
    use_render_rgb = True
    log_time = get_time()
    log_folder = Path(os.path.join(track_path, f"Memo/{log_time}"))
    log_dir = Path(os.path.join(log_folder, "main.log"))
    log.remove()
    log.add(log_dir, filter=lambda record: record["extra"]["name"] == "main")
    log.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
        level="INFO",
        filter=lambda record: record["extra"]["name"] == "main",
    )
    test_log = log.bind(name="main")
    save_data_helper = RlDataSaveLoadHelper(log_folder)

    vision_params = {
        "vision_type": [
            "rgb",
            "point_cloud",
            "depth",
        ],
        "max_points": 128,  # sample the points from the point cloud
    }

    env = PegInsertionRealEnvV2(
        motion_manager=motion_manager,
        # step_penalty=1,
        # final_reward=10,
        peg="cuboid",
        max_action_mm_deg=np.array([1.0, 1.0, 1.0, 1.0]),
        max_steps=50,
        insertion_depth_mm=2,
        normalize=False,
        grasp_height_offset=0,
        render_rgb=use_render_rgb,
        vision_params=vision_params,
        log_path=log_folder,
        logger=log,
    )
    np.set_printoptions(precision=4)
    # episode_count = 0

    # # normal test
    # normal_test(env,test_log)
    # for PC registration
    for_PC_registration(env, test_log)
