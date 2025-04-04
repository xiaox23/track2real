import pickle
import os
import cv2
import json
import numpy as np

SAVE_DATA_CATAGEORY = [
    "gt_direction",
    "gt_offset",
    "relative_motion",
    "marker_flow",
    "raw_color_image",
    "raw_depth_image",
    "raw_point_cloud",
    "mask",
    "rgb_images",
]


class RlDataSaveLoadHelper:
    def __init__(self, main_folder_path):
        self.main_folder_path = main_folder_path
        if not os.path.exists(main_folder_path):
            os.makedirs(main_folder_path)

    def set_main_folder_path(self, main_folder_path):
        self.main_folder_path = main_folder_path

    def save_obs_and_info(self, episode, step, obs_data, info_data):
        obs_info_data_folder_path = os.path.join(
            self.main_folder_path, f"Episode_{episode}/obs_and_info_step_{step}"
        )
        if not os.path.exists(obs_info_data_folder_path):
            os.makedirs(obs_info_data_folder_path)
        info_json_path = os.path.join(obs_info_data_folder_path, "info.json")

        for key, value in info_data.items():
            if isinstance(value, np.ndarray):
                info_data[key] = value.tolist()

        with open(info_json_path, "w") as file:
            json.dump(info_data, file, indent=4)

        for key, array in obs_data.items():
            # 生成文件名，以键加上.npy 后缀
            if key not in SAVE_DATA_CATAGEORY:
                continue
            filename = os.path.join(
                obs_info_data_folder_path, f"{key}_{episode}_{step}.npy"
            )
            # 保存 numpy 数组到.npy 文件
            np.save(filename, array)

    def save_tracker_img(self, image_tracker, episode, step, is_mask=False):
        image_folder_path = os.path.join(
            self.main_folder_path, "Episode_" + str(episode)
        )
        if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)
        if is_mask:
            image_path = os.path.join(image_folder_path, "mask.png")
        else:
            image_path = os.path.join(
                image_folder_path, f"tracker_{episode}_{step}.png"
            )
        cv2.imwrite(image_path, image_tracker)

    def save_one_time(self, episode, step, obs_data, info_data, image_tracker, is_mask=False):
        self.save_obs_and_info(episode, step, obs_data, info_data)
        self.save_tracker_img(image_tracker, episode, step, is_mask)

    # @staticmethod
    # def save_data(data, file_path):
    #     with open(file_path, "wb") as f:
    #         pickle.dump(data, f)

    # @staticmethod
    # def load_data(file_path):
    #     with open(file_path, "rb") as f:
    #         return pickle.load(f)
