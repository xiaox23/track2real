import math
import os, sys
import cv2
import numpy as np
import torch
from gymnasium.vector.utils import spaces
from matplotlib import pyplot as plt

from utils.img_process_utils import convert_to_binary_torch
from utils.img_process_utils import get_valid_marker_sequence
from utils.np_utils import estimate_rigid_transform

import tkinter as tk
from tkinter import messagebox, Toplevel, font
script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)


def get_rotation_and_force(img_seq):
    marker_seq = get_valid_marker_sequence(img_seq)
    m_mask = np.logical_and.reduce([marker_seq[0, :, 0] > 95,
                                    marker_seq[0, :, 0] < 230,
                                    marker_seq[0, :, 1] > 160,
                                    marker_seq[0, :, 1] < 300])

    marker_seq = marker_seq[:, m_mask]
    marker_motion = np.mean(marker_seq[1] - marker_seq[0], axis=0)
    # angles = get_marker_flow_angle(marker_seq)
    # mean_angle = np.mean(angles)
    # lengths = np.sqrt(np.sum((marker_seq[-1, ...] - marker_seq[0, ...]) ** 2, axis=1))
    # mean_length = np.mean(lengths)
    init_pts = np.concatenate((marker_seq[0], np.zeros((marker_seq[0].shape[0], 1))), axis=1)
    curr_pts = np.concatenate((marker_seq[1], np.zeros((marker_seq[1].shape[0], 1))), axis=1)

    R, t = estimate_rigid_transform(init_pts, curr_pts)
    angle = math.atan2(R[1, 0], R[0, 0]) * 180 / math.pi
    f_u = marker_motion[0]
    f_v = marker_motion[1]

    return angle, f_u, f_v


def get_marker_flow_rotation_and_force(marker_seq):
    m_mask = np.logical_and.reduce(
        [
            marker_seq[0, :, 0] > 95,
            marker_seq[0, :, 0] < 230,
            marker_seq[0, :, 1] > 160,
            marker_seq[0, :, 1] < 300,
        ]
    )

    marker_seq = marker_seq[:, m_mask]
    marker_motion = np.mean(marker_seq[1] - marker_seq[0], axis=0)
    init_pts = np.concatenate((marker_seq[0], np.zeros((marker_seq[0].shape[0], 1))), axis=1)
    curr_pts = np.concatenate((marker_seq[1], np.zeros((marker_seq[1].shape[0], 1))), axis=1)

    R, t = estimate_rigid_transform(init_pts, curr_pts)
    angle = math.atan2(R[1, 0], R[0, 0]) * 180 / math.pi
    f_u = marker_motion[0]
    f_v = marker_motion[1]

    return angle, f_u, f_v


def get_total_force(f_l, f_r):
    # print(f_l, f_r)
    angle_l, f_u_l, f_v_l = f_l
    angle_r, f_u_r, f_v_r = f_r

    f_x = - f_u_l + f_u_r
    f_z = - f_v_l - f_v_r
    M_z = f_u_l + f_u_r
    M_x = -f_v_l + f_v_r
    M_y = angle_l - angle_r

    return f_x, f_z, M_x, M_y, M_z


def manual_policy(f_l, f_r):
    angle_l, f_u_l, f_v_l = f_l
    angle_r, f_u_r, f_v_r = f_r
    f_x, f_z, M_x, M_y, M_z = get_total_force(f_l, f_r)
    print("fx:{:.2f}, fz:{:.2f}, Mx:{:.2f}, My:{:.2f}, Mz:{:.2f}".format(f_x, f_z, M_x, M_y, M_z))
    if -f_v_l >= -0.75 and -f_v_r >= -0.75:
        next_action = (0.25 * f_x + 0.25 * M_y, -0.4 * M_x, 1.5 * M_z)
    else:
        print("abnormal vertical force", -f_v_l, -f_v_r)
        # if -f_v_l < -4:
        #     next_action = (0, -0.5, 0)
        # elif -f_v_r < -4:
        #     next_action = (0, 0.5, 0)
        next_action = (0.25 * f_x - 0.25 * M_y, 0.4 * M_x, 1.5 * M_z)
    # if np.sum(np.abs(np.array(next_action))) < 0.1:
    #     next_action = (0, 0, 0)
    return next_action


def evaluate_error(offset):
    offset_squared = offset ** 2
    # radius offset: max 7^2
    # angular offset: max (10*3.14159/180)^2 = 0.1745^2

    error = math.sqrt(offset_squared[0] + offset_squared[1] + offset_squared[2])
    return error


def visualize_marker_flow(o_l, o_r, f_l, f_r, action):
    def overlap_marker_img(marker_img_1, marker_img_2):
        marker_position_1 = (marker_img_1[:, :, 0] < 200)
        marker_position_2 = (marker_img_2[:, :, 0] < 200)

        colored_img = cv2.cvtColor((np.ones_like(marker_img_1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        colored_img[marker_position_1, 0] = 255  # blue
        colored_img[marker_position_1, 1] = marker_img_1[marker_position_1, 0]
        colored_img[marker_position_1, 2] = marker_img_1[marker_position_1, 0]
        colored_img[marker_position_2, 0] = marker_img_2[marker_position_2, 0]
        colored_img[marker_position_2, 1] = marker_img_2[marker_position_2, 0]
        colored_img[marker_position_2, 2] = 255  # red
        return colored_img

    visualize_l = overlap_marker_img(o_l[0], o_l[1])
    visualize_r = overlap_marker_img(o_r[0], o_r[1])
    visualize_lr = np.concatenate([visualize_l, visualize_r], axis=1)

    visualize_string_l = (np.ones((40, 320, 3)) * 255).astype(np.uint8)
    visualize_string_r = visualize_string_l.copy()
    visualize_string_l = cv2.putText(visualize_string_l, "L: {:+2.2f}, {:+2.2f}, {:+2.2f}".format(*f_l), (5, 35),
                                     fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.7, color=(0, 0, 0),
                                     lineType=cv2.LINE_AA)
    visualize_string_r = cv2.putText(visualize_string_r, "R: {:+2.2f}, {:+2.2f}, {:+2.2f}".format(*f_r), (5, 35),
                                     fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.7, color=(0, 0, 0),
                                     lineType=cv2.LINE_AA)

    visualize_string_lr = np.concatenate([visualize_string_l, visualize_string_r], axis=1)

    total_force = get_total_force(f_l, f_r)
    visualize_total_force = (np.ones((40, 640, 3)) * 255).astype(np.uint8)
    visualize_total_force = cv2.putText(visualize_total_force,
                                        "Total: {:+2.2f}, {:+2.2f};".format(*total_force[:2]), (35, 35),
                                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 0),
                                        lineType=cv2.LINE_AA)
    visualize_total_force = cv2.putText(visualize_total_force,
                                        "{:+2.2f}, {:+2.2f}, {:+2.2f}".format(*total_force[2:]), (325, 35),
                                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 0),
                                        lineType=cv2.LINE_AA)

    visualize_action = (np.ones((40, 640, 3)) * 255).astype(np.uint8)
    visualize_action = cv2.putText(visualize_action,
                                   "action: {:+2.2f}, {:+2.2f}, {:+2.2f}".format(*action), (100, 35),
                                   fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 0),
                                   lineType=cv2.LINE_AA)

    visualize_all = np.concatenate([visualize_lr, visualize_string_lr, visualize_total_force, visualize_action],
                                   axis=0)
    cv2.imshow("marker flow", visualize_all)
    cv2.waitKey(1)


def process_observation(orginal_obs, use_visual=False):
    if orginal_obs[1] is not None and orginal_obs[2] is not None:
        if not use_visual:
            observation = (orginal_obs[0],
                           torch.tensor((*get_rotation_and_force(orginal_obs[1]), *get_rotation_and_force(orginal_obs[2])), dtype=torch.float32))
        else:
            l_img = torch.squeeze(convert_to_binary_torch(orginal_obs[1]), dim=1)
            r_img = torch.squeeze(convert_to_binary_torch(orginal_obs[2]), dim=1)
            observation = (orginal_obs[0], torch.stack([l_img, r_img]))
    else:
        if not use_visual:
            observation = (orginal_obs[0],
                           torch.tensor((0, 0, 0, 0, 0, 0), dtype=torch.float32))
        else:
            l_img = torch.zeros((2, 256, 256))
            r_img = torch.zeros((2, 256, 256))
            observation = (orginal_obs[0], torch.stack([l_img, r_img]))
    return observation


def process_observations(original_observations, use_visual=False):
    ret = []
    for obs in original_observations:
        processed_observation = process_observation(obs, use_visual)
        # cur_gt = processed_observation[0]
        # cur_measure = processed_observation[1]
        # gt.append(cur_gt)
        ret.append(processed_observation)
    return ret


def visualize_marker_flow_torch_img(o_l, o_r, window_name="marker flow", timeout=1):
    def overlap_marker_img(marker_img_1, marker_img_2):
        marker_position_1 = (marker_img_1[:, :] == True)
        marker_position_2 = (marker_img_2[:, :] == True)

        colored_img = cv2.cvtColor((np.ones(marker_img_1.shape) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        colored_img[marker_position_1, 0] = 255  # blue
        colored_img[marker_position_1, 1] = 0
        colored_img[marker_position_1, 2] = 0
        colored_img[marker_position_2, 0] = 0
        colored_img[marker_position_2, 1] = 0
        colored_img[marker_position_2, 2] = 255  # red
        return colored_img

    visualize_l = overlap_marker_img(o_l[0], o_l[1])
    visualize_r = overlap_marker_img(o_r[0], o_r[1])
    visualize_lr = np.concatenate([visualize_l, visualize_r], axis=1)

    cv2.imshow(window_name, visualize_lr)
    cv2.waitKey(timeout)


def get_dtype_bounds(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, bool):
        return 0, 1
    else:
        raise TypeError(dtype)


def convert_observation_to_space(observation, prefix=""):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from `gym.envs.mujoco_env`
    """
    if isinstance(observation, (dict)):
        space = spaces.Dict({k: convert_observation_to_space(v, prefix + "/" + k) for k, v in observation.items()})
    elif isinstance(observation, np.ndarray):
        shape = observation.shape
        dtype = observation.dtype
        dtype_min, dtype_max = get_dtype_bounds(dtype)
        low = np.full(shape, dtype_min)
        high = np.full(shape, dtype_max)
        space = spaces.Box(low, high, dtype=dtype)
    elif isinstance(observation, float):
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    elif isinstance(observation, np.float32):
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

def visualize_marker_point_flow(o):
    lr_marker_flow = o["marker_flow"]
    l_marker_flow, r_marker_flow = lr_marker_flow[0], lr_marker_flow[1]
    plt.figure(1, (20, 9))
    ax = plt.subplot(1, 2, 1)
    ax.scatter(l_marker_flow[0, :, 0], l_marker_flow[0, :, 1], c="blue")
    ax.scatter(l_marker_flow[1, :, 0], l_marker_flow[1, :, 1], c="red")
    ax.invert_yaxis()
    ax = plt.subplot(1, 2, 2)
    ax.scatter(r_marker_flow[0, :, 0], r_marker_flow[0, :, 1], c="blue")
    ax.scatter(r_marker_flow[1, :, 0], r_marker_flow[1, :, 1], c="red")
    ax.invert_yaxis()
    plt.show()

class CustomMessageBox:
    def __init__(self, master):
        self.master = master
        self.master.withdraw()  # 隐藏主窗口

    def show_message(self, message: str):
        try:
            # 创建一个自定义的弹窗窗口
            dialog = tk.Toplevel(self.master)
            dialog.title("Warning!")  # 设置窗口标题
            dialog.geometry("400x200")  # 设置窗口大小

            # 设置字体大小
            big_font = font.Font(family="Monospaced", size=20)

            # 创建一个标签来显示消息
            label = tk.Label(dialog, text=message, font=big_font,wraplength=300)
            label.pack(pady=20)  # 添加内边距

            # 创建一个关闭按钮
            def close_dialog():
                dialog.destroy()
                dialog.quit()  # 结束消息循环

            ok_button = tk.Button(
                dialog, text="OK", command=close_dialog, font=big_font, padx=20, pady=10,
            )
            ok_button.pack(pady=10)

            # 绑定关闭事件，确保窗口可以正确关闭
            dialog.protocol("WM_DELETE_WINDOW", close_dialog)

            # 将窗口显示在屏幕中央
            self.center_window(dialog)

            # 进入消息循环
            dialog.mainloop()
        except Exception as e:
            print(f"An error occurred: {e}")

    def center_window(self, window):
        """将窗口居中显示在屏幕上"""
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")

if __name__ == "__main__":
    root = tk.Tk()
    msg_box = CustomMessageBox(root)
    msg_box.show_message("test")
    print("yes")