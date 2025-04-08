#!/bin/bash
import copy
import os
import sys
import keyboard
import time
from pynput import keyboard

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)
import numpy as np
from path import Path
from utils.common import get_time
from loguru import logger as log
from utils.RL_data_save_load_helper import RlDataSaveLoadHelper
from envs.motion_manager_stage_v2 import MotionManagerStageV2

from envs.peg_insertion_v2_real_motion import PegInsertionRealEnvV2_vis
import ruamel.yaml as yaml


# SAVE_DIR = 'real_tac_data'
# os.makedirs(SAVE_DIR, exist_ok=True)
# CNT = len(os.listdir(SAVE_DIR))


PRE_DELETE_ENV_KEY = [
    "env_name",
    "gui",
    "marker_interval_range",
    "marker_lose_tracking_probability",
    "marker_pos_shift_range",
    "marker_random_noise",
    "marker_rotation_range",
    "marker_translation_range",
    "params",
    "peg_dist_z_diff_mm",
        "peg_dist_z_mm",
    "peg_hole_path_file",
    "peg_theta_max_offset_deg",
    "peg_x_max_offset_mm",
    "peg_y_max_offset_mm",
    "step_penalty",
    "final_reward",
]

def on_press(key):
    global running, exit_program, CNT  # 声明全局变量
    try:
        if key.char == 'a':  # 检测是否按下 'a' 键
            CNT += 1 
            if not running:
                running = True
                print("循环开始...")

        elif key.char == 'b':  # 检测是否按下 'b' 键
            if running:
                running = False
                print("循环结束。")

        elif key.char == 'q':  # 检测是否按下 'q' 键
            print("程序退出。")
            exit_program = True  # 设置退出标志
            return False  # 停止监听器
    except AttributeError:
        pass  # 忽略特殊按键


if __name__ == "__main__":

    exp_start_time = get_time()
    exp_name = f"peg_insertion_v2_{exp_start_time}"
    log_folder = Path(os.path.join(track_path, f"eval_log/{exp_name}"))
    log_dir = Path(os.path.join(log_folder, "main.log"))
    log.remove()
    log.add(
        log_dir,
        filter=lambda record: record["extra"]["name"] == "main",
    )

    log.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
        level="INFO",
        filter=lambda record: record["extra"]["name"] == "main",
    )
    eval_log = log.bind(name="main")
    save_data_helper = RlDataSaveLoadHelper(log_folder)
    motion_manager = MotionManagerStageV2(
        "/dev/ttyUSB0", "/dev/ttyUSB1", "hexagon", 50
    )

    with open("/home/tars/workspace/xx/tactile/STEIIA-PENTAC-3rd-commit/stg2_2nd/Track_2/configs/parameters/peg_insertion_v2_points.yaml", "r") as f:
        cfg = yaml.YAML(typ="safe", pure=True).load(f)

    if "max_action" in cfg["env"].keys():
        cfg["env"]["max_action"] = np.array(cfg["env"]["max_action"])



    specified_env_args = copy.deepcopy(cfg["env"])
    for delete_key in PRE_DELETE_ENV_KEY:
        if delete_key in specified_env_args:
            del specified_env_args[delete_key]

    eval_log.info(specified_env_args)

    specified_env_args.update(
        {
            "motion_manager": motion_manager,
            "peg": "hexagon",
            "log_path": log_folder,
            "logger": log,
            "grasp_height_offset": 0
        }
    )

    env = PegInsertionRealEnvV2_vis(**specified_env_args)

    # 定义全局变量
    running = False  # 控制循环状态
    exit_program = False  # 控制程序退出状态


    # 创建键盘监听器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # 启动监听器

    frame_num = 0
    print("wait for keyboard input:")
    while True:
        if exit_program:  # 检查退出标志
            print("主循环退出")
            listener.stop()  # 显式停止监听器
            break

        if running:
            print("循环中...")
            # time.sleep(0.5)  # 模拟循环任务
            time.sleep(0.5)  # 模拟循环任务
            # obs = env.get_obs()
            # tac_mf = obs['marker_flow']
            # save_path = os.path.join(SAVE_DIR, str("%04d"%CNT))
            # os.makedirs(save_path, exist_ok=True)
            # save_name = os.path.join(save_path, str("%05d"%frame_num) + '.npy')
            # np.save(save_name, tac_mf)
            # frame_num += 1
            

        else:
            time.sleep(0.1)  # 减少 CPU 占用