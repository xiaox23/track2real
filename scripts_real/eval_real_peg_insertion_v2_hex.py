#!/bin/bash
import copy
import hashlib
import os
import sys

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
import torch
from utils.RL_data_save_load_helper import RlDataSaveLoadHelper
from envs.motion_manager_stage_v2 import MotionManagerStageV2

# import git


from envs.peg_insertion_v2_real_motion import PegInsertionRealEnvV2
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.save_util import load_from_zip_file
import ruamel.yaml as yaml

PRE_OBS_KEY = [
    "gt_direction",
    "gt_offset",
    "relative_motion",
    "marker_flow",
    "rgb_picture",
    "depth_picture",
    "point_cloud",
    "object_point_cloud",
    "rgb_images",
]
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

PRE_OFFSET_HEX = [
    [-1.3, 2.1, 9.1, 7.9],
    [1.1, -2.7, 1.9, 3.6],
    [-4.8, -4.0, 5.8, 6.3],
    [1.2, 0.2, -3.2, 6.8],
    [4.8, 3.6, 4.9, 8.4],
    [-4.2, -2.7, -3.3, 5.8],
    [-3.0, 3.8, -3.8, 4.2],
    [1.9, -2.4, 1.7, 3.1],
    [-1.2, 0.1, 8.8, 8.6],
    [-4.2, 4.7, -4.8, 5.7],
    [3.9, -0.2, 0.4, 8.8],
    [-2.3, 4.7, 3.3, 6.8],
    [4.5, -4.0, 9.0, 5.3],
    [3.8, -0.2, 8.8, 6.0],
    [-3.8, 0.6, -3.4, 5.9],
    # [-1.3, 2.1, 9.1, 7.9],
    # [1.1, -2.7, 1.9, 3.6],
    # [-4.8, -4.0, 5.8, 6.3],
    # [1.2, 0.2, -3.2, 6.8],
    # [4.8, 3.6, 4.9, 8.4],
    # [-4.2, -2.7, -3.3, 5.8],
    # [-3.0, 3.8, -3.8, 4.2],
    # [1.9, -2.4, 1.7, 3.1],
    # [-1.2, 0.1, 8.8, 8.6],
    # [-4.2, 4.7, -4.8, 5.7],
    # [3.9, -0.2, 0.4, 8.8],
    # [-2.3, 4.7, 3.3, 6.8],
    # [4.5, -4.0, 9.0, 5.3],
    # [3.8, -0.2, 8.8, 6.0],
    # [-3.8, 0.6, -3.4, 5.9],
]
PRE_OFFSET_CUB = []
ALL_OFFSET =  PRE_OFFSET_CUB+ PRE_OFFSET_HEX
PRE_OFFSET_HEX_LENGTH = len(PRE_OFFSET_HEX)
PRE_OFFSET_CUB_LENGTH = len(PRE_OFFSET_CUB)

# PRE_OFFSET_HEX_TEST = [
#     [-1.3, 2.1, 9.1, 7.9],
#     [1.1, -2.7, 1.9, 3.6],
# ]
# PRE_OFFSET_CUB_TEST = [
#     [2.3, -4.1, -3.2, 8.8],
#     [-3.0, 0.5, -6.3, 7.9],
#     # [3.2, 0.5, 4.2, 3.4],
# ]
# ALL_OFFSET = PRE_OFFSET_CUB_TEST + PRE_OFFSET_HEX_TEST
# PRE_OFFSET_HEX_LENGTH = len(PRE_OFFSET_HEX_TEST)
# PRE_OFFSET_CUB_LENGTH = len(PRE_OFFSET_CUB_TEST)

def evaluate_policy(model, key: str, cfg_path: str, start_episode) -> None:
    """评估策略模型的性能"""
    # logging file
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
        "/dev/translation_stage", "/dev/rotation_stage", "/dev/hande", "hexagon", 50
    )

    eval_log.info(f"#KEY: {key}")
    with open(cfg_path, "r") as f:
        cfg = yaml.YAML(typ="safe", pure=True).load(f)

    # get simulation and environment parameters

   

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
            # "env_type": "real",
        }
    )

    env = PegInsertionRealEnvV2(**specified_env_args)
    env.set_start_episode(start_episode)
    np.set_printoptions(precision=4)
    # set_random_seed(0)

    # 测试逻辑
    test_num_hex = PRE_OFFSET_HEX_LENGTH
    test_num_cub = PRE_OFFSET_CUB_LENGTH
    test_result = []
    eval_log.info(f"Testing all {test_num_hex + test_num_cub} cases")
    eval_log.info(f"Including {test_num_hex} hex cases")
    eval_log.info(f"Including {test_num_cub} cub cases")

    # 测试循环

    for kk in range(start_episode, test_num_hex + test_num_cub):
        if kk < test_num_cub:
            env.set_peg("cuboid")
        else:
            env.set_peg("hexagon")
        offset = ALL_OFFSET[kk]
        eval_log.opt(colors=True).info(
            f"<blue>#### Test No. {len(test_result) + 1} ####</blue>"
        )
        eval_log.info(f"Given offset: {offset}")
        obs, info, first_color_image_mask = env.reset(offset)
        save_data_helper.save_one_time(
            env.current_episode,
            env.current_episode_elapsed_steps,
            obs,
            info,
            first_color_image_mask,
            is_mask=True,
        )
        done, ep_len = False, 0
        while not done:
            # Take deterministic actions at test time (noise_scale=0)
            ep_len += 1
            for key in list(obs.keys()):
                if key not in PRE_OBS_KEY:
                    del obs[key]
            for obs_k, obs_v in obs.items():
                obs[obs_k] = torch.from_numpy(obs_v)
            action = model(obs)
            action = action.cpu().detach().numpy().flatten()
            eval_log.info(f"Step {ep_len} Action: {action}")
            obs, terminated, truncated, info = env.step(action)
            save_data_helper.save_one_time(
                env.current_episode,
                env.current_episode_elapsed_steps,
                obs,
                info,
                obs["raw_color_image_masked"],
            )
            done = terminated or truncated
            eval_log.info(f"info: {info}")
        if info["is_success"]:
            test_result.append([True, ep_len])
            eval_log.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
        else:
            test_result.append([False, ep_len])
            eval_log.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    
    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in test_result])) / (
        (test_num_hex + test_num_cub)
    )
    if success_rate > 0:
        avg_steps = (
            np.mean(np.array([int(v[1]) if v[0] else 0 for v in test_result]))
            / success_rate
        )
        eval_log.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        eval_log.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        eval_log.info(f"#SUCCESS_RATE: 0")
        eval_log.info(f"#AVG_STEP: NA")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--team_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--policy_file_path", type=str, required=True)
    parser.add_argument("--cfg_file_path", type=str, required=True)
    parser.add_argument("--start_episode", type=int, required=False, default=0)
    args = parser.parse_args()

    data, params, _ = load_from_zip_file(args.policy_file_path)

    # 假设 policies 模块已经正确导入
    from solutions import policies

    model_class = getattr(policies, args.model_name)
    model = model_class(
        observation_space=data["observation_space"],
        action_space=data["action_space"],
        lr_schedule=data["lr_schedule"],
        **data["policy_kwargs"],
    )
    model.load_state_dict(params["policy"])
    evaluate_policy(model, args.team_name, args.cfg_file_path, args.start_episode)
