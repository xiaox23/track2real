#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# 定义数据目录和可视化函数
SAVE_DIR = 'real_tac_data'  # 数据存储的目录

def visualize_marker_point_flow(data, i, name, save_dir="marker_flow_images"):
    """
    可视化 marker_flow 数据并保存为图片。
    :param data: NumPy 数组，包含触觉数据。
    :param i: 当前文件索引，用于生成保存的图片文件名。
    :param name: 当前文件所属的目录名，用于生成保存的图片。
    :param save_dir: 图片保存的目录。
    """
    # 创建一个保存图片的目录（如果不存在则创建）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 假设输入数据的 shape 为 (2, 2, 128, 2)，第一个维度表示左右手
    if data.shape != (2, 2, 128, 2):
        print(f"Unexpected data shape: {data.shape}. Skipping visualization.")
        return

    l_marker_flow, r_marker_flow = data[0], data[1]  # 左手和右手数据
    plt.figure(figsize=(20, 9))

    # 可视化左手的 marker_flow
    ax = plt.subplot(1, 2, 1)
    ax.scatter(l_marker_flow[0, :, 0], l_marker_flow[0, :, 1], c="blue", label="Left Flow 1")
    ax.scatter(l_marker_flow[1, :, 0], l_marker_flow[1, :, 1], c="red", label="Left Flow 2")
    plt.xlim(15, 315)
    plt.ylim(15, 235)
    ax.invert_yaxis()
    ax.set_title("Left Hand Marker Flow")
    ax.legend()

    # 可视化右手的 marker_flow
    ax = plt.subplot(1, 2, 2)
    ax.scatter(r_marker_flow[0, :, 0], r_marker_flow[0, :, 1], c="blue", label="Right Flow 1")
    ax.scatter(r_marker_flow[1, :, 0], r_marker_flow[1, :, 1], c="red", label="Right Flow 2")
    plt.xlim(15, 315)
    plt.ylim(15, 235)
    ax.invert_yaxis()
    ax.set_title("Right Hand Marker Flow")
    ax.legend()

    # 保存图片到指定目录
    filename = os.path.join(save_dir, f"{name}_marker_flow_{i}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")

def load_and_plot_data(save_dir):
    """
    加载并可视化保存的数据。
    :param save_dir: 数据存储的目录。
    """
    if not os.path.exists(save_dir):
        print(f"Error: Directory '{save_dir}' does not exist.")
        return

    # 遍历保存的数据目录
    for subdir in sorted(os.listdir(save_dir)):
        subdir_path = os.path.join(save_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing directory: {subdir_path}")
            for i, file_name in enumerate(sorted(os.listdir(subdir_path))):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(subdir_path, file_name)
                    print(f"Loading file: {file_path}")
                    try:
                        data = np.load(file_path)  # 加载 .npy 文件
                        visualize_marker_point_flow(data, i, subdir)  # 调用可视化函数
                    except Exception as e:
                        print(f"Error loading or visualizing file {file_path}: {e}")

if __name__ == "__main__":
    # 调用加载和可视化函数
    load_and_plot_data(SAVE_DIR)