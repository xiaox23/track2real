import os
import sys
import numpy as np
import cv2
import torch
import open3d as o3d
import pyrealsense2 as rs
from segment_anything import sam_model_registry, SamPredictor
import argparse

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

from TrackAnythingmaster.track_anything import TrackingAnything


T3 = np.array(
    [
        [0.99983607, 0.01619122, 0.00810356, -0.01812094],
        [-0.01578929, 0.99875016, -0.0474216, 0.00797355],
        [-0.00886125, 0.04728588, 0.99884209, 0.01844764],
        [0.0, 0.0, 0.0, 1.0]
    ]
)
T3_2 = np.array([[ 9.99983309e-01 , 1.47117224e-03 ,-5.58721859e-03 , 2.60000000e-03],
 [-1.32985837e-03 , 9.99681233e-01 , 2.52123839e-02, -1.38000000e-02],
 [ 5.62252933e-03 ,-2.52045329e-02 , 9.99666504e-01, -4.00000000e-04],
 [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

# T3 = np.eye(4)


def apply_transformation(points, transform_matrix):
    """应用4x4变换矩阵到点云"""
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = (transform_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]


class TrackAnything:
    def __init__(self):
        # 解析参数
        # def parse_augment():
        #     parser = argparse.ArgumentParser()
        #     parser.add_argument("--device", type=str, default="cuda:0")
        #     parser.add_argument("--sam_model_type", type=str, default="vit_h")
        #     parser.add_argument("--port", type=int, default=6080)
        #     parser.add_argument("--debug", action="store_true")
        #     parser.add_argument("--mask_save", default=False)
        #     args = parser.parse_args()
        #     return args

        args = argparse.ArgumentParser()
        args.sam_model_type = "vit_h"

        # 模型路径
        SAM_checkpoint = r"/home/tars/workspace/xx/tactile/sam_vit_h_4b8939.pth"
        xmem_checkpoint = r"TrackAnythingmaster/checkpoints/XMem-s012.pth"
        e2fgvi_checkpoint = r"/home/xulab10/lcy/challenge_2025_real/TrackAnythingmaster/checkpoints/E2FGVI-HQ-CVPR22.pth"
        args.port = 12212
        args.device = "cuda"

        # 初始化模型
        self.model = TrackingAnything(
            SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args
        )

        # 初始化掩码
        self.masks = {"masks": [], "mask_names": []}
        self.mask_dropdown = []

    def add_mask(self, image, points, labels):
        self.model.samcontroler.sam_controler.reset_image()
        self.model.samcontroler.sam_controler.set_image(image)
        mask, _, _ = self.model.first_frame_click(image, points, labels, multimask=True)
        self.masks["masks"].append(mask)
        mask_name = f"mask_{len(self.masks['masks']):03d}"
        self.masks["mask_names"].append(mask_name)
        self.mask_dropdown.append(mask_name)

    def clean_mask(self):
        self.masks = {"masks": [], "mask_names": []}
        self.mask_dropdown = []

    def generate_template_mask(self):
        if not self.masks["masks"]:
            raise ValueError("No masks added yet")
        mask_stack = np.stack(self.masks["masks"], axis=0)
        mask_stack = mask_stack.astype(np.int64)
        self.template_mask = mask_stack[0].copy()
        for i in range(1, len(self.masks["masks"])):
            self.template_mask += mask_stack[i] * (i + 1)
        self.template_mask = np.clip(self.template_mask, 0, i + 1)

    def track(self, image, is_first=False):
        if is_first:
            self.clean_mem()
        mask, logit, painted_image = self.model.generator_image(
            image, self.template_mask, is_first
        )
        return mask, logit, painted_image

    def clean_mem(self):
        self.model.xmem.clear_memory()


class RealSenseHelp:
    def __init__(self, peg_hole_type):
        # 初始化 RealSense 流水线
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.rs_pipeline.start(self.rs_config)
        self.align = rs.align(rs.stream.color)

        # 初始化 TrackAnything
        self.track_anything = TrackAnything()

        # 初始化标定点
        self.peg_hole_mark_point = {
            # "hexagon": {
            #     "pt": np.array(
            #         [  [365, 243], [425, 242], [413, 284], [344, 268],[380,269],[358,262],[372,262],[347,205],[447,203],
            #            [350, 288], [371, 293], [350, 313], [364, 328]]),
            #     "label_peg": np.array([  0, 0, 0,0,0,  0,0,0,0,
            #                              1, 1, 1, 1]),
            #     "label_hole": np.array([  1, 1, 1,1,1, 1,1,1,1,
            #                               0, 0, 0, 0, ]),
            # },
            "hexagon": {
                "pt": np.array(
                    [
                        [347, 205],
                        [447, 203],
                        [365, 243],
                        [425, 242],
                        [413, 284],
                        [350, 288],
                        [371, 293],
                        [350, 313],
                        [364, 328],
                    ]
                ),
                "label_peg": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]),
                "label_hole": np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
            },
            "cuboid": {
                "pt": np.array(
                    [
                        [287, 212],
                        [380, 207],
                        [302, 243],
                        [362, 244],
                        [304, 282],
                        [350, 288],
                        [371, 293],
                        [350, 313],
                        [364, 328],
                    ]
                ),
                "label_peg": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]),
                "label_hole": np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
            },
        }
        self.peg_hole_type = peg_hole_type

        # 共享的帧数据缓存
        self.shader_frame = {
            "color_frame": None,
            "depth_frame": None,
            "aligned_frames": None,
        }

    def set_peg(self, peg_hole_type):
        self.peg_hole_type = peg_hole_type

    def reset_tracker(self):
        self.take_picture()
        first_color_image = self.get_rgb()
        # 添加掩码
        self.add_masks(first_color_image)
        # 生成模板掩码
        self.track_anything.generate_template_mask()
        # 跟踪
        mask, _, _ = self.track_anything.track(first_color_image, is_first=True)
        # self.save_images_together
        # self.show_painted_image()
        self.take_picture_end()
        first_color_image_mask = self.add_mask_to_rgb(first_color_image, mask)

        return first_color_image_mask

    def add_mask_to_rgb(self, rgb_image, mask, alpha=0.5):
        blue = [255, 0, 0]
        orange = [0, 165, 255]
        color_layer = np.zeros_like(rgb_image)
        color_layer[mask == 1] = blue
        color_layer[mask == 2] = orange
        result = cv2.addWeighted(rgb_image, 1 - alpha, color_layer, alpha, 0)
        return result

    def add_masks(self, color_image):
        self.track_anything.clean_mask()
        # 添加第一个掩码
        self.track_anything.add_mask(
            color_image,
            self.peg_hole_mark_point[self.peg_hole_type]["pt"],
            self.peg_hole_mark_point[self.peg_hole_type]["label_peg"],
        )
        # 添加第二个掩码
        self.track_anything.add_mask(
            color_image,
            self.peg_hole_mark_point[self.peg_hole_type]["pt"],
            self.peg_hole_mark_point[self.peg_hole_type]["label_hole"],
        )

    def take_picture(self):
        self.shader_frame["aligned_frames"] = self.get_raw_frames()
        self.shader_frame["color_frame"], self.shader_frame["depth_frame"] = (
            self.shader_frame["aligned_frames"].get_color_frame(),
            self.shader_frame["aligned_frames"].get_depth_frame(),
        )
        depth_image = (
            np.asanyarray(self.shader_frame["depth_frame"].get_data()).astype(
                np.float32
            )
            / 1000
        )
        color_image = np.asanyarray(self.shader_frame["color_frame"].get_data())
        return color_image, depth_image

    def get_raw_frames(self):
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                frames = self.rs_pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                return aligned_frames
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        "Failed to get valid frames after multiple attempts."
                    )

    def get_rgb(self) -> np.ndarray:
        if self.shader_frame["color_frame"]:
            color_image = np.asanyarray(self.shader_frame["color_frame"].get_data())
            return color_image
        else:
            raise ValueError("No color frame available. Call take_picture() first.")

    def get_depth(self) -> np.ndarray:
        if self.shader_frame["depth_frame"]:
            depth_image = (
                np.asanyarray(self.shader_frame["depth_frame"].get_data()).astype(
                    np.float32
                )
                / 1000
            )
            return depth_image
        else:
            raise ValueError("No depth frame available. Call take_picture() first.")

    def show_painted_image(self):
        cv2.imshow("painted_image", self.painted_image)
        cv2.waitKey(1)

    def show_images_together(self, rgb_image, depth_image, window_name="RGB and Depth"):
        normalized_depth = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3
        )
        depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        combined = np.hstack([rgb_image, depth_colormap])
        cv2.imshow(window_name, combined)
        cv2.waitKey(1)

    def save_images_together(self, rgb_image, depth_image, path):
        normalized_depth = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3
        )
        depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        combined = np.hstack([rgb_image, depth_colormap])
        cv2.imwrite(path, combined)

    def get_pointcloud(self, mask=None):
        color_frame = self.shader_frame["color_frame"]
        depth_frame = self.shader_frame["depth_frame"]
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000
        color_image = np.asanyarray(color_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        valid_depth = depth_image != 0
        if mask is not None:
            valid = valid_depth & mask.astype(bool)
        else:
            valid = valid_depth

        valid_indices = np.where(valid)
        v, u = valid_indices
        depths = depth_image[valid]

        x = (u - intrinsics.ppx) * depths / intrinsics.fx
        y = (v - intrinsics.ppy) * depths / intrinsics.fy
        z = depths

        points = np.column_stack((x, y, z))
        colors = color_image[valid] / 255.0

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud

    def get_vision_result(self, max_points):
        self.take_picture()
        color_image = self.get_rgb()
        depth_image = self.get_depth()
        mask, logit, painted_image = self.track_anything.track(color_image)
        self.painted_image = painted_image

        # 生成掩罩后的图像和深度图
        color_image_masked = color_image.copy()
        color_image_masked[~mask.astype(bool)] = [255, 255, 255]
        depth_image_masked = depth_image.copy()
        depth_image_masked[~mask.astype(bool)] = 0

        # 生成点云
        pc = rs.pointcloud()
        points = pc.calculate(self.shader_frame["depth_frame"])
        depth_intrinsics = rs.video_stream_profile(
            self.shader_frame["depth_frame"].profile
        ).get_intrinsics()
        points_vertics = np.asarray(points.get_vertices()).view(np.float32)
        points_vertics = np.array(points_vertics).reshape(-1, 3)
        A = apply_transformation(points_vertics, T3@T3_2)
        w, h = depth_intrinsics.width, depth_intrinsics.height
        # points_frame = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h,w, 3)
        points_frame = np.asanyarray(A.reshape(h, w, 3))
        vision_result = {
            "raw_color_image": color_image.copy(),
            "raw_depth_image": depth_image.copy(),
            "raw_point_cloud": points_frame.copy(),
        }

        # 应用掩码
        valid = mask != 0
        points_frame[~valid] = [0, 0, 0]
        peg_points = points_frame[mask == 1]
        hole_points = points_frame[mask == 2]

        # 采样点云
        peg_points = self.sample_points(peg_points, max_points)
        hole_points = self.sample_points(hole_points, max_points)
        object_point_cloud = np.stack([peg_points, hole_points], axis=0)

        vision_result["mask"] = mask.copy()
        vision_result["color_image"] = color_image_masked.copy()
        vision_result["depth_image"] = depth_image_masked.copy()
        vision_result["point_cloud"] = points_frame.copy()
        vision_result["object_point_cloud"] = object_point_cloud.copy()
        vision_result["raw_color_image_masked"] = self.add_mask_to_rgb(
            color_image.copy(), mask
        )

        self.take_picture_end()
        return vision_result

    def sample_points(self, points: np.ndarray, num_samples: int):
        if len(points) <= num_samples:
            indices = np.random.choice(len(points), num_samples, replace=True)
        else:
            indices = np.random.choice(len(points), num_samples, replace=False)
            return points[indices]

    def take_picture_end(self):
        self.shader_frame["color_frame"] = None
        self.shader_frame["depth_frame"] = None
        self.shader_frame["aligned_frames"] = None

    def visualize_pointcloud(self, point_cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd])

    def close(self):
        self.rs_pipeline.stop()


# 示例使用
if __name__ == "__main__":
    # 初始化 RealSenseHelp
    rs_helper = RealSenseHelp("cuboid")
    color_img, depth_img = rs_helper.take_picture()
    # draw point
    peg_hole_mark_point = {
            # "hexagon": {
            #     "pt": np.array(
            #         [  [365, 243], [425, 242], [413, 284], [344, 268],[380,269],[358,262],[372,262],[347,205],[447,203],
            #            [350, 288], [371, 293], [350, 313], [364, 328]]),
            #     "label_peg": np.array([  0, 0, 0,0,0,  0,0,0,0,
            #                              1, 1, 1, 1]),
            #     "label_hole": np.array([  1, 1, 1,1,1, 1,1,1,1,
            #                               0, 0, 0, 0, ]),
            # },
            "hexagon": {
                "pt": np.array(
                    [
                        [347, 205],
                        [447, 203],
                        [365, 243],
                        [425, 242],
                        [413, 284],
                        [350, 288],
                        [371, 293],
                        [350, 313],
                        [364, 328],
                    ]
                ),
                "label_peg": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]),
                "label_hole": np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
            },
            "cuboid": {
                "pt": np.array(
                    [
                        [287, 212],
                        [380, 207],
                        [302, 243],
                        [362, 244],
                        [304, 282],
                        [350, 288],
                        [371, 293],
                        [350, 313],
                        [364, 328],
                    ]
                ),
                "label_peg": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]),
                "label_hole": np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
            },
        }
    for point, label in zip(
        peg_hole_mark_point["cuboid"]["pt"], peg_hole_mark_point["cuboid"]["label_peg"]
    ):
        if label:
            cv2.circle(color_img, tuple(point), 5, (0, 0, 255), -1)
        else:
            cv2.circle(color_img, tuple(point), 5, (255, 0, 0), -1)
    cv2.imshow("test", color_img)
    cv2.waitKey(0)

    # 初始化
    rs_helper.peg_hole_mark_point = peg_hole_mark_point
    aa = rs_helper.reset_tracker()
    cv2.imshow("test", aa)
    cv2.waitKey(0)
    #
    # # 获取视觉结果
    # # pathl
    # vision_result_1 = rs_helper.get_vision_result(max_points=128)
    # rs_helper.save_images_together(vision_result_1["color_image"], vision_result_1["depth_image"])

    # combined_image = np.hstack([vision_result_1["color_image"], cv2.applyColorMap(
    #     cv2.normalize(vision_result_1["depth_image"], None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)])
    # cv2.imwrite("combined_image.png", combined_image)

    # 可视化点云
    # rs_helper.visualize_pointcloud(vision_result["object_point_cloud"][0])
    # rs_helper.visualize_pointcloud(vision_result["object_point_cloud"][1])
    # draw peg pc and hole pc in one window
    # peg_pcd = o3d.geometry.PointCloud()
    # peg_pcd.points = o3d.utility.Vector3dVector(vision_result["object_point_cloud"][0])
    # hole_pcd = o3d.geometry.PointCloud()
    # hole_pcd.points = o3d.utility.Vector3dVector(vision_result["object_point_cloud"][1])
    # o3d.visualization.draw_geometries([peg_pcd, hole_pcd])

    # 结束
    rs_helper.close()
