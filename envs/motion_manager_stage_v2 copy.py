import time
import os
import sys
script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)
from utils.real_sense_help import RealSenseHelp


class Config:

    # FROM MODEL
    CUBIOD_HOLE_CENTER_TO_HEXAGON_CENTER_Y = 45.25
    HOLE_CENTER_TO_GARAGE_CENTER_X = 55
    GELSIGHT_LENGTH = 32.0
    TABLE_GAP = 25

    # MEASURE
    BASE_Z_POS = -249.4  # mm, 测量得到，gelsight的表面刚好与在存放区的50mm高的轴相碰
    CUBIOD_GARAGE_POS_X = -104.6
    ORIGIN_POS_Y = 160.4
    CUBIOD_GARAGE_POS_THETA = -15


    GRASP_OFFSET = 8.0
    GARAGR_DEPTH = 17.4 + 0.1  # 存放区底部到孔表面距离， 给0.1mm的余量
    MOVE_PEG_HEIGHT = 6.0  # 在孔的表面运送peg的z坐标， 6mm余量
    GEIPPER_SAFE_HEIGHT = 20.0  # 夹爪安全高度
    GRIPPER_CLOSE_POS = 117
    CUBIOD_GARAGE_POS_Y = ORIGIN_POS_Y - CUBIOD_HOLE_CENTER_TO_HEXAGON_CENTER_Y/2







class MotionManagerStageV2:
    def __init__(
        self,
        translation_stage_port,
        rotation_stage_port,
        gripper_port,
        peg,
        gripper_speed,
        grasp_height_offset=0,
    ):
        # 初始化路径和依赖模块
        script_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(script_path)
        sys.path.append(os.path.join(script_path, ".."))

        from StageController.combined_stage import CombinedStage
        from envs.robotiq_controller import RobotiqGripper

        # 运动阶段相关参数
        # 关键z坐标
        self.base_z_pos = Config.BASE_Z_POS
        self.set_grasp_height_offset(grasp_height_offset)
        self.gripper_safe_z_pos = (
            self.base_z_pos + Config.GEIPPER_SAFE_HEIGHT
        )  # 夹爪安全高度

        self.gripper_speed = gripper_speed

        # 位置定义
        cuboid_garage_pos = {
            "x": Config.CUBIOD_GARAGE_POS_X,
            "y": Config.CUBIOD_GARAGE_POS_Y,
            "theta": Config.CUBIOD_GARAGE_POS_THETA,
        }  # 测量得到
        cubiod_save_pos = {
            "x": cuboid_garage_pos["x"],
            "y": cuboid_garage_pos["y"] - 5 * Config.TABLE_GAP,
            "theta": cuboid_garage_pos["theta"],
        }
        cuboid_hole_pos = {
            "x": cuboid_garage_pos["x"] + Config.HOLE_CENTER_TO_GARAGE_CENTER_X,
            "y": cuboid_garage_pos["y"],
        }
        hexagon_garage_pos = {
            "x": cuboid_garage_pos["x"],
            "y": cuboid_garage_pos["y"] + Config.CUBIOD_HOLE_CENTER_TO_HEXAGON_CENTER_Y,
            "theta": cuboid_garage_pos["theta"],
        }
        hexagon_hole_pos = {
            "x": cuboid_hole_pos["x"],
            "y": cuboid_hole_pos["y"] + Config.CUBIOD_HOLE_CENTER_TO_HEXAGON_CENTER_Y,
        }
        hexagon_save_pos = {
            "x": cubiod_save_pos["x"],
            "y": cubiod_save_pos["y"]
            + 9 * Config.TABLE_GAP
            + Config.CUBIOD_HOLE_CENTER_TO_HEXAGON_CENTER_Y,
            "theta": cuboid_garage_pos["theta"],
        }
        origin_pos = {
            "x": cuboid_hole_pos["x"],
            "y": cuboid_hole_pos["y"]
            + Config.CUBIOD_HOLE_CENTER_TO_HEXAGON_CENTER_Y / 2,
        }

        self.pre_pose = {
            "cuboid": {
                "garage_pos": cuboid_garage_pos,
                "save_pos": cubiod_save_pos,
                "hole_pos": cuboid_hole_pos,
                "gripper_close_pos": Config.GRIPPER_CLOSE_POS,
            },
            "hexagon": {
                "garage_pos": hexagon_garage_pos,
                "save_pos": hexagon_save_pos,
                "hole_pos": hexagon_hole_pos,
                "gripper_close_pos": Config.GRIPPER_CLOSE_POS,
            },
            "origin_point": origin_pos,
        }

        self.peg = peg
        if self.peg not in self.pre_pose.keys():
            raise ValueError(f"Invalid peg type: {self.peg}")

        # 初始化硬件控制器
        self.gripper = RobotiqGripper(gripper_port, default_speed=self.gripper_speed)
        self.stage = CombinedStage(translation_stage_port, rotation_stage_port)

    """setting"""

    def set_peg(self, peg):
        self.peg = peg
        if self.peg not in self.pre_pose.keys():
            raise ValueError(f"Invalid peg type: {self.peg}")

    def set_grasp_height_offset(self, grasp_height_offset):
        self.grasp_peg_in_garage_z_pos = (
            self.base_z_pos
            - Config.GELSIGHT_LENGTH / 2
            - Config.GRASP_OFFSET
            + grasp_height_offset
        )  # 减去gelsight长度的一半到中间，再往深走8mm，因为z方向随机是-10~-6, 再给个可以变动的量
        self.insertion_start_z_pos = (
            self.grasp_peg_in_garage_z_pos + Config.GARAGR_DEPTH
        )  # 存放区底部到孔表面距离， 给0.1mm的余量
        self.move_peg_z_pos = (
            self.insertion_start_z_pos + Config.MOVE_PEG_HEIGHT
        )  # 在孔的表面运送peg的z坐标， 6mm余量


    """base move for stage"""

    def relative_move(self, axis, pos, vel=2, wait=False):
        """相对移动"""
        if axis in ["x", "y", "z", "theta"]:
            self.stage.rel_move(axis, pos, vel, wait)
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def absolute_move(self, axis, pos, vel=2, wait=False):
        """绝对移动"""
        if axis in ["x", "y", "z", "theta"]:
            self.stage.abs_move(axis, pos, vel, wait)
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def stop_axis(self, axis):
        """停止指定轴"""
        self.stage.stop_axis(axis)

    def stop(self):
        """停止所有轴"""
        self.stage.stop_all_axes()

    def wait_for_move_stop(self):
        """等待移动停止"""
        self.stage.wait_for_move_stop()

    def get_position(self):
        """获取当前位置"""
        return self.stage.get_position()

    def is_moving(self):
        """判断是否在移动"""
        return self.stage.is_moving()

    """gripper control"""

    def get_gripper_pos(self):
        return self.gripper.get_pos()

    def open_gripper(self):
        return self.gripper.open()

    def close_gripper(self, gripper_close_pos=None):
        if gripper_close_pos:
            self.gripper.go_to_position(gripper_close_pos)
        else:
            self.gripper.go_to_position(self.pre_pose[self.peg]["gripper_close_pos"])

    def reset_gripper(self):
        self.go_to_safe_height()
        self.gripper.reset()

    # def set_gripper_close_pos(self, pos):
    #     self.gripper_close_pos = pos

    """complex move"""

    def go_to_safe_height(self, wait=True):
        """移动到夹爪安全z坐标"""
        self.absolute_move("z", self.gripper_safe_z_pos, vel=10, wait=wait)

    def go_to_move_peg_height(self, wait=True, from_save_or_garage = False):
        """移动到侧移高度"""
        if from_save_or_garage:
            self.relative_move("z", 3, vel=2, wait=wait)
            self.absolute_move("z", self.move_peg_z_pos, vel=10, wait=wait)
        else:
            z_pos_now = self.get_position()[2]
            if z_pos_now < self.insertion_start_z_pos:
                print("111")
                self.absolute_move("z", self.insertion_start_z_pos+1, vel=2, wait=wait)
                print("222")
            self.absolute_move("z", self.move_peg_z_pos,vel=10,  wait=wait)

    def go_to_insertion_start_height(self, wait=True):
        """移动到插孔起始高度，也就是之后就要添加偏置开始插孔"""
        self.absolute_move("z", self.insertion_start_z_pos, vel=10, wait=wait)

    def go_to_grasp_peg_in_garage_height(self, wait=True):
        """从存放区取轴的高度"""
        self.absolute_move("z", self.grasp_peg_in_garage_z_pos, vel=10, wait=wait)

    def go_to_release_peg_in_garage_height(self, wait=True):
        """从存放区放轴的高度"""
        self.absolute_move("z", self.grasp_peg_in_garage_z_pos + 4,vel=10,  wait=wait)
        self.absolute_move("z", self.grasp_peg_in_garage_z_pos + 2, vel=2, wait=wait)

    def go_to_garage_xytheta(self):
        """移动到存放区位置"""
        for axis in self.pre_pose[self.peg]["garage_pos"].keys():
            self.stage.abs_move(
                axis, self.pre_pose[self.peg]["garage_pos"][axis], vel=10, wait=False
            )
        self.stage.wait_for_move_stop()

    def go_to_save_xytheta(self):
        """移动到存放区位置"""
        for axis in self.pre_pose[self.peg]["save_pos"].keys():
            self.stage.abs_move(
                axis, self.pre_pose[self.peg]["save_pos"][axis], vel=10, wait=False
            )
        self.stage.wait_for_move_stop()

    def go_to_hole_xytheta(self, peg_or_origin):
        """移动到插孔位置"""
        if peg_or_origin == "peg":
            for axis in self.pre_pose[self.peg]["hole_pos"].keys():
                self.stage.abs_move(
                    axis, self.pre_pose[self.peg]["hole_pos"][axis], vel=10, wait=False
                )
            self.stage.wait_for_move_stop()
        else:
            for axis in self.pre_pose["origin_point"].keys():
                self.stage.abs_move(
                    axis, self.pre_pose["origin_point"][axis], vel=10, wait=False
                )
            self.stage.wait_for_move_stop()

    def move_peg_from_anywhere_to_garage(self, reset_tracker = False):
        """将 peg 返回车库"""
        self.go_to_move_peg_height()
        self.go_to_garage_xytheta()
        self.go_to_release_peg_in_garage_height()
        # self.relative_move("z", -0.3, vel=5, wait=True) # 0.3mm的进量，保证能吸上
        self.open_gripper()
        time.sleep(0.5)
        if reset_tracker:
            self.go_to_safe_height()
        # time.sleep(0.5)
        # self.go_to_safe_height()

    def move_peg_from_garage_to_save(self):

        self.open_gripper()
        self.go_to_safe_height()
        self.go_to_garage_xytheta()
        self.go_to_grasp_peg_in_garage_height()
        if self.peg=="cuboid":
            self.relative_move("x", 5, wait=True)
        # self.relative_move("x", 5, wait=True)
        self.close_gripper()
        time.sleep(0.5)

        self.go_to_move_peg_height( from_save_or_garage=True)
        self.go_to_save_xytheta()
        self.go_to_release_peg_in_garage_height()
        self.open_gripper()
        time.sleep(0.5)
        self.go_to_safe_height()

    def move_peg_from_save_to_garage(self):

        self.open_gripper()
        self.go_to_safe_height()
        self.go_to_save_xytheta()
        self.go_to_grasp_peg_in_garage_height()
        self.close_gripper()
        time.sleep(0.5)
        self.go_to_move_peg_height( from_save_or_garage=True)
        self.go_to_garage_xytheta()
        if self.peg=="cuboid":
            self.relative_move("x", 5, wait=True)
        self.go_to_release_peg_in_garage_height()
        self.open_gripper()
        time.sleep(0.5)
        self.go_to_safe_height()

    def move_peg_from_garage_to_origin(self, reset_tracker = False):
        if reset_tracker:
            self.open_gripper()
            self.go_to_safe_height()
            self.go_to_garage_xytheta()
            self.go_to_grasp_peg_in_garage_height()

        self.close_gripper()
        time.sleep(0.5)
        self.go_to_move_peg_height( from_save_or_garage=True)
        self.go_to_hole_xytheta("origin")
        self.go_to_insertion_start_height()






    # def grasp_peg_from_garage(self):
    #     """从车库抓取 peg"""
    #     self.go_to_safe_height()
    #     self.open_gripper()
    #     self.go_to_garage_xytheta()
    #     self.go_to_grasp_peg_in_garage_height()
    #     self.gripper.go_to_position(self.pre_pose[self.peg]["gripper_close_pos"])
    #     time.sleep(0.5)
    #     self.go_to_move_peg_height()

    # def regrasp_peg(self):
    #     """重新抓取 peg"""
    #     self.move_peg_from_anywhere_to_garage()
    #     self.move_peg_from_garage_to_origin()

    # def go_to_hole_origin(self):
    #     """移动到插孔原点"""
    #     self.go_to_move_peg_height()
    #     self.go_to_hole_xytheta(peg_or_origin="origin")
    #     self.go_to_insertion_start_height()

    def go_relative_offset(self, x, y, theta, z, vel=5):
        """移动一个offset"""
        self.stage.rel_move("z", z, vel=vel, wait=False)
        self.stage.rel_move("x", x, vel=vel, wait=False)
        self.stage.rel_move("y", y, vel=vel, wait=False)
        self.stage.rel_move("theta", theta, wait=False)
        self.stage.wait_for_move_stop()

    # def reset_peg(self):
    #     """重置 peg"""
    #     self.regrasp_peg()
    #     self.go_to_hole_origin()

    def close(self):
        """关闭系统"""
        self.stage.return_to_zero_safe()


def main():
    # 初始化 MotionManagerStageV2
    motion_manager = MotionManagerStageV2(
        translation_stage_port="/dev/translation_stage",
        rotation_stage_port="/dev/rotation_stage",
        gripper_port="/dev/hande",
        peg="hexagon",
        grasp_height_offset=0,
        # gripper_close_pos=117,
        gripper_speed=50,
    )
    # motion_manager.go_to_garage_xytheta()
    # motion_manager.go_to_safe_height()
    motion_manager.move_peg_from_anywhere_to_garage(True)
    # rs_helper = RealSenseHelp("cuboid")
    # rs_helper.reset_tracker()
    # path = r"/home/xulab10/lcy/challenge_2025_real/test/test_track_img_2/"
    # vision_result = rs_helper.get_vision_result(max_points=128)
    # rs_helper.save_images_together(vision_result["color_image"], vision_result["depth_image"],path=path+"2-1.jpg")



    # 测试流程
    print("Starting test...")
    print(motion_manager.get_position())
    # motion_manager.go_relative_offset(10,-10,30,5)
    # motion_manager.absolute_move("y", 169.8,wait=True)
    # print(motion_manager.get_position())

    # motion_manager.move_peg_from_save_to_garage()
    # motion_manager.move_peg_from_garage_to_origin()
    # vision_result = rs_helper.get_vision_result(max_points=128)
    # rs_helper.save_images_together(vision_result["color_image"], vision_result["depth_image"], path=path + "2-2.jpg")
    # motion_manager.go_relative_offset(-5.1,-5.1,10.3,1.8)
    # print(motion_manager.get_position())
    # vision_result = rs_helper.get_vision_result(max_points=128)
    # rs_helper.save_images_together(vision_result["color_image"], vision_result["depth_image"], path=path + "2-3.jpg")
    #
    #
    # motion_manager.move_peg_from_anywhere_to_garage(True)
    # vision_result = rs_helper.get_vision_result(max_points=128)
    # rs_helper.save_images_together(vision_result["color_image"], vision_result["depth_image"], path=path + "2-4.jpg")

    # motion_manager.move_peg_from_garage_to_save()
    # motion_manager.set_peg("cuboid")
    # motion_manager.move_peg_from_save_to_garage()
    # motion_manager.move_peg_from_garage_to_origin()
    # motion_manager.move_peg_from_anywhere_to_garage()
    # motion_manager.move_peg_from_garage_to_save()
    # motion_manager.set_peg("hexagon")
    # motion_manager.move_peg_from_save_to_garage()
    # motion_manager.move_peg_from_garage_to_origin()
    # motion_manager.move_peg_from_anywhere_to_garage()
    # motion_manager.move_peg_from_garage_to_save()


    # motion_manager.go_to_safe_height()
    # motion_manager.absolute_move("z",-52,wait=True)

    # pos = motion_manager.get_position()
    # print(pos)
    # print(motion_manager.insertion_start_z_pos)
    # motion_manager.open_gripper()
    # motion_manager.go_to_safe_height()
    # motion_manager.go_to_save_xytheta()
    # motion_manager.go_to_grasp_peg_in_garage_height()
    # motion_manager.go_to_garage_xytheta()
    # motion_manager.go_to_release_peg_in_garage_height()
    # motion_manager.open_gripper()
    # motion_manager.relative_move("z",-2)
    # motion_manager.go_to_garage_xytheta()
    # motion_manager.close_gripper()
    # motion_manager.go_to_move_peg_height()
    # motion_manager.go_to_save_xytheta()
    # motion_manager.go_to_release_peg_in_garage_height()
    # motion_manager.go_to_garage_xytheta()
    # motion_manager.go_to_release_peg_in_garage_height()
    # motion_manager.open_gripper()
    # motion_manager.go_to_safe_height()
    # motion_manager.go_to_garage_xytheta()
    # print(motion_manager.pre_pose["origin_point"])
    # motion_manager.return_peg_to_garage()



    

    # motion_manager.
    # motion_manager.go_to_safe_height()
    # motion_manager.go_to_grasp_peg_in_garage_height()
    # motion_manager.open_gripper()
    # motion_manager.go_to_move_peg_height()
    # pos = motion_manager.get_position()
    # print(pos)
    # pass

    # # 1. 回到安全高度
    # print("\nGoing to safe height...")
    # motion_manager.go_to_safe_height()
    # time.sleep(2)  # 等待运动完成

    #     # 2. 抓取 peg
    #     print("\nGrasping peg from garage...")
    #     motion_manager.grasp_peg_from_garage()
    #     time.sleep(2)
    #
    #     # 3. 移动到插孔位置
    #     print("\nMoving to hole...")
    #     motion_manager.go_to_hole_origin()
    #     time.sleep(2)
    #
    #     # 4. 插入 peg
    #     print("\nInserting peg...")
    #     motion_manager.go_to_insertion_start_height()
    #     time.sleep(2)
    #
    #     # 5. 返回车库
    #     print("\nReturning peg to garage...")
    #     motion_manager.return_peg_to_garage()
    #     time.sleep(2)
    #
    #     print("\nTest completed successfully!")
    #
    # except Exception as e:
    #     print(f"Error occurred: {e}")
    #     motion_manager.stop()  # 停止所有运动

    # finally:
    # 关闭系统
    # print("\nClosing system...")
    # motion_manager.close()
    print(motion_manager.get_position())

def test_motion():
    motion_manager = MotionManagerStageV2(
        translation_stage_port="/dev/translation_stage",
        rotation_stage_port="/dev/rotation_stage",
        gripper_port="/dev/hande",
        peg="cuboid",
        grasp_height_offset=0,
        # gripper_close_pos=117,
        gripper_speed=50,
    )
    motion_manager.go_to_garage_xytheta()
    motion_manager.go_relative_offset(-5.1, -5.1, 10.3, 1.8)
    motion_manager.go_to_garage_xytheta()
    motion_manager.close_gripper()
    motion_manager.open_gripper()
    motion_manager.close()
    print("motion_manager is ok")

def test_motion():
    motion_manager = MotionManagerStageV2(
        translation_stage_port="/dev/translation_stage",
        rotation_stage_port="/dev/rotation_stage",
        gripper_port="/dev/hande",
        peg="cuboid",
        grasp_height_offset=0,
        # gripper_close_pos=117,
        gripper_speed=50,
    )
    motion_manager.go_to_garage_xytheta()
    motion_manager.go_relative_offset(-5.1, -5.1, 10.3, 1.8)
    motion_manager.go_to_garage_xytheta()
    motion_manager.close_gripper()
    motion_manager.open_gripper()
    motion_manager.close()
    print("motion_manager is ok")




if __name__ == "__main__":
    main()
    # test_motion()
    # reset_zero()
