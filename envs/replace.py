
# motion_manager = MotionManagerStageV2("/dev/translation_stage", "/dev/rotation_stage", "/dev/hande", "hexagon", 50)


# # 打开机器人夹具，释放可能正在抓取的物体，以便安全地操控钉子。
#     def open_gripper(self):
#         return self.gripper.open()


# 将夹具重置为默认状态，可能包括关闭夹具或进行校准。
    # def reset_gripper(self):
    #     self.go_to_safe_height()
    #     self.gripper.reset()


# 将机器人手臂移动到预定义的安全高度，以避免碰撞，确保在进行任何操作前的安全。
    # def go_to_safe_height(self, wait=True):
    #     """移动到夹爪安全z坐标"""
    #     self.absolute_move("z", self.gripper_safe_z_pos, vel=10, wait=wait)1


# 检查夹具是否当前处于激活状态（即正在抓取物体）。返回一个布尔值指示其状态。
# motion_manager.gripper.is_active()


# 将机器人手臂移动到特定的“车库”位置，该位置由XY坐标和方向（theta）定义，可能是一个休息或存储钉子的位置。
# motion_manager.go_to_garage_xytheta()





# 在运动管理器中设置当前的钉子类型（例如六角形或立方体），允许其执行特定于该钉子的操作。
self.motion_manager.set_peg(self.peg)


# 将钉子从当前位置移动到车库位置，确保安全存放。
# self.motion_manager.move_peg_from_anywhere_to_garage(True)


# 将钉子从车库位置移动到“保存”位置，可能是钉子在不使用时的指定区域。
self.motion_manager.move_peg_from_garage_to_save()


# 将钉子从保存位置移动回车库位置，为未来的使用做好准备。
self.motion_manager.move_peg_from_save_to_garage()


# 将钉子从车库位置移动到原点位置，通常是操作开始的位置。参数指示是否等待完成。
self.motion_manager.move_peg_from_garage_to_origin(True)


# 根据指定的X、Y、Z和旋转（theta）偏移量，将钉子相对其当前位置移动。也定义了移动的速度。
self.motion_manager.go_relative_offset(
    x=float(offset_x),
    y=float(offset_y),
    theta=float(offset_theta),
    z=float(offset_z),
    vel=10,
)


# 获取机器人手臂的当前Z位置，指示其在工作空间中的高度。
# self.motion_manager.get_position()


# 检查机器人手臂是否当前正在移动。返回一个布尔值指示其状态。
# self.motion_manager.is_moving()


# 立即停止正在进行的机器人手臂移动，确保安全和控制。
self.motion_manager.stop()


# 将机器人手臂移动到特定的Z位置（current_z），以定义的速度进行移动。wait参数指示是否在移动完成前阻塞程序执行。
# self.motion_manager.absolute_move("z", current_z, vel=3, wait=True)


# 关闭运动管理器，可能涉及关闭通信或清理资源。
self.motion_manager.close()


# 阻塞执行，直到机器人手臂完全停止移动，确保后续命令仅在手臂静止后执行
# self.motion_manager.wait_for_move_stop()


