import math
import time

from StageController.combined_stage import CombinedStage
from envs.robotiq_controller import RobotiqGripper

if __name__ == "__main__":
    """
    现在z的范围是-39 到 37.5 。。。。 抽象 归零后正确了
    
    gripper范围是0-255，数字从小到大，两个gelsightmini之间减小
    """

    stage = CombinedStage("/dev/translation_stage", "/dev/rotation_stage")
    gripper = RobotiqGripper("/dev/hande")
    # gripper.open()
    # gripper.go_to_position(0)

    stage.rel_move("theta", 30, 1, True)
    stage.seek_and_set_zero()
    stage.seek_and_set_zero()
    # time.sleep(20)
    # stage.abs_move('z', -10, 1, True)
    # print(stage.get_position())
    # stage.rel_move("x", -1, 1, True)
    # stage.abs_move("z", -36.5, 1, True)
    # stage.abs_move("theta", -15, 1, True)
    # stage.abs_move("y", 47-25+5*25,5,True)
    # gripper.go_to_position(90)
    # gripper.go_to_position(100)  # 117
    # gripper.go_to_position(117) #117
    # gripper.open()
    # stage.rel_move("y",-45.25-4*25,2,True)
    # gripper.go_to_position(117)

    # stage.rel_move('y', -30, 5, True)
    # stage.rel_move('z', 0, 5, True)
    # stage.abs_move('x', -20, 5, True)
    # stage.rel_move('x', -20, 5, True)
    # print(stage.get_position())
    # gripper.open()
    # gripper.reset()
    # gripper.go_to_position(210)
    #
    # for i in range(9):
    #     rotation_degree = i * 10 + 10
    #     last_rotation_degree = i * 10
    #     # input("Press Enter to continue...")
    #     stage.rel_move("x", 5 * math.cos(rotation_degree / 180 * math.pi) - 5 * math.cos(last_rotation_degree / 180 * math.pi), wait=False)
    #     # 5 mm is the distance between the center of the lock and the center of the gripper when pins are lifted
    #     stage.rel_move("y", - 5 * math.sin(rotation_degree / 180 * math.pi) + 5 * math.sin(last_rotation_degree / 180 * math.pi), wait=False)
    #     stage.rel_move('theta', -10, wait=True)



    # stage.abs_move('z', -30, 5, True)
    # stage.rel_move('y', 1, 5, True)
    #           self.zero_x = -10.0
    #             self.zero_y = 205
    #             self.zero_z = 37.5
    # stage.abs_move('y', 205, 5, True)
    # stage.abs_move('z', -31.5, 5, True)
    # stage.abs_move('x', -10, 5, True)
    # stage.rel_move('theta', 10, vel=10, wait=True)
    # print(stage.get_position())
    # stage.rel_move('z', 1, wait=True)
    # gripper.open()
    # gripper.reset()
    # x range: [-75,0]
    # y range: [0 500]
    # z range: [-75 0]
    # gripper.go_to_position(106)
    # gripper.open()
    # stage.rel_move('x', -2, wait=True)
    # stage.abs_move('x', -31, 5, True)
    # stage.abs_move('z', -31.5, wait=True)
    # stage.rel_move('y', 1, wait=True)
    # stage.rel_move('z', 7.5, wait=True)
    print(stage.get_position())
    # stage.return_to_zero_safe()
    # stage.abs_move('theta', 45, wait=True)

    # stage.seek_and_set_zero()
    # exit()