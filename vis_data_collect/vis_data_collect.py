import sys
import time
# 动态添加项目的根目录到 PYTHONPATH
project_root = "/home/tars/workspace/xx/tactile/track2real"
if project_root not in sys.path:
    sys.path.append(project_root)


import random
from control import control
from envs.robotiq_controller import RobotiqGripper



######   走到夹持peg的点   ######
controller = control.MoveControl(port='/dev/ttyUSB0', baudrate=115200)
gripper = RobotiqGripper('/dev/ttyUSB1', default_speed=50)
gripper.reset()
### zero2cubhome ###
speed = 15000
controller.absoulte_movement('Y',  -199.734, speed, wait = False)  # y右移
controller.absoulte_movement('Z', -122.481, speed, wait = False)    # z向上移动
controller.absoulte_movement('C', -2.4, 0.01*speed, wait = False) # c轴逆时针旋转
controller.absoulte_movement('X', -180.012, speed, wait = False)  # x前进
### graspcubpeg ###
gripper.close()
### cubhome2origin ###
speed = 1500
controller.absoulte_movement('X', -160.012, speed, wait = False)  # x前进
controller.absoulte_movement('Y', -177.109, speed, wait = False)  # y右移
controller.absoulte_movement('Z', -177.481, speed, wait = False)    # z向上移动
    

######  创建移动轨迹  ######
# 极值：
X_min = -162.278
Y_min = -246.975
Z_min = -222.243
C_min = -3.4
X_max = -132.278
Y_max = -117.743
Z_max = -61.584
C_max = -1.4

random.seed(41)

def generate_waypoints(num_points=10):
    waypoints = []
    for _ in range(num_points):
        # 随机生成坐标，范围是负极值到 0
        x = random.uniform(X_min, X_max)
        y = random.uniform(Y_min, Y_max)
        z = random.uniform(Z_min, Z_max)
        c = random.uniform(C_min, C_max)
        waypoints.append((round(x, 1), round(y, 1), round(z, 1), round(c, 3)))  # 保留 1 位小数
    return waypoints

waypoints = generate_waypoints(3)

# 打印生成的途径点
print("生成的固定随机途径点：")
for i, point in enumerate(waypoints):
    print(f"途径点 {i + 1}: X={point[0]}, Y={point[1]}, Z={point[2]}, C={point[3]}")


######  执行移动轨迹  ######
speed = 15000  # 设置运动速度
for i, point in enumerate(waypoints):
# 确保三位小数：
    # controller.absoulte_movement('X', round(point[0], 3), speed, wait=False)
    # controller.absoulte_movement('Y', round(point[1], 3), speed, wait=False)
    # controller.absoulte_movement('Z', round(point[2], 3), speed, wait=False)
    controller.absoulte_movement('X', point[0], speed, wait=False)
    controller.absoulte_movement('Y', point[1], speed, wait=False)
    controller.absoulte_movement('Z', point[2], speed, wait=False)
    controller.absoulte_movement('C', point[3], 0.01*speed, wait=False)
    time.sleep(1)

controller.close()
print("===========over==============")

######  实例化相机  ######
