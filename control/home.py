import control

""" reset the X/Y/Z/C axis to home state"""

if __name__ == "__main__":
    # 初始化运动控制实例
    controller = control.MoveControl(port='/dev/ttyUSB0', baudrate=115200)
    case = input("是否第一次回零: 1/2 ?: ")
    if case == '1':

    ########### 设备刚接电时候,第一次回零,必须使用下面的命令 ##############
        controller.axis_homing('X')
        controller.axis_homing('Y')
        controller.axis_homing('Z')
        controller.axis_homing('C')

    ########### 第2次及之后的回零,使用下面的命令加快回零速度 ##############
    elif case == '2':
        speed = 15000
        controller.absoulte_movement('X', 0, speed)
        controller.absoulte_movement('Y', 0, speed)
        controller.absoulte_movement('Z', 0, speed)
        controller.absoulte_movement('C', 0, 0.01*speed)

    # 关闭串口连接
    controller.close()