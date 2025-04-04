import control

""" control X/Y/Z to specific position with specific speed """

if __name__ == "__main__":
     # 初始化运动控制实例
    controller = control.MoveControl(port='/dev/ttyUSB0', baudrate=115200)
    
    # 增量运动示例
    speed = 15000  # 设置运动速度
    """ x:-前进 +后退 """
    """ y:-向右 +向左 """
    """ z:-向上 +向下 """
    """ c:-逆时 +顺时 """
    # X axis
    controller.incremental_movement('X', -10, speed, wait = False)  #  x前进
    controller.incremental_movement('X', 10, speed, wait = False)  # x后退
    
    # Y axis
    controller.incremental_movement('Y', -10, speed, wait = False)  # y右移
    controller.incremental_movement('Y',  10, speed, wait = False)  # y左移
    
    # Z axis
    controller.incremental_movement('Z', -10, speed, wait = False)    # z向上移动
    controller.incremental_movement('Z', 10, speed, wait = False)    # z向下移动
    
    # C axis
    controller.incremental_movement('C', -2.4, 0.01*speed, wait = False)  # c轴逆时针旋转
    controller.incremental_movement('C', 2.4, 0.01*speed, wait = False)  # c轴顺时针旋转


    # 关闭串口连接
    controller.close()