import control

""" calibrate X/Y/Z/C axis to inital position """

if __name__ == "__main__":
     # 初始化运动控制实例
    controller = control.MoveControl(port='/dev/ttyUSB0', baudrate=115200)
    
    # 增量运动示例
    speed = 15000  # 设置运动速度

    controller.incremental_movement('Y',  -199.734, speed, wait = False)  # y左移
    
    controller.incremental_movement('Z', -122.481, speed, wait = False)    # z向上移动

    controller.incremental_movement('C', -2.4, 0.01*speed, wait = False) # c轴逆时针旋转

    controller.incremental_movement('X', -180.012, speed, wait = False)  # x前进

    # 关闭串口连接
    controller.close()