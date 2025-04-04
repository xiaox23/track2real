import control

""" control X/Y/Z to specific position with specific speed """

if __name__ == "__main__":
     # 初始化运动控制实例
    controller = control.MoveControl(port='/dev/ttyUSB0', baudrate=115200)
    
    # 增量运动示例
    speed = 15000  # 设置运动速度
    
    """ x负为前进方向,,正为后退方向 y轴正为左,副为右 z轴负为向上方向, 正为向下方向 """
    # X axis
    controller.incremental_movement('X', 0, speed, wait = False)  # x前进
    # controller.incremental_movement('X', 5, speed, wait = False)  #  x后退
  
#     # Y axis
    # controller.incremental_movement('Y', -50, speed, wait = False)  # y右移
    # controller.incremental_movement('Y',  50, speed, wait = False)  # y左移
    
#     # Z axis
    # controller.incremental_movement('Z', 50, speed, wait = False)    # z向下移动
    # controller.incremental_movement('Z', -50, speed, wait = False)    # z向上移动
    
    # C axis
    # controller.incremental_movement('C', -2.4, 0.01*speed, wait = False)  # c轴逆时针旋转
    # controller.incremental_movement('C', 2.4, 0.01*speed, wait = False)  # c轴顺时针旋转


    # 关闭串口连接
    controller.close()