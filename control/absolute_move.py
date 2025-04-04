import control
""" control X/Y/Z/C to specific position with specific speed """

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
    controller.absoulte_movement('X', -30, speed)  #  x前进
#     controller.absoulte_movement('X', 30, speed)  #  x后退
  
    # Y axis
#     controller.absoulte_movement('Y',  30, speed)  # y左移
    controller.absoulte_movement('Y',  -30, speed)  # y右移
 
    # Z axis
    controller.absoulte_movement('Z', -20, speed)    # z向上移动
#     controller.absoulte_movement('Z', 20, speed)    # z向下移动
    
    # C axis
    controller.absoulte_movement('C', -2.4, 0.01*speed)  # c轴逆时针旋转
#     controller.absoulte_movement('C', 2.4, 0.01*speed)  # c轴顺时针旋转


    # 关闭串口连接
    controller.close()