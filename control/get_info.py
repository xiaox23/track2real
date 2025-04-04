import control 

""" get the information about the X/Y/Z/C axis """

if __name__ == '__main__':
    # 初始化运动控制实例
    controller = control.MoveControl(port='/dev/ttyUSB0', baudrate=115200)
    
    # 获取轴位置信息
    controller.get_axis_position('X')
    controller.get_axis_position('Y')
    controller.get_axis_position('Z')
    controller.get_axis_position('C')

    # 获取轴速度信息
    v = controller.get_all_velocities()
    print("v: ", v)

    # 判断平台是否有移动
    result = controller.is_moving()
    print("result:", result)
    
    # 关闭串口连接
    controller.close()