import serial
import time
import copy
import numpy as np

class MoveControl:
    """运动控制类, 用于操作XYZC轴的回零、信息获取和增量运动"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1):
        """
        初始化串口连接
        :param port: 串口端口号
        :param baudrate: 波特率
        :param timeout: 超时时间
        """
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout
        )
        if self.ser.is_open:
            print("串口打开成功")
        else:
            print("串口打开失败")

    @staticmethod
    def f_hexToSignedInt(hexStr, numBits=32):
        """
        十六进制字符串转换为有符号十进制数
        :param hexStr: 输入的十六进制字符串
        :param numBits: 位数 (默认为32位)
        :return: 转换后的有符号整数
        """
        unsignedInt = int(hexStr, 16)
        if (unsignedInt >> (numBits - 1)) & 1:
            signedInt = unsignedInt - (1 << numBits)
        else:
            signedInt = unsignedInt
        return signedInt

    def axis_homing(self, axis):
        """
        将指定的XYZC轴回零
        :param axis: 轴名称 (X, Y, Z, C)
        """
        # 根据轴的类型生成回零指令和查询指令
        if axis == 'X':
            command = 'CJXZx'
            query_command = 'CJXBX'
        elif axis == 'Y':
            command = 'CJXZy'
            query_command = 'CJXBY'
        elif axis == 'Z':
            command = 'CJXZz'
            query_command = 'CJXBZ'
        elif axis == 'C':
            command = 'CJXZc'        # C 轴正向回机械零
            query_command = 'CJXBC'

        # 发送回零指令
        self.ser.write(command.encode())
        print(f"发送回零指令: {command}")
        time.sleep(0.1)

        zero_count = 0
        while True:
            self.ser.write(query_command.encode())
            data = self.ser.read(4)
            if len(data) == 4:
                position = int.from_bytes(data, byteorder='little')  # 小端模式读取
                if position == 0:  # 检测到零点
                    zero_count += 1
                    if zero_count >= 5:  # 连续检测到5次零点位置
                        print(f"{axis}轴回零成功")
                        break
                else:
                    zero_count = 0  # 重置计数器
            time.sleep(0.1)

    def get_axis_position(self, axis):
        """
        获取指定轴的当前位置信息
        :param axis: 轴名称 (X, Y, Z)
        :return: 当前轴的位置（整数）
        """
        if axis == 'X':
            query_command = 'CJXBX'
        elif axis == 'Y':
            query_command = 'CJXBY'
        elif axis == 'Z':
            query_command = 'CJXBZ'
        elif axis == 'C':
            query_command = 'CJXBC'

        # 发送查询指令
        self.ser.write(query_command.encode())
        data = self.ser.read(4)
        if len(data) == 4:
            position = self.f_hexToSignedInt(data.hex())  # 转换为有符号整数
            print(f"{axis}轴当前的位置: {position}")
            return position
        else:
            print(f"未能获取{axis}轴的位置")
            return None

    def absoulte_movement(self, axis, target_position, speed, wait=False):
        """
        控制指定轴以绝对方式运动
        :param axis: 轴名称 (X, Y, Z, C)
        :param target_position: 目标位置（绝对位置）
        :param speed: 运动速度
        """
        if axis == 'X':
            command = f'CJXCgX{target_position}F{speed}$'
            query_command = 'CJXBX'
        elif axis == 'Y':
            command = f'CJXCgY{target_position}F{speed}$'
            query_command = 'CJXBY'
        elif axis == 'Z':
            command = f'CJXCgZ{target_position}F{speed}$'
            query_command = 'CJXBZ'
        elif axis == 'C':
            command = f'CJXCgC{target_position}F{speed}$'
            query_command = 'CJXBC'

        # 查询当前轴的位置
        self.ser.write(query_command.encode())
        data = self.ser.read(4)
        if len(data) == 4:
            position_last = self.f_hexToSignedInt(data.hex())
        else:
            print(f"无法获取{axis}轴的当前位置")
            return

        # 发送增量运动指令
        self.ser.write(command.encode())
        print(f"发送绝对运动指令: {command}")
        time.sleep(0.1)

        while True:
            self.ser.write(query_command.encode())
            data = self.ser.read(4)
            if len(data) == 4:
                position = self.f_hexToSignedInt(data.hex())
                if position == target_position * 1000:
                    print(f"{axis}轴运动到绝对目标位置 {target_position}")
                    break
            time.sleep(0.1)


    def incremental_movement(self, axis, target_position, speed, wait):
        """
        控制指定轴以增量方式运动
        :param axis: 轴名称 (X, Y, Z, C)
        :param target_position: 目标位置（相对当前的位置偏移量）
        :param speed: 运动速度
        """
        # 增量运动不能为0
        if target_position == 0:
            return
        
        if axis == 'X':
            command = f'CJXCGX{target_position}F{speed}$'
            query_command = 'CJXBX'
        elif axis == 'Y':
            command = f'CJXCGY{target_position}F{speed}$'
            query_command = 'CJXBY'
        elif axis == 'Z':
            command = f'CJXCGZ{target_position}F{speed}$'
            query_command = 'CJXBZ'
        elif axis == 'C':
            command = f'CJXCGC{target_position}F{speed}$'
            query_command = 'CJXBC'

        # 查询当前轴的位置
        self.ser.write(query_command.encode())
        data = self.ser.read(4)
        if len(data) == 4:
            position_last = self.f_hexToSignedInt(data.hex())
        else:
            print(f"无法获取{axis}轴的当前位置")
            return

        # 发送增量运动指令
        self.ser.write(command.encode())
        print(f"发送增量运动指令: {command}")
        time.sleep(0.1)

        while True:
            self.ser.write(query_command.encode())
            data = self.ser.read(4)
            if len(data) == 4:
                position = self.f_hexToSignedInt(data.hex())
                if (position - position_last) == target_position * 1000:
                    print(f"{axis}轴运动到目标位置 {target_position}")
                    break
            time.sleep(0.1)

    def get_all_velocities(self):
        X = self.get_axis_position('X')
        Y = self.get_axis_position('Y')
        Z = self.get_axis_position('Z')
        C = self.get_axis_position('C')
        position = [X, Y, Z, C]
        last_position = copy.deepcopy(position)
        last_time = time.time()
        time.sleep(0.1)
        X = self.get_axis_position('X')
        Y = self.get_axis_position('Y')
        Z = self.get_axis_position('Z')
        C = self.get_axis_position('C')
        position = [X, Y, Z, C]
        cur_position = copy.deepcopy(position)
        cur_time = time.time()
        velocity = (np.array(cur_position) - np.array(last_position)) / (cur_time - last_time)
        self.velocity = velocity.tolist()
        return self.velocity
    
    def is_moving(self):
        vel = self.get_all_velocities()
        # print("vel: ", vel)
        if abs(vel[0]) < 1e-3 and abs(vel[1]) < 1e-3 and abs(vel[2]) < 1e-3:
            return False
        else:
            return True

    def wait_for_move_stop(self):
        while self.is_moving():
            time.sleep(0.1)

    def close(self):
        """关闭串口连接"""
        self.ser.close()
        print("串口已关闭")


# 示例代码
if __name__ == "__main__":
    # 初始化运动控制实例
    controller = MoveControl(port='/dev/ttyUSB0', baudrate=115200)

    # 回零操作
    # controller.axis_homing('X')
    # controller.axis_homing('Y')
    # controller.axis_homing('Z')
    controller.axis_homing('C')

    # 获取轴位置信息
    # controller.get_axis_position('X')
    # controller.get_axis_position('Y')
    # controller.get_axis_position('Z')

    # 增量运动示例
    speed = 15000  # 设置运动速度
    """ x负为前进方向,,正为后退方向 y轴正为左,副为右 z轴负为向上方向, 正为向下方向 """
    # controller.incremental_movement('X', 10, speed)
    # controller.incremental_movement('Y', -10, speed)
    # controller.incremental_movement('Z', 5, speed)
    
    # controller.incremental_movement('X', -20, speed) 
    # controller.incremental_movement('Y',  20, speed)
    # controller.incremental_movement('Z', -20, speed)

    controller.incremental_movement('C', -2.4, 0.01*speed)
    time.sleep(1)

    # 关闭串口连接
    controller.close()