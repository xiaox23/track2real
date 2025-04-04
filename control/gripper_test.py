import serial
import time
import binascii


class GripperController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1,
                 parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS):
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout,
                                 parity=parity, stopbits=stopbits, bytesize=bytesize)

    def activate(self):
        """
        激活夹爪。
        """
        print("Activating the gripper...")
        self.ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30')
        data_raw = self.ser.readline()
        data = binascii.hexlify(data_raw)
        print("Activate response: ", data)
        time.sleep(0.1)

    def close_gripper(self):
        """
        关闭夹爪（完全闭合）。
        """
        print("Closing the gripper...")
        self.ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29')
        data_raw = self.ser.readline()
        data = binascii.hexlify(data_raw)
        print("Close response: ", data)
        time.sleep(2)

    def open_gripper(self):
        """
        打开夹爪（完全张开）。
        """
        print("Opening the gripper...")
        self.ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19')
        data_raw = self.ser.readline()
        data = binascii.hexlify(data_raw)
        print("Open response: ", data)
        time.sleep(2)

    def get_position(self):
        """
        获取夹爪当前位置信息，寄存器地址为 0x07D2 (Byte 4)。
        """
        print("Getting current position of the gripper...")
        # 发送读取指令，读取0x07D0寄存器，读取3个寄存器（6字节数据，包含位置和状态信息）
        self.ser.write(b'\x09\x04\x07\xD0\x00\x03\xB1\xCE')  # 起始地址0x07D0，读取3个寄存器
        time.sleep(0.05)  # 等待设备响应

        # 尝试读取完整的响应数据（Modbus RTU协议要求完整性）
        data_raw = self.ser.read(11)  # Modbus响应应为11字节
        if len(data_raw) < 11:
            print(f"Error: Incomplete response received ({len(data_raw)} bytes).")
            return None

        # 打印原始数据用于调试
        print("Raw response: ", binascii.hexlify(data_raw))

        # 验证返回数据是否正确
        if data_raw[0] != 0x09 or data_raw[1] != 0x04:  # 校验从机地址和功能码
            print("Error: Invalid response header.")
            return None

        # 提取位置数据（Byte 4，即data_raw[5]）
        position = data_raw[5]  # 位置值在响应的第6字节
        print(f"Current Position: {position} (0-255 scale)")
        return position

    def is_moving(self):
        """
        检查夹爪是否正在移动。
        """
        print("Checking if the gripper is moving...")
        self.ser.write(b'\x09\x04\x07\xD0\x00\x01\x30\x0F')  # 读取状态寄存器指令
        data_raw = self.ser.readline()
        if not data_raw:
            print("No response from the gripper.")
            return False

        data = binascii.hexlify(data_raw)
        print("Received data for moving check: ", data)

        if len(data) >= 6:  # 确保数据长度正确
            status_byte = data[4:6]  # 状态字节
            gOBJ = (int(status_byte, 16) >> 6) & 0x03  # 提取gOBJ状态
            is_moving = gOBJ == 0x00  # gOBJ为0表示正在移动
            print(f"Gripper is {'moving' if is_moving else 'not moving'}.")
            return is_moving
        else:
            print("Invalid response length.")
            return False

    def go_to_position(self, position):
        """
        移动夹爪到指定位置。
        :param position: 目标位置 (0-255)
        """
        if position < 0 or position > 255:
            print("Invalid position value. It must be between 0 and 255.")
            return

        print(f"Moving the gripper to position {position}...")
        position_hex = f"{position:02X}"  # 转换为两位十六进制字符串
        command = f'09 10 03 E8 00 03 06 09 00 00 {position_hex} FF FF'
        command_bytes = bytes.fromhex(command.replace(" ", ""))
        self.ser.write(command_bytes)

        data_raw = self.ser.readline()
        data = binascii.hexlify(data_raw)
        print("Go to position response: ", data)

        # 等待夹爪停止移动
        while self.is_moving():
            time.sleep(0.1)

    def reset(self):
        """
        重置夹爪。
        """
        print("Resetting the gripper...")
        # 关闭夹爪
        self.ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30')
        time.sleep(2)
        # 激活夹爪
        self.activate()

if __name__ == "__main__":
    gripper = GripperController('/dev/ttyUSB1')
    gripper.activate()
    time.sleep(1)

    # 获取当前位置
    current_position = gripper.get_position()
    print(f"Current Position: {current_position}")

    # # 移动到指定位置
    # gripper.go_to_position(128)

    # # 检查是否正在移动
    # is_moving = gripper.is_moving()
    # print(f"Is moving: {is_moving}")

    # # 打开和关闭夹爪
    # gripper.open_gripper()
    # gripper.close_gripper()