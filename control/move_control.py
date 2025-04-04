import serial
import time

# 配置串口, 115200, 8, n, 1
ser = serial.Serial(
    port= '/dev/ttyUSB0',
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)
# 检查 is_open 属性
if ser.is_open:
    print("串口打开成功")
else:
    print("串口打开失败")

# 十六进制字符串转换为有符号十进制数
def f_hexToSignedInt(hexStr, numBits=32):
    # 将十六进制数转换为无符号整数
    unsignedInt = int(hexStr, 16)
    # 检查最高位
    if (unsignedInt >> (numBits - 1)) & 1:
        # 如果最高位是 1，执行二进制补码转换
        signedInt = unsignedInt - (1 << numBits)
    else:
        # 如果最高位是 0，则无需转换
        signedInt = unsignedInt
    # 返回int类型的有符号十进制数
    return signedInt

# 轴回零函数
def axis_homing(axis):
    #print(f"{axis}轴回零中")
    if axis == 'X':
        command = 'CJXZx'        # X 轴正向回机械零
        query_command = 'CJXBX'  # 查询 X 轴信息 ( 四字节 , 高位在前 , 低位在后 )
    elif axis == 'Y': 
        command = 'CJXZy'        # Y 轴正向回机械零
        query_command = 'CJXBY'  # 查询 Y 轴信息 ( 四字节 , 高位在前 , 低位在后 )
    elif axis == 'Z':
        command = 'CJXZz'        # Z 轴正向回机械零
        query_command = 'CJXBZ'  # 查询 Z 轴信息 ( 四字节 , 高位在前 , 低位在后 )
    elif axis == 'C':
        command = 'CJXZc'        # C 轴正向回机械零
        query_command = 'CJXBC'  # 查询 C 轴信息 ( 四字节 , 高位在前 , 低位在后 )

    # 发送回零指令
    ser.write(command.encode())
    print(command.encode())
    time.sleep(0.1)

    last_time = time.time()
    zero_count = 0
    while True:
        # 发送查询指令
        ser.write(query_command.encode())
        #time.sleep(0.01)
        data = ser.read(4)
        print(data.hex())
        if len(data) == 4:
            position = int.from_bytes(data, byteorder='little')  # 假设小端模式
            if position == 0:
                zero_count += 1
                if (zero_count >= 10):  # 连续5次（每秒10次采样）为0
                    print(f"{axis}轴回零成功")
                    break
            else:
                zero_count = 0
        time.sleep(0.1)

# 增量运动函数
def incremental_movement(axis, target_position, speed):
    if axis == 'X':
        command = f'CJXCGX{target_position}F{speed}$' # CJXCG 增量运动，X/Y/Z:需要哪个轴运动， F后为速度
        query_command = 'CJXBX'                       # 查询轴信息
    elif axis == 'Y':
        command = f'CJXCGY{target_position}F{speed}$'
        query_command = 'CJXBY'
    elif axis == 'Z':
        command = f'CJXCGZ{target_position}F{speed}$'
        query_command = 'CJXBZ'
    elif axis == 'C':
        command = f'CJXCGC{target_position}F{speed}$'
        query_command = 'CJXBC'
    
    # 先查询一次
    ser.write(query_command.encode())
    data = ser.read(4)
    if len(data) == 4:
        position_last = f_hexToSignedInt(data.hex())

    # 发送增量运动指令
    ser.write(command.encode())
    time.sleep(0.1)
    #print(command.encode())

    while True:
        # 发送查询指令
        ser.write(query_command.encode())
        data = ser.read(4)
       # print(data.hex())
        if len(data) == 4:
            position = f_hexToSignedInt(data.hex())
            print(f"pos{position}")
            print(f"pos_last{position_last}")
            print(f"pos_target{target_position}")

            if (position-position_last) == target_position*1000:
                print(f"{axis}轴运动到目标位置 {target_position}")
                break
            time.sleep(0.1)

# 增量运动循环
speed = 15000  # 速度值，可根据实际情况调整

################################################归零##############################################
# # #先稍向中心位置运动一端距离
# incremental_movement('X', -10, speed)
# incremental_movement('Y',  5, speed)
# incremental_movement('Z', -10, speed)
# incremental_movement('C', -1, 0.01*speed)

# # # # # 轴回零操作
# axis_homing('X')
# axis_homing('Y')
# axis_homing('Z')
# axis_homing('C')

# time.sleep(1)

# incremental_movement('C', -2.4, 0.01*speed)

# incremental_movement('C', -0.2, 0.01*speed)
################################################归零##############################################

################################################使用##############################################
# #先稍向中心位置运动一端距离
# incremental_movement('X', -20, speed)
# incremental_movement('Y',  20, speed)
# incremental_movement('Z', -20, speed)

# 增量运动
incremental_movement('X', -5, speed)
incremental_movement('X',  5, speed)
incremental_movement('X', -5, speed)
incremental_movement('X',  5, speed)
incremental_movement('X', -5, speed)
incremental_movement('X',  5, speed)

# incremental_movement('Y', -5, speed)
# incremental_movement('Y',  5, speed)
# incremental_movement('Y', -5, speed)
# incremental_movement('Y',  5, speed)
# incremental_movement('Y', -5, speed)
# incremental_movement('Y',  5, speed)

# incremental_movement('Z', -5, speed)
# incremental_movement('Z',  5, speed)
# incremental_movement('Z', -5, speed)
# incremental_movement('Z',  5, speed)
# incremental_movement('Z', -5, speed)
# incremental_movement('Z',  5, speed)

# incremental_movement('C', -0.1, 0.01*speed)
# incremental_movement('C',  0.1, 0.01*speed)
# incremental_movement('C', -0.1, 0.01*speed)
# incremental_movement('C',  0.1, 0.01*speed)
# incremental_movement('C', -0.1, 0.01*speed)
# incremental_movement('C',  0.1, 0.01*speed)
################################################使用##############################################

# 关闭串口
ser.close()