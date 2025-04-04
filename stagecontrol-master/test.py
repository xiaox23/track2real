import time
import socket
import pickle
from StageController import CombinedStage

def find_zero(stage):
    stage.seek_and_set_zero()

def go_to_init(stage):
    stage.abs_move('y', y_origin, vel=8, wait=False)
    stage.abs_move('x', x_origin, vel=8, wait=False)
    stage.abs_move('theta', theta_origin, wait=True)
    stage.wait_for_move_stop()
    stage.abs_move('z', z_origin, vel=10, wait=False)
    stage.wait_for_move_stop()

# 全局变量定义
x_origin = -50
y_origin = 50
z_origin = -10
theta_origin = 45

server_socket = socket.socket()  # 创建一个socket对象

host = "172.17.0.1"
port = 34568  # 设置端口

server_socket.bind((host, port))  # 绑定端口

server_socket.listen(5)  # 等待客户端连接

try:
    # 监听成功
    print("Server started listening...")
except Exception as e:
    # 监听失败
    print("Error occurred while starting the server: ", str(e))

while True:  # 无限循环，持续监听客户端连接
    c, addr = server_socket.accept()  # 建立客户端连接

    while True:  # 循环接收和处理客户端消息
        data_bytes = c.recv(4096)
        msg = (data_bytes.decode())  # 将接收到的字节流反序列化


        if msg == "start":
            if __name__ == "__main__":
                combined_stage = CombinedStage("/dev/translation_stage", '/dev/rotation_stage')

                go_to_init(combined_stage)
                time.sleep(3)
                combined_stage.return_to_zero_safe()
                print(combined_stage.get_position())

    c.close()  # 关闭连接