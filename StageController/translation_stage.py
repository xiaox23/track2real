import copy
import time

import numpy as np
import serial
import struct


# 查询x位置
# AA 06 00 F9 16 F6 06 1A 00 00
# X轴正向运动
# AA 14 00 EB 0A F4 14 01 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00
# X轴负向运动
# AA 14 00 EB 09 F4 14 02 00 00 FF 00 00 00 00 00 00 00 00 00 00 00 00 00
# 修改X轴第距离参数
# AA 0C 00 F3 0A F2 0C 09 00 04 FF 00 00 00 00 00

class StageCommand():
    count = 0

    def __init__(self):
        StageCommand.count = (StageCommand.count + 1) % 0x100

        self.header = 0xAA
        self.data_length = 0
        self.card_id = 0
        self.data_length_inv = 0xFF

        self.checksum = 0
        self.cmd_type = 0
        self.data_length2 = 0
        self.cmd_number = StageCommand.count
        self.data_buf: bytearray = bytearray()

    def _calculate_checksum(self):
        checksum = 0
        checksum += self.cmd_type + self.data_length2 + self.cmd_number
        for byte in self.data_buf:
            checksum += byte
        self.checksum = checksum % 0x100

    def pack(self):
        self.data_length = 4 + len(self.data_buf)
        self.data_length2 = self.data_length
        self.data_length_inv = 255 - self.data_length
        self._calculate_checksum()

        byte_stream = bytearray([self.header, self.data_length, self.card_id, self.data_length_inv, self.checksum,
                                 self.cmd_type, self.data_length2, self.cmd_number])
        byte_stream += self.data_buf
        return byte_stream

    def gen_relative_move_cmd(self, axis: int, pos: float, vel=None, acc=None):
        self.cmd_type = 0xF4

        self.data_buf = bytearray(16)
        self.data_buf[1] = axis
        # self.data_buf[2] = 1 if dir > 0 else -1
        self.data_buf[4:8] = struct.pack('<f', pos)

        if vel is None and acc is None:
            self.data_buf[0] = 0x01
        if vel is not None:
            self.data_buf[0] = 0x03
            self.data_buf[8:12] = struct.pack('<f', vel)
            if acc is None:
                acc = 0
            self.data_buf[12:16] = struct.pack('<f', acc)

    def gen_absolute_move_cmd(self, axis: int, pos: float, vel, acc=None):
        self.cmd_type = 0xF4

        self.data_buf = bytearray(16)
        self.data_buf[0] = 0x04
        self.data_buf[1] = axis
        self.data_buf[4:8] = struct.pack('<f', pos)
        self.data_buf[8:12] = struct.pack('<f', vel)
        if acc is None:
            acc = 0
        self.data_buf[12:16] = struct.pack('<f', acc)

    def gen_stop_axis_cmd(self, axis: int):
        self.cmd_type = 0xF4

        self.data_buf = bytearray(16)
        self.data_buf[0] = 0x81
        self.data_buf[1] = axis

    def gen_ret_axis_to_zero_cmd(self, axis: int):
        self.cmd_type = 0xF4

        self.data_buf = bytearray(16)
        self.data_buf[0] = 0xA1
        self.data_buf[1] = axis

    def gen_set_param_cmd(self, axis: int, param_id, value):
        pass

    def gen_get_param_cmd(self, axis: int, param_id):
        self.cmd_type = 0xf5
        self.data_buf = bytearray(2)
        self.data_buf[0] = axis
        self.data_buf[1] = param_id
        pass

    def gen_query_info_cmd(self, axis: int, info_type: str):
        self.cmd_type = 0xF6

        self.data_buf = bytearray(2)
        self.data_buf[0] = axis
        if info_type == 'pos':
            self.data_buf[1] = 0x00
        elif info_type == 'vel':
            self.data_buf[1] = 0x01
        elif info_type == 'move_status':
            self.data_buf[1] = 0x0B
        elif info_type == "all_pos":
            self.data_buf[0] = 0xff
            self.data_buf[1] = 0x80
        elif info_type == "all_vel":
            self.data_buf[0] = 0xff
            self.data_buf[1] = 0x81

    def gen_reset_pos_cmd(self, axis: int):
        self.cmd_type = 0xF8
        self.data_buf = bytearray(8)
        self.data_buf[0] = 0x01
        self.data_buf[1] = axis
        self.data_buf[4:8] = struct.pack('<f', 0)


class TranslationStage():
    def __init__(self, serial_port: str):
        self.port: serial.Serial = serial.Serial(serial_port, 115200)
        self.position = [0, 0, 0]
        self.velocity = [0, 0, 0]

    def get_move_status(self, axis):
        cmd = StageCommand()
        cmd.gen_query_info_cmd(0, 'move_status')
        self.port.write(cmd.pack())

    def get_position(self, axis):
        cmd = StageCommand()
        cmd.gen_query_info_cmd(axis, 'pos')
        self.port.write(cmd.pack())
        header = self.port.read(4)
        if header[0] == 0xF6:
            if header[3] == 0x00:
                data = self.port.read(header[2] + 1)
                pos = struct.unpack('<f', data[0:4])[0]
                self.position[axis] = pos
                return pos

    def get_all_positions(self):
        cmd = StageCommand()
        cmd.gen_query_info_cmd(0, 'all_pos')
        self.port.write(cmd.pack())
        header = self.port.read(4)
        response_ok = False
        if header[0] == 0xF6:
            if header[3] == 0x00:
                data = self.port.read(header[2] + 1)
                self.position[0] = struct.unpack('<f', data[0:4])[0]
                self.position[1] = struct.unpack('<f', data[4:8])[0]
                self.position[2] = struct.unpack('<f', data[8:12])[0]
                response_ok = True
        if not response_ok:
            self.port.flushInput()
            self.port.flushOutput()
        return [self.position[0], self.position[1], self.position[2]]

    def get_velocity(self, axis):
        self.get_all_velocities()
        return self.velocity[axis]

    def get_all_velocities(self):
        last_position = copy.deepcopy(self.get_all_positions())
        last_time = time.time()
        time.sleep(0.1)
        cur_position = copy.deepcopy(self.get_all_positions())
        cur_time = time.time()
        velocity = (np.array(cur_position) - np.array(last_position)) / (cur_time - last_time)
        self.velocity = velocity.tolist()
        return self.velocity

    def is_moving(self):
        self.get_all_velocities()
        if abs(self.velocity[0]) < 1e-3 and abs(self.velocity[1]) < 1e-3 and abs(self.velocity[2]) < 1e-3:
            return False
        else:
            return True

    def rel_move(self, axis, pos, vel=2, wait=False):
        cmd = StageCommand()
        cmd.gen_relative_move_cmd(axis, pos, vel)
        self.port.write(cmd.pack())
        ret = self.port.read(5)
        time.sleep(0.1)
        if ret[0] == 0xf4 and ret[3] == 0x00:
            result = True
        else:
            result = False
        if wait:
            vel_zero_count = 0
            time.sleep(0.5)
            while vel_zero_count < 5:
                if abs(self.get_velocity(axis)) < 1e-3:
                    vel_zero_count += 1
                else:
                    time.sleep(0.1)
        return result

    def abs_move(self, axis, pos, vel=2, wait=False):
        cmd = StageCommand()
        cmd.gen_absolute_move_cmd(axis, pos, vel)
        self.port.write(cmd.pack())
        ret = self.port.read(5)
        time.sleep(0.1)
        if ret[0] == 0xf4 and ret[3] == 0x00:
            result = True
        else:
            result = False
        if wait:
            vel_zero_count = 0
            time.sleep(0.5)
            while vel_zero_count < 5:
                if abs(self.get_velocity(axis)) < 1e-3:
                    vel_zero_count += 1
                else:
                    time.sleep(0.1)
        return result

    def stop_axis(self, axis):
        cmd = StageCommand()
        cmd.gen_stop_axis_cmd(axis)
        self.port.write(cmd.pack())
        header = self.port.read(4)
        if header[0] == 0xf4:
            if header[3] == 0x00:
                data = self.port.read(header[2] + 1)

    def reset_axis_pos(self, axis):
        cmd = StageCommand()
        cmd.gen_reset_pos_cmd(axis)
        self.port.write(cmd.pack())
        header = self.port.read(4)
        if header[0] == 0xf8:
            if header[3] == 0x00:
                data = self.port.read(header[2] + 1)
                return True

    def seek_and_set_zero(self):
        self.rel_move(0, 80, 1, False)
        self.rel_move(1, 80, 1, False)
        self.rel_move(2, 80, 1, False)
        print("Going to the positive limits...")
        time.sleep(1)
        while self.is_moving():
            time.sleep(1)

        self.rel_move(0, -2, 1, False)
        self.rel_move(1, -2, 1, False)
        self.rel_move(2, -2, 1, False)
        time.sleep(1)
        while self.is_moving():
            time.sleep(1)

        self.rel_move(0, 80, 1, False)
        self.rel_move(1, 80, 1, False)
        self.rel_move(2, 80, 1, False)
        time.sleep(1)
        while self.is_moving():
            time.sleep(1)

        time.sleep(1)
        print("Positive limits reached.")
        
        self.rel_move(0, -1, 1, False)
        self.rel_move(1, -1, 1, False)
        self.rel_move(2, -1, 1, False)
        while self.is_moving():
            time.sleep(1)
        time.sleep(4)
        print("1mm away from the positive limits.")

        self.reset_axis_pos(0)
        self.reset_axis_pos(1)
        time.sleep(2)
        self.reset_axis_pos(2)
        print("Position is reset. Current position: " + str(self.get_all_positions()))

    def return_to_zero(self):
        self.abs_move(0, 0, 5, False)
        self.abs_move(1, 0, 5, False)
        self.abs_move(2, 0, 5, False)
        while self.is_moving():
            time.sleep(1)
        time.sleep(1)
        print("Successfully return to zero.")

    def __del__(self):
        if self.port.is_open:
            self.port.close()


if __name__ == "__main__":
    stage = TranslationStage("/dev/translation_stage")
    stage.rel_move('z', -5, 5, True)
    # stage.rel_move('z', -5, 5, True)
    # stage.rel_move('z', -5, 5, True)
    # stage.rel_move('z', -5, 5, True)

    print(stage.get_position(0), stage.get_position(1), stage.get_position(2))

    stage.return_to_zero()

    print(stage.get_position(0), stage.get_position(1), stage.get_position(2))

    #
    # print(stage.get_all_velocities())
    #
    # print(stage.get_all_positions())
    #
    # print(stage.reset_axis_pos(0))
    # print(stage.get_all_positions())
    #
    # stage.reset_axis_pos(1)
    #
    # print(stage.get_all_positions())
    #
    # stage.reset_axis_pos(2)
    #
    # print(stage.get_all_positions())
    #
    # print(stage.get_all_positions())
    #
    # print(stage.get_all_velocities())
    #
    # stage.rel_move(0, -50, 0.5, False)
    # time.sleep(0.1)
    # stage.rel_move(1, -50, 0.5, False)
    # time.sleep(0.1)
    # stage.rel_move(2, -50, 0.5, False)
    #
    # while stage.is_moving():
    #     print(stage.velocity)
    #     time.sleep(0.1)
    #
    # cmd = StageCommand()
    # cmd_stream = cmd.pack()
    # print(cmd.pack())
    #
    # ser.write(cmd_stream)
    #
    # cmd = StageCommand()
    # cmd_stream = cmd.pack()
    # print(cmd.pack())
    #
    # ser.write(cmd_stream)

    # ser.close()
