import time

from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input  as inputMsg
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
from pymodbus.client.sync import ModbusSerialClient
from math import ceil


def update_robotiq_command(char, command, speed=100):
    """Update the command according to the character entered by the user."""

    if char == 'a':
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 1
        command.rGTO = 1
        command.rSP = speed
        command.rFR = 150

    if char == 'r':
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 0

    if char == 'c':
        command.rPR = 168

    if char == 'o':
        command.rPR = 0

        # If the command entered is a int, assign this value to rPRA
    try:
        command.rPR = int(char)
        if command.rPR > 255:
            command.rPR = 255
        if command.rPR < 0:
            command.rPR = 0
    except ValueError:
        pass

    if char == 'f':
        command.rSP += 25
        if command.rSP > 255:
            command.rSP = 255

    if char == 'l':
        command.rSP -= 25
        if command.rSP < 0:
            command.rSP = 0

    if char == 'i':
        command.rFR += 25
        if command.rFR > 255:
            command.rFR = 255

    if char == 'd':
        command.rFR -= 25
        if command.rFR < 0:
            command.rFR = 0

    return command


class Communication:
    def __init__(self):
        self.client = None

    def connectToDevice(self, device):
        """Connection to the client - the method takes the IP address (as a string, e.g. '192.168.1.11') as an argument."""
        self.client = ModbusSerialClient(method='rtu', port=device, stopbits=1, bytesize=8, baudrate=115200, timeout=2)
        if not self.client.connect():
            print("Unable to connect to %s" % device)
            return False
        return True

    def disconnectFromDevice(self):
        """Close connection"""
        self.client.close()

    def sendCommand(self, data):
        """Send a command to the Gripper - the method takes a list of uint8 as an argument. The meaning of each variable depends on the Gripper model (see support.robotiq.com for more details)"""
        # make sure data has an even number of elements
        if (len(data) % 2 == 1):
            data.append(0)

        # Initiate message as an empty list
        message = []

        # Fill message by combining two bytes in one register
        for i in range(0, len(data) // 2):
            message.append((data[2 * i] << 8) + data[2 * i + 1])

        response = self.client.write_registers(0x03E8, message, unit=0x0009)
        if response.isError():
            raise Exception("modbus exception")

    def getStatus(self, numBytes):
        """
        Sends a request to read, wait for the response and returns the Gripper status. The method gets the number
        of bytes to read as an argument
        """
        numRegs = int(ceil(numBytes / 2.0))

        # Get status from the device
        response = self.client.read_holding_registers(0x07D0, numRegs, unit=0x0009)
        if response.isError():
            raise Exception("modbus exception")
        # Instantiate output as an empty list
        output = []

        # Fill the output with the bytes in the appropriate order
        for i in range(0, numRegs):
            output.append((response.getRegister(i) & 0xFF00) >> 8)
            output.append(response.getRegister(i) & 0x00FF)

        # Output the result
        return output


class RobotiqGripper:
    """Base class (communication protocol agnostic) for sending commands and receiving the status of the Robotic 2F gripper"""

    def __init__(self, serial_port: str, default_speed=100):
        # Initiate output message as an empty list
        self.message = []
        self.client = Communication()
        self.client.connectToDevice(serial_port)
        self.hand_command = outputMsg.Robotiq2FGripper_robot_output()
        self.status = self.get_status()
        if self.is_active():
            self.hand_command = update_robotiq_command('a', self.hand_command, default_speed)

    def verify_command(self, command):
        """Function to verify that the value of each variable satisfy its limits."""
        # Verify that each variable is in its correct range
        command.rACT = max(0, command.rACT)
        command.rACT = min(1, command.rACT)

        command.rGTO = max(0, command.rGTO)
        command.rGTO = min(1, command.rGTO)

        command.rATR = max(0, command.rATR)
        command.rATR = min(1, command.rATR)

        command.rPR = max(0, command.rPR)
        command.rPR = min(255, command.rPR)

        command.rSP = max(0, command.rSP)
        command.rSP = min(255, command.rSP)

        command.rFR = max(0, command.rFR)
        command.rFR = min(255, command.rFR)

        # Return the modified command
        return command

    def send_command(self, command):
        """Function to update the command which will be sent during the next sendCommand() call."""
        # Limit the value of each variable
        command = self.verify_command(command)
        # Initiate command as an empty list
        message = []
        # Build the command with each output variable
        # To-Do: add verification that all variables are in their authorized range
        message.append(command.rACT + (command.rGTO << 3) + (command.rATR << 4))
        message.append(0)
        message.append(0)
        message.append(command.rPR)
        message.append(command.rSP)
        message.append(command.rFR)

        num_trials = 0
        while num_trials < 10:
            try:
                num_trials += 1
                self.client.sendCommand(message)
                break
            except Exception as e:
                print(e)

        if num_trials >= 10:
            raise Exception("Modbus communication failure")

    def get_status(self):
        """Request the status from the gripper and return it in the Robotiq2FGripper_robot_input msg type."""

        # Acquire status from the Gripper
        num_trials = 0
        while num_trials < 10:
            try:
                num_trials += 1
                status = self.client.getStatus(6)
                break
            except Exception as e:
                print(e)

        if num_trials >= 10:
            raise Exception("Modbus communication failure")

        # Message to output
        message = inputMsg.Robotiq2FGripper_robot_input()

        # Assign the values to their respective variables
        message.gACT = (status[0] >> 0) & 0x01
        message.gGTO = (status[0] >> 3) & 0x01
        message.gSTA = (status[0] >> 4) & 0x03
        message.gOBJ = (status[0] >> 6) & 0x03
        message.gFLT = status[2]
        message.gPR = status[3]
        message.gPO = status[4]
        message.gCU = status[5]

        return message

    def reset(self):
        self.hand_command = outputMsg.Robotiq2FGripper_robot_output()
        self.hand_command = update_robotiq_command('r', self.hand_command)
        self.send_command(self.hand_command)
        time.sleep(2)
        self.hand_command = update_robotiq_command('a', self.hand_command)
        self.send_command(self.hand_command)
        time.sleep(10)

    def go_to_position(self, pos: int, wait=True):
        if self.get_status().gACT == 0:
            return False
        self.hand_command = update_robotiq_command(str(pos), self.hand_command)
        self.send_command(self.hand_command)
        time.sleep(0.1)
        if wait:
            while self.is_moving():
                time.sleep(0.1)

    def open(self, wait=True):
        if self.get_status().gACT == 0:
            return False
        self.hand_command = update_robotiq_command('o', self.hand_command)
        self.send_command(self.hand_command)
        time.sleep(0.1)
        if wait:
            while self.is_moving():
                time.sleep(0.1)

    def close(self, wait=True):
        if self.get_status().gACT == 0:
            return False
        self.hand_command = update_robotiq_command('c', self.hand_command)
        self.send_command(self.hand_command)
        time.sleep(0.1)
        if wait:
            while self.is_moving():
                time.sleep(0.1)

    def is_active(self):
        self.status = self.get_status()
        return self.status.gSTA == 0x03

    def is_moving(self):
        self.status = self.get_status()
        return self.status.gOBJ == 0x00

    def get_pos(self):
        self.status = self.get_status()
        return int(self.status.gPO)


if __name__ == "__main__":
    gripper = RobotiqGripper('/dev/ttyUSB1')
    # gripper.reset()
    gripper.close()
    time.sleep(1)
    gripper.open()
    time.sleep(1)
    gripper.close()
    exit()
    #time.sleep(1)

    # for i in range(255):
    #     gripper.go_to_position(i)

