import sys, os
import time
import rotation_stage

class RotationStage():
    def __init__(self, port:str, id=1):
        self.stage = rotation_stage.RotationStage(port, id)

    def get_position(self):
        return -self.stage.get_position()

    def get_velocity(self):
        return -self.stage.get_velocity()

    def is_moving(self):
        return self.stage.is_moving()

    def rel_move(self, pos, wait=False):
        return self.stage.rel_move(-pos, wait)

    def abs_move(self, pos, wait=False):
        return self.stage.abs_move(-pos, wait)

    def stop(self):
        return self.stage.stop()

    def seek_and_set_zero(self):
        return self.stage.seek_and_set_zero()

    def wait_for_motion_finished(self):
        return self.stage.wait_for_motion_finished()


if __name__ == "__main__":
    stage = RotationStage("/dev/ttyUSB1", 1)

    time.sleep(1)

    #stage.seek_and_set_zero()

    stage.abs_move(45)
    stage.wait_for_motion_finished()
    stage.rel_move(-90)
    stage.wait_for_motion_finished()
    stage.abs_move(0)
    stage.wait_for_motion_finished()