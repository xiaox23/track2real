import os, sys
import time

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)
from translation_stage import TranslationStage
from rotation_stage_py import RotationStage


class CombinedStage():
    def __init__(self, translation_port: str, rotation_port: str):
        self.translation_stage = TranslationStage(translation_port)
        self.rotation_stage = RotationStage(rotation_port, 1)
        self.axis_mapping_dict = {'x': 2, 'y': 0, 'z': 1}
        self.axis_direction_dict = {'x': 1, 'y': -1, 'z': 1}

    def transform_to_combined(self, translation_position):
        combined_pos = []
        for axis in ['x', 'y', 'z']:
            combined_pos.append(translation_position[self.axis_mapping_dict[axis]] * self.axis_direction_dict[axis])
        return combined_pos

    def get_position(self):
        translation_position = self.translation_stage.get_all_positions()
        self.position = self.transform_to_combined(translation_position)
        self.position.append(self.rotation_stage.get_position())
        return self.position

    def get_velocity(self):
        translation_velocity = self.translation_stage.get_all_velocities()
        self.velocity = self.transform_to_combined(translation_velocity)
        self.velocity.append(self.rotation_stage.get_velocity())
        return self.velocity

    def is_moving(self):
        return self.translation_stage.is_moving() or self.rotation_stage.is_moving()

    def wait_for_move_stop(self):
        while self.is_moving():
            time.sleep(0.1)

    def rel_move(self, axis, pos, vel=2, wait=False):
        self_mapping_dict = {'x': 0, 'y': 1, 'z': 2}
        if axis in ['x', 'y', 'z']:
            target_abs_pos = self.get_position()[self_mapping_dict[axis]] + pos  # in combined stage coordinate
            self.translation_stage.rel_move(axis=self.axis_mapping_dict[axis], pos=self.axis_direction_dict[axis] * pos,
                                            vel=vel, wait=wait)  # in translation stage coordinate
            if wait:
                check_ok = False
                while not check_ok:
                    current_pos = self.get_position()[self_mapping_dict[axis]]  # in combined stage coordinate
                    # add position check
                    if (
                            abs(current_pos - target_abs_pos) > 0.05
                    ):
                        print("Position error:", current_pos - target_abs_pos)
                        self.translation_stage.abs_move(axis=self.axis_mapping_dict[axis],
                                                        pos=self.axis_direction_dict[axis] * target_abs_pos,
                                                        vel=vel, wait=wait)
                    else:
                        check_ok = True

        else:
            self.rotation_stage.rel_move(pos, wait=wait)

    def abs_move(self, axis, pos, vel=2, wait=False):
        if axis in ['x', 'y', 'z']:
            self.translation_stage.abs_move(axis=self.axis_mapping_dict[axis], pos=self.axis_direction_dict[axis] * pos,
                                        vel=vel, wait=wait)
            if wait:
                check_ok = False
                while not check_ok:
                    mapping_dict = {'x': 0, 'y': 1, 'z': 2}
                    current_pos = self.get_position()[mapping_dict[axis]]
                    # add position check
                    if (
                            abs(current_pos - pos) > 0.05
                    ):
                        print("Position error:", current_pos - pos)
                        self.translation_stage.abs_move(axis=self.axis_mapping_dict[axis],
                                                        pos=self.axis_direction_dict[axis] * pos,
                                                        vel=vel, wait=wait)
                    else:
                        check_ok = True
        else:
            self.rotation_stage.abs_move(pos, wait=wait)

    def stop_axis(self, axis):
        if axis in ['x', 'y', 'z']:
            self.translation_stage.stop_axis(axis=self.axis_mapping_dict[axis])
        else:
            self.rotation_stage.stop()

    def stop_all_axes(self):
        self.translation_stage.stop_axis(axis=self.axis_mapping_dict['x'])
        self.translation_stage.stop_axis(axis=self.axis_mapping_dict['y'])
        self.translation_stage.stop_axis(axis=self.axis_mapping_dict['z'])
        self.rotation_stage.stop()

    def seek_and_set_zero(self):
        self.translation_stage.seek_and_set_zero()
        self.rotation_stage.seek_and_set_zero()

    def return_to_zero(self):
        self.rotation_stage.abs_move(0, False)
        self.translation_stage.return_to_zero()
        self.rotation_stage.wait_for_motion_finished()

    def return_to_zero_safe(self):
        self.translation_stage.abs_move(axis=self.axis_mapping_dict['z'], pos=0, vel=5, wait=True)
        self.rotation_stage.abs_move(0, False)
        self.translation_stage.return_to_zero()
        self.rotation_stage.wait_for_motion_finished()

    def __del__(self):
        del self.translation_stage
        del self.rotation_stage
