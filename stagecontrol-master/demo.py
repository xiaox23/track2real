import time

from StageController import CombinedStage


x_origin = -50
y_origin = 50
z_origin = -10
theta_origin = 45



def find_zero(stage):
    stage.seek_and_set_zero()


def go_to_init(stage):
    stage.abs_move('y', y_origin, vel=8, wait=False)
    stage.abs_move('x', x_origin, vel=8, wait=False)
    stage.abs_move('theta', theta_origin, wait=True)
    stage.wait_for_move_stop()
    stage.abs_move('z', z_origin, vel=10, wait=False)
    stage.wait_for_move_stop()


if __name__ == "__main__":
    combined_stage = CombinedStage("/dev/translation_stage", '/dev/rotation_stage')

    go_to_init(combined_stage)
    time.sleep(3)
    combined_stage.return_to_zero_safe()
    print(combined_stage.get_position())
