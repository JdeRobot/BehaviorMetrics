from pydantic import BaseModel


class QLearnConfig(BaseModel):
    actions: int = 3
    debug_level: int = 0
    telemetry: bool = False
    telemetry_mask: bool = True
    plotter_graphic: bool = False
    my_board: bool = True
    save_positions: bool = False
    save_model: bool = True
    load_model: bool = False
    output_dir = "./logs/qlearn_models/qlearn_camera_solved/"
    poi = 1  # The original pixel row is: 250, 300, 350, 400 and 450 but we work only with the half of the image
    actions_set = "simple"  # test, simple, medium, hard
    max_distance = 0.5
    # === CAMERA ===
    # Images size
    width = 640
    height = 480
    center_image = width / 2

    # Maximum distance from the line
    ranges = [300, 280, 250]  # Line 1, 2 and 3
    reset_range = [-40, 40]
    last_center_line = 0
    if poi == 1:
        x_row = [60]
    elif poi == 2:
        x_row = [60, 110]
    elif poi == 3:
        x_row = [
            10,
            60,
            110,
        ]  # The first and last element is not used. Just for metrics
    elif poi == 5:
        x_row = [250, 300, 350, 400, 450]


qlearn = QLearnConfig()