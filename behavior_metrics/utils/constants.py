from pathlib import Path

PRETRAINED_MODELS_DIR = 'models/'
DATASETS_DIR = 'datasets_opencv/'
CARLA_TEST_SUITE_DIR = 'configs/CARLA/test_suites/'
ROOT_PATH = str(Path(__file__).parent.parent)
MIN_EXPERIMENT_PERCENTAGE_COMPLETED = 3
# General timeout reference for each circuit extracted using explicit_brain
CIRCUITS_TIMEOUTS = {
    'simple_circuit.launch': 75,
    'simple_circuit_basic.launch': 75,
    'simple_circuit_no_red_line.launch': 75,
    'simple_circuit_no_red_line_no_wall.launch': 75,
    'simple_circuit_no_sun.launch': 75,
    'simple_circuit_no_wall.launch': 75,
    'simple_circuit_white_line.launch': 75,
    'simple_circuit_white_road.launch': 75,
    'simple_circuit_white_road_no_red_line.launch': 75,
    'many_curves.launch': 151,
    'many_curves_no_red_line.launch': 151,
    'many_curves_no_red_line_no_wall.launch': 151,
    'many_curves_no_wall.launch': 151,
    'montmelo_line.launch': 122,
    'montmelo_line_no_sun.launch': 122,
    'montmelo_line_no_wall.launch': 122,
    'montmelo_no_red_line.launch': 122,
    'montmelo_no_red_line_no_wall.launch': 122,
    'montmelo_white_line.launch': 122,
    'montmelo_white_road.launch': 122,
    'montmelo_white_road_no_red_line.launch': 122,
    'montreal_line.launch': 248,
    'montreal_line_no_wall.launch': 248,
    'montreal_no_red_line.launch': 248,
    'montreal_no_red_line_no_wall.launch': 248,
    'nurburgring_line.launch': 88,
    'nurburgring_line_no_wall.launch': 88,
    'nurburgring_no_red_line.launch': 88,
    'nurburgring_no_red_line_no_wall.launch': 88,
}

CARLA_TOWNS_SPAWN_POINTS = {
    'Town01':
        [
            "10.0, 2.0, 1.37, 0.0, 0.0, 180.0",
            "40.0, 2.0, 1.37, 0.0, 0.0, 180.0",
            "-2.0, -10.0, 1.37, 0.0, 0.0, -90.0",
            "-2.0, -280.0, 1.37, 0.0, 0.0, -90.0",
            "20.0, -330.0, 1.37, 0.0, 0.0, 0.0",
            "300.0, -330.0, 1.37, 0.0, 0.0, 0.0",
            "397.0, -310.0, 1.37, 0.0, 0.0, 90.0",
            "397.0, -50.0, 1.37, 0.0, 0.0, 90.0",
            "20.0, -2.0, 1.37, 0.0, 0.0, 0.0",
            "200.0, -2.0, 1.37, 0.0, 0.0, 0.0",
            "300.0, -2.0, 1.37, 0.0, 0.0, 0.0",
            "350.0, -2.0, 1.37, 0.0, 0.0, 0.0",
            "392.0, -50.0, 1.37, 0.0, 0.0, -90.0",
            "392.0, -300.0, 1.37, 0.0, 0.0, -90.0",
            "300.0, -327.0, 1.37, 0.0, 0.0, 180.0",
            "20.0, -327.0, 1.37, 0.0, 0.0, 180.0",
            "2.0, -280.0, 1.37, 0.0, 0.0, 90.0",
            "2.0, -20.0, 1.37, 0.0, 0.0, 90.0",
        ],
    'Town02' :
        [
            "55.3, -105.6, 1.37, 0.0, 0.0, 180.0",
            "10.3, -105.6, 1.37, 0.0, 0.0, 180.0",
            "100.3, -105.6, 1.37, 0.0, 0.0, 180.0",
            "-7.0, -120.6, 1.37, 0.0, 0.0, -90.0",
            "-7.0, -270.6, 1.37, 0.0, 0.0, -90.0",
            "10.0, -307.0, 1.37, 0.0, 0.0, 0.0",
            "100.0, -307.0, 1.37, 0.0, 0.0, 0.0",
            "150.0, -307.0, 1.37, 0.0, 0.0, 0.0",
            "193.0, -290.0, 1.37, 0.0, 0.0, 90.0",
            "193.0, -150.6, 1.37, 0.0, 0.0, 90.0",
            "55.3, -110, 1.37, 0.0, 0.0, 0.0",
            "10.3, -110, 1.37, 0.0, 0.0, 0.0",
            "100.3, -110, 1.37, 0.0, 0.0, 0.0",
            "-3.0, -120.6, 1.37, 0.0, 0.0, 90.0",
            "-3.0, -270.6, 1.37, 0.0, 0.0, 90.0",
            "10.0, -303.0, 1.37, 0.0, 0.0, 180.0",
            "100.0, -303.0, 1.37, 0.0, 0.0, 180.0",
            "150.0, -303.0, 1.37, 0.0, 0.0, 180.0",
            "190.0, -290.0, 1.37, 0.0, 0.0, -90.0",
            "190.0, -150.0, 1.37, 0.0, 0.0, -90.0",
        ],
    'Town03' :
        [
            "246.0, 150.0, 1.37, 0.0, 0.0, 90.0",
            "246.0, 0.5, 1.37, 0.0, 0.0, 90.0",
            "245.0, -43.5, 1.37, 0.0, 0.0, 90.0",
            "243.0, -100., 1.37, 0.0, 0.0, 90.0",
            "241.0, -150.0, 1.37, 0.0, 0.0, 90.0",
            "200.0, 208, 1.37, 0.0, 0.0, 180.0",
            "0.0, 208, 1.37, 0.0, 0.0, 180.0",
            "-50.0, 210, 1.37, 0.0, 0.0, 180.0",
            "-88.0, 170, 1.37, 0.0, 0.0, -90.0",
            "-88.0, 0, 1.37, 0.0, 0.0, -90.0",
            "-88.0, -150.0, 1.37, 0.0, 0.0, -90.0",
            "-50.0, -206.0, 1.37, 0.0, 0.0, 0.0",
            "0.0, -207.0, 1.37, 0.0, 0.0, 0.0",
            "100.0, -207.0, 1.37, 0.0, 0.0, 0.0",
            "232.0, 0.0, 1.37, 0.0, 0.0, -90.0",
            "232.0, -40.0, 1.37, 0.0, 0.0, -90.0"
            "232.0, -100.0, 1.37, 0.0, 0.0, -90.0",
            "0.0, 193, 1.37, 0.0, 0.0, 0.0",
            "-50.0, 195, 1.37, 0.0, 0.0, 0.0",
        ],
    'Town04':
        [
            "248.7, 371.2, 1.37, 0.0, 0.0, 0.0",
            "-16.0, -184.6, 1.37, 0.0, 0.0, -90.0",
            "381.5, 60.0, 1.37, 0.0, 0.0, -90.0"
        ],
    "Town05":
        [
            "20, -187.5, 1.37, 0.0, 0.0, 180.0",
            "210.1, -87.3, 1.37, 0.0, 0.0, 90.0",
            "189, -87.3, 1.37, 0.0, 0.0, -90.0"
        ],
    "Town06":
        [
            "659.0, -70.5, 1.37, 0.0, 0.0, -90.0",
            "351.5, 10.5, 1.37, 0.0, 0.0, 0.0",
            "351.5, 24.5, 1.37, 0.0, 0.0, 180.0",
            "672.5, -70.5, 1.37, 0.0, 0.0, 90.0"
        ],
    "Town07":
        [
            "-3.0, 243.0, 1.37, 0.0, 0.0, 180.0",
            "70.5, 5.0, 1.37, 0.0, 0.0, 60.0",
            "-184.5, -107.2, 1.37, 0.0, 0.0, 180.0"
        ],
}

CARLA_TOWNS_TIMEOUTS = {
    "Carla/Maps/Town01": 210,
    "Carla/Maps/Town02": 120,
    "Carla/Maps/Town03": 210,
    "Carla/Maps/Town04": 180,
    "Carla/Maps/Town05": 230,
    "Carla/Maps/Town06": 210,
    "Carla/Maps/Town07": 120,
}

CARLA_INFRACTION_PENALTIES = {
    'collision_walker': 0.5,
    'collision_vehicle': 0.6,
    'collision_static': 0.65,
    'wrong_turn': 0.7,
    'time_out': 0.7,
    'red_light': 0.7
}