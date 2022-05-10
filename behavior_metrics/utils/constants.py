from pathlib import Path

PRETRAINED_MODELS_DIR = 'models/'
DATASETS_DIR = 'datasets_opencv/'
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

