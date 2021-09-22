from pathlib import Path

PRETRAINED_MODELS_DIR = 'models/'
DATASETS_DIR = 'datasets/'
ROOT_PATH = str(Path(__file__).parent.parent)
MIN_EXPERIMENT_PERCENTAGE_COMPLETED = 3
# General timeout reference for each circuit extracted using explicit_brain
CIRCUITS_TIMEOUTS = {
    'simple_circuit.launch': 75,
    'many_curves.launch': 151,
    'montmelo_line.launch': 122,
    'montreal_line.launch': 248,
    'nurburgring_line.launch': 88,
}

