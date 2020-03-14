import os
from pathlib import Path
from environs import Env

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

env = Env()
env.read_env(str(PROJECT_DIR))

PAUSE_SIMULATION = env.str("PAUSE_SIMULATION ", "p")
RESUME_SIMULATION = env.str("RESUME_SIMULATION", "r")
LOAD_DATASET = env.str("LOAD_DATASET", "l")
RECORD_DATASET = env.str("RECORD_DATASET", "d")
STOP_RECORD_DATASET = env.str("STOP_RECORD_DATASET", "s")
CHANGE_BRAIN = env.str("CHANGE_BRAIN", "c")
EVALUATE_BRAIN = env.str("EVALUATE_BRAIN", "e")
