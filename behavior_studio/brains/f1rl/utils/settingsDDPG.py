# Global variables

import numpy as np
from utils.configuration import Config

app_configuration = Config('default-rlddpg.yml')

debug_level = 0
telemetry = False
telemetry_mask = False
plotter_graphic = True
my_board = True
save_model = True
load_model = False

tau = app_configuration.tau
gamma = app_configuration.gamma
hidden_size = app_configuration.hidden_size
total_episodes = app_configuration.total_episodes

gazebo_positions_set = app_configuration.gazebo_positions_set

current_env = app_configuration.env

# === POSES ===
if gazebo_positions_set == "pista_simple":
    gazebo_positions = [(0, 53.462, -41.988, 0.004, 0, 0, 1.57, -1.57),
                        (1, 53.462, -8.734,  0.004, 0, 0, 1.57, -1.57),
                        (2, 39.712, -30.741, 0.004, 0, 0, 1.56, 1.56),
                        (3, -6.861,  -36.481, 0.004, 0, 0.01, -0.858, 0.613),
                        (4, 20.043, 37.130, 0.003, 0, 0.103, -1.4383, -1.4383)]

elif gazebo_positions_set == "nurburgring":
    gazebo_positions = [(0, -23.0937, -2.9703, 0, 0.0050, 0.0013, -0.9628, 0.2699),
                        (1, -32.3188, 12.2921, 0, 0.0014, 0.0049, -0.2727, 0.9620),
                        (2, -17.4155, -24.1243, 0, 0.0001, 0.0051, -0.0192, 1),
                        (3, 31.3967, -4.6166, 0, 0.0030, 0.0041, 0.6011, 0.7991),
                        (4, -56.1261, 4.1047, 0, 0.0043, -0.0027, -0.8517, -0.5240)]

# === CAMERA ===
# Images size
width = 640
height = 480
center_image = width/2

# Coord X ROW
# x_row = [250, 300, 350, 400, 450]
# x_row = [250, 450]
x_row = [350]

# Maximum distance from the line
ranges = [300, 280, 250]  # Line 1, 2 and 3
reset_range = [-40, 40]
last_center_line = 0

lets_go = '''
   ______      __
  / ____/___  / /
 / / __/ __ \/ / 
/ /_/ / /_/ /_/  
\____/\____(_)   
'''

description = '''
 ___   ___   ____   ____             
|  _ \|  _ \|  _ \ / ___|                                          
| | \ | | \ | (_| | /  _ 
| |_/ | |_/ |  __/| \_| |                                                    
|____/|____/|_|    \____/
          ____                               
         / ___|__ _ _ __ ___   ___ _ __ __ _ 
 _____  | |   / _` | '_ ` _ \ / _ \ '__/ _` |
|_____| | |__| (_| | | | | | |  __/ | | (_| |
         \____\__,_|_| |_| |_|\___|_|  \__,_|
                                                                             
'''

title = '''
   ___     _     ______      _           _   
  |_  |   | |    | ___ \    | |         | |  
    | | __| | ___| |_/ /___ | |__   ___ | |_ 
    | |/ _` |/ _ \    // _ \| '_ \ / _ \| __|
/\__/ / (_| |  __/ |\ \ (_) | |_) | (_) | |_ 
\____/ \__,_|\___\_| \_\___/|_.__/ \___/ \__|
                                             
'''
