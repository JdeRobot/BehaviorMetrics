from robot.sensors import Sensors
from robot.actuators import Actuators
import yaml

class Config:

    def __init__(self, config_data):

        if config_data:

            robot = config_data['Behaviors']['Robot']
            sensors_config = robot['Sensors']
            actuators_config = robot['Actuators']
            self.brain_path = robot['BrainPath']
            self.robot_type = robot['Type']

            self.actuators = Actuators(actuators_config)
            self.sensors = Sensors(sensors_config)
            
            self.current_world = config_data['Behaviors']['Simulation']['World']
            self.layout = self.create_layout_from_cfg(config_data['Behaviors']['Layout'])

            self.dataset_in = config_data['Behaviors']['Dataset']['In']
            self.dataset_out = config_data['Behaviors']['Dataset']['Out']
        else:
            self.brain_path = None
            self.robot_type = None

            self.actuators = None
            self.sensors = None
            
            self.current_world = None
            self.layout = {}

            self.dataset_in = None
            self.dataset_out = None


    def create_layout_from_cfg(self, cfg):
        layout = {}
        for frame in cfg:
            layout[cfg[frame]['Name']] = (cfg[frame]['Geometry'], cfg[frame]['Data'])
        
        return layout
    
    def create_layout_from_gui(self, cfg):
        for frame in cfg:
            name = 'frame_' + str(frame[-1])
            self.layout[name] = (frame[:-1], 'rgbimage') # default datatype

    def robot_type_set(self, robot_type):
        self.robot_type = robot_type
        self.create_sensors_actuators(robot_type)

    def create_sensors_actuators(self, robot_type):

        robot_config_path = '/home/fran/github/BehaviorSuite/behavior_suite/robot/configurations/'
        with open(robot_config_path + robot_type + '.yml') as file:
            cfg = yaml.safe_load(file)
        robot = cfg['Robot']
        sensors_config = robot['Sensors']
        actuators_config = robot['Actuators']
        # self.actuators = Actuators(actuators_config)
        # self.sensors = Sensors(sensors_config)
        
    def change_frame_name(self, old, new):
        self.layout[new] = self.layout.pop(old)
    

