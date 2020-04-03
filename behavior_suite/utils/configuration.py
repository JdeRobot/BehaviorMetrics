import yaml
from utils.colors import Colors


class Config:

    def __init__(self, config_file):

        self.empty = True
        if config_file:
            config_data = self.get_config_data(config_file)

            if config_data:
                self.initialize_configuration(config_data)
                self.empty = False
        else:
            self.initialize_empty_configuration()
            self.empty = True

    def get_config_data(self, config_file):

        try:
            with open(config_file) as file:
                cfg = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise SystemExit('{}Error: Cannot read/parse YML file. Check YAML syntax: {}.{}'.format(
                Colors.FAIL, e, Colors.ENDC))
        return cfg

    def initialize_empty_configuration(self):
        self.brain_path = None
        self.robot_type = None

        self.actuators = None
        self.sensors = None

        self.current_world = None
        self.layout = {}

        self.dataset_in = None
        self.dataset_out = None

    def initialize_configuration(self, config_data):
        robot = config_data['Behaviors']['Robot']
        self.brain_path = robot['BrainPath']
        self.robot_type = robot['Type']
        self.current_world = config_data['Behaviors']['Simulation']['World']

        self.actuators = robot['Actuators']
        self.sensors = robot['Sensors']

        self.layout = self.create_layout_from_cfg(config_data['Behaviors']['Layout'])

        self.dataset_in = config_data['Behaviors']['Dataset']['In']
        self.dataset_out = config_data['Behaviors']['Dataset']['Out']

    def create_layout_from_cfg(self, cfg):
        layout = {}
        for frame in cfg:
            layout[cfg[frame]['Name']] = (cfg[frame]['Geometry'], cfg[frame]['Data'])

        return layout

    def create_layout_from_gui(self, cfg):
        for frame in cfg:
            name = 'frame_' + str(frame[-1])
            self.layout[name] = (frame[:-1], 'rgbimage')  # default datatype

    def robot_type_set(self, robot_type):
        self.robot_type = robot_type
        self.create_sensors_actuators(robot_type)

    def create_sensors_actuators(self, robot_type):

        robot_config_path = '/home/fran/github/BehaviorSuite/behavior_suite/robot/configurations/'
        with open(robot_config_path + robot_type + '.yml') as file:
            cfg = yaml.safe_load(file)
        robot = cfg['Robot']
        self.sensors = robot['Sensors']
        self.actuators = robot['Actuators']

    def change_frame_name(self, old, new):
        self.layout[new] = self.layout.pop(old)
