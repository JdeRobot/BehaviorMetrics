#!/usr/bin/env python

"""This module contains all the configuration data of the application.

This module is capable of create configuration profiles and store them as well as load data from yaml configuration
files specified at launch time.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import yaml

from utils.colors import Colors
from utils.constants import ROOT_PATH

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class Config:
    """This class handles all the configuration of the application

    It's able to load and save configuration profiles in yaml format.

    Attributes:
        empty {bool} -- Flag to determine if a configuration file was passed to the application
        congif_data {dict} -- Configuration read from the YAML file.
    """

    def __init__(self, config_file):
        """Constructor of the class

        Arguments:
            config_file {str} -- YAML configuration file path
        """

        self.empty = True
        if config_file:
            config_data = self.__get_config_data(config_file)

            if config_data:
                self.initialize_configuration(config_data)
                self.empty = False
        else:
            self.initialize_empty_configuration()
            self.empty = True

    def __get_config_data(self, config_file):
        """ Read YAML file into python dict"""

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
        
        self.stats_out = None
        
        self.experiment_timeouts = None

    def initialize_configuration(self, config_data):
        """Initialize the configuration of the application based on a YAML profile file

        Arguments:
            config_data {dict} -- Configuration data read from the YAML file
        """
        robot = config_data['Behaviors']['Robot']
        self.brain_path = robot['BrainPath']
        self.robot_type = robot['Type']
        self.current_world = config_data['Behaviors']['Simulation']['World']

        self.actuators = robot['Actuators']
        self.sensors = robot['Sensors']

        self.layout = self.create_layout_from_cfg(config_data['Behaviors']['Layout'])

        self.dataset_in = config_data['Behaviors']['Dataset']['In']
        self.dataset_out = config_data['Behaviors']['Dataset']['Out']
        
        self.stats_out = config_data['Behaviors']['Stats']['Out']
        self.stats_perfect_lap = config_data['Behaviors']['Stats']['PerfectLap']
        
        if 'Model' in config_data['Behaviors']['Robot']:
            self.experiment_model = config_data['Behaviors']['Robot']['Model']
                
        if 'Experiment' in config_data['Behaviors']:
            self.experiment_name = config_data['Behaviors']['Experiment']['Name']
            self.experiment_description = config_data['Behaviors']['Experiment']['Description']
            self.experiment_timeouts = config_data['Behaviors']['Experiment']['Timeout']
            self.experiment_repetitions = config_data['Behaviors']['Experiment']['Repetitions']
        
        if self.robot_type == 'f1rl':
            self.action_set = robot['Parameters']['action_set']
            self.gazebo_positions_set = robot['Parameters']['gazebo_positions_set']
            self.alpha = robot['Parameters']['alpha']
            self.gamma = robot['Parameters']['gamma']
            self.epsilon = robot['Parameters']['epsilon']
            self.total_episodes = robot['Parameters']['total_episodes']
            self.epsilon_discount = robot['Parameters']['epsilon_discount']
            self.env = robot['Parameters']['env']

    def create_layout_from_cfg(self, cfg):
        """Creates the configuration for the layout of the sensors view panels specified from configuration file

        Arguments:
            cfg {list} -- List of positions for each sensor panel read from configuration file

        Returns:
            dict -- configuration info transformed in a dictionary
        """

        layout = {}
        for frame in cfg:
            layout[cfg[frame]['Name']] = (cfg[frame]['Geometry'], cfg[frame]['Data'])

        return layout

    def create_layout_from_gui(self, cfg):
        """Creates the configuration for the layout of the sensor view panels specified from GUI

        Arguments:
            cfg {list} -- List of positions for each sensor panels obtained from GUI.
        """
        for frame in cfg:
            name = 'frame_' + str(frame[-1])
            self.layout[name] = (frame[:-1], 'rgbimage')  # default datatype

    def robot_type_set(self, robot_type):
        """Set the type of robot that will be used.

        Arguments:
            robot_type {str} -- Type of supported robot (f1, drone, turtlebot, ...)
        """
        self.robot_type = robot_type
        self.create_sensors_actuators(robot_type)

    def create_sensors_actuators(self, robot_type):
        """Create configuration for an specific robot selected from GUI.

        Arguments:
            robot_type {str} -- Type of supported robot (f1, drone, turtlebot, ...)
        """

        robot_config_path = ROOT_PATH + '/robot/configurations/' + robot_type + '.yml'
        cfg = self.__get_config_data(robot_config_path)
        robot = cfg['Robot']
        self.sensors = robot['Sensors']
        self.actuators = robot['Actuators']

    def change_frame_name(self, old, new):
        """Change the identificator of the frame in the GUI.

        Arguments:
            old {str} -- Old name of the frame
            new {str} -- New name for the frame
        """
        self.layout[new] = self.layout.pop(old)
