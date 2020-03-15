import sys
import os

import argparse
import yaml

from pilot import Pilot
from ui.cui.cui import CUI
from ui.gui.main import ExampleWindow
from PyQt5.QtWidgets import QApplication
from robot.sensors import Sensors
from ui.gui.threadGUI import ThreadGUI

class Colors:
    """
    Colors defined for improve the prints in each Stage
    """
    DEBUG = '\033[1;36;1m'
    OKCYAN = '\033[96m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def check_args(argv):

    config_data = {}

    parser = argparse.ArgumentParser(description='Neural Behaviors Suite',
                                     epilog='Enjoy the program! :)')

    parser.add_argument('-c',
                        '--config',
                        action='store',
                        type=str,
                        required=True,
                        help='{}Path to the configuration file in YML format.{}'.format(Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-g',
                        '--gui',
                        action='store_true',
                        help='{}Load the GUI. If not set, the console UI will start by default{}'.format(Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-l',
                        '--launch',
                        action='store',
                        type=str,
                        required=False,
                        help='''{}Path to the ROS launch file of the desired environment. If not set, the porgram will
                        assume that there is already a simulation or a real robot ready to rock!{}'''.format(Colors.OKBLUE, Colors.ENDC))

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        parser.error('{}No such file {} {}'.format(Colors.FAIL, args.config, Colors.ENDC))

    config_data['config'] = args.config

    if args.launch:
        if not os.path.isfile(args.launch):
            parser.error('{}No such file {} {}'.format(Colors.FAIL, args.launch, Colors.ENDC))
        config_data['launch'] = args.launch

    return config_data

def launch_env(launch_file):
    pass

def get_config_data(config_file):

    try:
        with open(config_file) as file:
            cfg = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SystemExit('{}Error: Cannot read/parse YML file. Check YAML syntax.{}'.format(Colors.FAIL, Colors.ENDC))
    return cfg


if __name__ == '__main__':

    config_data = check_args(sys.argv)

    launch_file = config_data.get('launch', None)
    if launch_file:
        launch_env(launch_file)

    config_file = config_data.get('config', None)
    configuration = get_config_data(config_file)

    ss = configuration['Behaviors']['Robot']['Sensors']
    sensors = Sensors(ss)
    # start pilot thread
    pilot = Pilot(configuration, sensors)
    pilot.daemon=True
    pilot.start()

    # start cui thread
    # c = CUI()
    # c.start()

    # start gui thread
    app = QApplication(sys.argv)
    

    ex = ExampleWindow(sensors)
    ex.show()

    t2 = ThreadGUI(ex)
    t2.daemon=True
    t2.start()

    sys.exit(app.exec_())

    # join all threads
    pilot.join()


