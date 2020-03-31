import argparse
import os
import sys

import yaml
from PyQt5.QtWidgets import QApplication

from controller import Controller
from pilot import Pilot
from ui.cui.cui import CUI
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
                        help='{}Path to the configuration file in YML format.{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-g',
                        '--gui',
                        action='store_true',
                        help='{}Load the GUI. If not set, the console UI will start by default{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-l',
                        '--launch',
                        action='store',
                        type=str,
                        required=False,
                        help='''{}Path to the ROS launch file of the desired environment. If not set, the porgram will
                        assume that there is already a simulation or a real robot ready to rock!{}'''.format(
                            Colors.OKBLUE, Colors.ENDC))

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
    except yaml.YAMLError as e:
        raise SystemExit('{}Error: Cannot read/parse YML file. Check YAML syntax: {}.{}'.format(
            Colors.FAIL, e, Colors.ENDC))
    return cfg


if __name__ == '__main__':

    config_data = check_args(sys.argv)

    launch_file = config_data.get('launch', None)
    if launch_file:
        launch_env(launch_file)

    config_file = config_data.get('config', None)
    configuration = get_config_data(config_file)

    controller = Controller()

    # start cui thread
    # from ui.cui.test_npy import MyTestApp
    # try:
    #     TA = MyTestApp()
    #     TA.run()
    # except KeyboardInterrupt:
    #     print("Exiting... Press Esc to confirm")
    #     TA.stop()
    #     exit(0)

    # start gui thread
    from ui.gui.views_controller import ParentWindow, ViewsController
    app = QApplication(sys.argv)

    main_window = ParentWindow()
    main_window.show()

    views_controller = ViewsController(main_window, controller)
    # views_controller.show_title()
    views_controller.show_layout_selection()

    # t2 = ThreadGUI(views_controller)
    # t2.daemon = True
    # t2.start()

    # start pilot thread
    pilot = Pilot(configuration, controller)
    pilot.daemon = True
    pilot.start()

    sys.exit(app.exec_())

    # join all threads
    pilot.join()
