import argparse
import os
import sys

from pilot import Pilot
from utils import environment
from utils.configuration import Config
from utils.controller import Controller
from utils.colors import Colors
from utils.logger import logger


def check_args(argv):

    config_data = {}

    parser = argparse.ArgumentParser(description='Neural Behaviors Suite',
                                     epilog='Enjoy the program! :)')

    parser.add_argument('-c',
                        '--config',
                        action='store',
                        type=str,
                        required=False,
                        help='{}Path to the configuration file in YML format.{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-g',
                        '--gui',
                        action='store_true',
                        help='{}Load the GUI. If not set, the console UI will start by default{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    args = parser.parse_args()

    config_data = {'config': None, 'gui': None}
    if args.config:
        config_data = {}
        if not os.path.isfile(args.config):
            parser.error('{}No such file {} {}'.format(Colors.FAIL, args.config, Colors.ENDC))

        config_data['config'] = args.config

    if args.gui:
        config_data['gui'] = args.gui

    return config_data


def conf_window(configuration, controller=None):
    try:

        from PyQt5.QtWidgets import QApplication
        from ui.gui.views_controller import ParentWindow, ViewsController

        app = QApplication(sys.argv)
        main_window = ParentWindow()

        views_controller = ViewsController(main_window, configuration)
        views_controller.show_title()

        main_window.show()

        app.exec_()
    except Exception:
        pass


def main_win(configuration, controller):
    try:
        from PyQt5.QtWidgets import QApplication
        from ui.gui.views_controller import ParentWindow, ViewsController

        app = QApplication(sys.argv)
        main_window = ParentWindow()

        views_controller = ViewsController(main_window, configuration, controller)
        views_controller.show_main_view(True)

        main_window.show()

        app.exec_()
    except Exception as e:
        logger.error(e)


def main():

    # Check and generate configuration
    config_data = check_args(sys.argv)
    app_configuration = Config(config_data['config'])

    # If there's no config, configure the app through the GUI
    if app_configuration.empty and config_data['gui']:
        conf_window(app_configuration)

    # Launch the simulation
    if app_configuration.current_world:
        logger.debug('Launching Simulation... please wait...')
        environment.launch_env(app_configuration.current_world)

    # Create controller of model-view
    controller = Controller()

    # Launch control
    pilot = Pilot(app_configuration, controller)
    pilot.daemon = True
    pilot.start()
    logger.info('Executing app')

    main_win(app_configuration, controller)

    logger.info('closing all processes...')
    pilot.kill_event.set()
    logger.debug('Pilot: pilot killed.')
    environment.close_gazebo()
    logger.info('DONE! Bye, bye :)')


if __name__ == '__main__':
    main()
    sys.exit(0)
