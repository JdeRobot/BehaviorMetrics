import argparse
import os
import sys
import threading
import time
import rospy

from pilot_carla import PilotCarla
from ui.tui.main_view import TUI
from utils import environment
from utils.colors import Colors
from utils.configuration import Config
from utils.controller_carla import ControllerCarla
from utils.logger import logger
from utils.tmp_world_generator import tmp_world_generator


def check_args(argv):
    """Function that handles argument checking and parsing.

    Arguments:
        argv {list} -- list of arguments from command line.

    Returns:
        dict -- dictionary with the detected configuration.
    """
    parser = argparse.ArgumentParser(description='Neural Behaviors Suite',
                                     epilog='Enjoy the program! :)')

    parser.add_argument('-c',
                        '--config',
                        type=str,
                        action='append',
                        required=True,
                        help='{}Path to the configuration file in YML format.{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g',
                       '--gui',
                       action='store_true',
                       help='{}Load the GUI (Graphic User Interface). Requires PyQt5 installed{}'.format(
                           Colors.OKBLUE, Colors.ENDC))

    group.add_argument('-t',
                       '--tui',
                       action='store_true',
                       help='{}Load the TUI (Terminal User Interface). Requires npyscreen installed{}'.format(
                           Colors.OKBLUE, Colors.ENDC))

    group.add_argument('-s',
                       '--script',
                       action='store_true',
                       help='{}Run Behavior Metrics as script{}'.format(
                           Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-r',
                        '--random',
                        action='store_true',
                        help='{}Run Behavior Metrics F1 with random spawning{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    args = parser.parse_args()

    config_data = {'config': None, 'gui': None, 'tui': None, 'script': None, 'random': False}
    if args.config:
        config_data['config'] = []
        for config_file in args.config:
            if not os.path.isfile(config_file):
                parser.error('{}No such file {} {}'.format(Colors.FAIL, config_file, Colors.ENDC))

        config_data['config'] = args.config

    if args.gui:
        config_data['gui'] = args.gui

    if args.tui:
        config_data['tui'] = args.tui

    if args.script:
        config_data['script'] = args.script

    if args.random:
        config_data['random'] = args.random

    return config_data

def main_win(configuration, controller):
    """shows the Qt main window of the application

    Arguments:
        configuration {Config} -- configuration instance for the application
        controller {Controller} -- controller part of the MVC model of the application
    """
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
    """Main function for the app. Handles creation and destruction of every element of the application."""

    config_data = check_args(sys.argv)
    app_configuration = Config(config_data['config'][0])
    if not config_data['script']:
        environment.launch_env(app_configuration.current_world, carla_simulator=True)
        controller = ControllerCarla()

        # Launch control
        pilot = PilotCarla(app_configuration, controller, app_configuration.brain_path)
        pilot.daemon = True
        pilot.start()
        logger.info('Executing app')
        main_win(app_configuration, controller)
        logger.info('closing all processes...')
        pilot.kill_event.set()
        environment.close_ros_and_simulators()
    else:
        for world_counter, world in enumerate(app_configuration.current_world):
            for brain_counter, brain in enumerate(app_configuration.brain_path):
                for repetition_counter in range(app_configuration.experiment_repetitions):
                    success = -1
                    experiment_attempts = 0
                    while success != 0:
                        logger.info("Launching: python3 script_manager_carla.py -c configs/default_carla_multiple.yml -s -world_counter " + str(world_counter) + " -brain_counter " + str(brain_counter) + " -repetition_counter " + str(repetition_counter))
                        success = os.system("python3 script_manager_carla.py -c configs/default_carla_multiple.yml -s -world_counter " + str(world_counter) + " -brain_counter " + str(brain_counter) + " -repetition_counter " + str(repetition_counter))
                        if success != 0:
                            root = './'
                            folders = list(os.walk(root))[1:]
                            for folder in folders:
                                if len(folder[0].split('/')) == 2 and not folder[1] and not folder[2]:
                                    logger.info("Removing empty folder: " + folder[0])
                                    os.rmdir(folder[0])
                        if success != 0 and experiment_attempts <= 5:
                            logger.info("Python process finished with error! Repeating experiment")
                        elif success != 0 and experiment_attempts > 5:
                            success = 0
                            logger.info("Too many failed attempts for this experiment.")
                        logger.info("Python process finished.")

    logger.info('DONE! Bye, bye :)')
                    

if __name__ == '__main__':
    main()
    sys.exit(0)
