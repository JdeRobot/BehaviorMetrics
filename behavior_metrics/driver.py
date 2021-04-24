#!/usr/bin/env python
""" Main module of the BehaviorMetrics application.

This module is the responsible for initializing and destroying all the elements of the application when it is launched.

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

import argparse
import os
import sys
import threading

from pilot import Pilot
from ui.tui.main_view import TUI
from utils import environment, script_manager
from utils.colors import Colors
from utils.configuration import Config
from utils.controller import Controller
from utils.logger import logger


__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


def check_args(argv):
    """Function that handles argument checking and parsing.

    Arguments:
        argv {list} -- list of arguments from command line.

    Returns:
        dict -- dictionary with the detected configuration.
    """
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

    args = parser.parse_args()

    config_data = {'config': None, 'gui': None, 'tui': None, 'script': None}
    if args.config:
        if not os.path.isfile(args.config):
            parser.error('{}No such file {} {}'.format(Colors.FAIL, args.config, Colors.ENDC))

        config_data['config'] = args.config

    if args.gui:
        config_data['gui'] = args.gui

    if args.tui:
        config_data['tui'] = args.tui
        
    if args.script:
        config_data['script'] = args.script

    return config_data


def conf_window(configuration, controller=None):
    """Gui windows for configuring the app. If not configuration file specified when launched, this windows appears,
    otherwise, main_win is called.

    Arguments:
        configuration {Config} -- configuration instance for the application

    Keyword Arguments:
        controller {Controller} -- controller part of the MVC of the application (default: {None})
    """
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

    # Check and generate configuration
    config_data = check_args(sys.argv)
    app_configuration = Config(config_data['config'])

    # Create controller of model-view
    controller = Controller()
    
    # If there's no config, configure the app through the GUI
    if app_configuration.empty and config_data['gui']:
        conf_window(app_configuration)
        
    # Launch the simulation
    if app_configuration.current_world and not config_data['script']:
        logger.debug('Launching Simulation... please wait...')
        environment.launch_env(app_configuration.current_world)

    if config_data['tui']:
        rows, columns = os.popen('stty size', 'r').read().split()
        if int(rows) < 34 and int(columns) < 115:
            logger.error("Terminal window too small: {}x{}, please resize it to at least 35x115".format(rows, columns))
            sys.exit(1)
        else:
            t = TUI(controller)
            ttui = threading.Thread(target=t.run)
            ttui.start()
          
    if not config_data['script']:
        # Launch control
        pilot = Pilot(app_configuration, controller, app_configuration.brain_path)
        pilot.daemon = True
        pilot.start()
        logger.info('Executing app')
    else:
        script_manager.run_brains_worlds(app_configuration, controller)
        logger.info('closing all processes...')
        environment.close_gazebo()
        logger.info('DONE! Bye, bye :)')
        return
        
    # If GUI specified, launch it. Otherwise don't
    if config_data['gui']:
        main_win(app_configuration, controller)
    else:
        pilot.join()

    # When window is closed or keypress for quit is detected, quit gracefully.
    logger.info('closing all processes...')
    pilot.kill_event.set()
    environment.close_gazebo()
    logger.info('DONE! Bye, bye :)')


if __name__ == '__main__':
    main()
    sys.exit(0)
