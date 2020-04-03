import argparse
import os
import sys

import multiprocessing


import environment

from configuration import Config
from controller import Controller
from pilot import Pilot

kill_event_gazebo = multiprocessing.Event()
process_gazebo = None

"""
    TODO: configurar el main view controlando si la configuracion viene de fichero o de gui
"""


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

def conf_window(cont, conf):
    try:

        from PyQt5.QtWidgets import QApplication
        from ui.gui.views_controller import ParentWindow, ViewsController

        app = QApplication(sys.argv)
        main_window = ParentWindow()

        views_controller = ViewsController(main_window, cont, conf)
        views_controller.show_title()

        main_window.show()

        app.exec_()
    except Exception:
        pass


def main_win(cont, conf):
    try:
        from PyQt5.QtWidgets import QApplication
        from ui.gui.views_controller import ParentWindow, ViewsController

        app = QApplication(sys.argv)
        main_window = ParentWindow()

        views_controller = ViewsController(main_window, cont, conf)
        views_controller.show_main_view(True)

        main_window.show()

        app.exec_()
    except Exception as e:
        print(e)


def main():

    # Check and generate configuration
    config_data = check_args(sys.argv)
    app_configuration = Config(config_data['config'])

    # Create controller of model-view
    controller = Controller()

    if app_configuration.empty and config_data['gui']:
        conf_window(controller, app_configuration)

    # Launch the simulation
    if app_configuration.current_world:
        print('Launching Simulation... please wait...')
        environment.launch_env(app_configuration.current_world)

    # Launch the user interface
    # if config_data['gui']:
    #     # gui_up, app = start_gui(app_configuration.empty)
    #     try:

    #         from PyQt5.QtWidgets import QApplication
    #         from ui.gui.views_controller import ParentWindow, ViewsController

    #         app = QApplication(sys.argv)
    #         main_window = ParentWindow()

    #         views_controller = ViewsController(main_window, controller, app_configuration)
    #         if False:
    #             views_controller.show_title()
    #         else:
    #             views_controller.show_main_view(True)
    #         main_window.show()
    #         gui_up = True

    #     except Exception as e:
    #         print("Could not load the GUI: {}".format(e))
    #         gui_up = False

    # if not config_data['gui'] or not gui_up:
    #     # start cui thread
    #     from ui.cui.test_npy import MyTestApp
    #     try:
    #         TA = MyTestApp()
    #         TA.run()
    #     except KeyboardInterrupt:
    #         print("Exiting... Press Esc to confirm")
    #         TA.stop()
    #         exit(0)

    # Launch control
    pilot = Pilot(app_configuration, controller)
    pilot.daemon = True
    pilot.start()
    print('Executing app')

    main_win(controller, app_configuration)
    

    print('closing all processes...')
    pilot.kill_event.set()
    print('Pilot: pilot killed.')
    environment.close_gazebo()
    print('DONE! Bye, bye :)')


if __name__ == '__main__':
    main()
    sys.exit(0)
