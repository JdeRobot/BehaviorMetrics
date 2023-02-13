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
from utils.constants import CARLA_TOWNS_TIMEOUTS

def check_args(argv):
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

    parser.add_argument('-world_counter',
                        type=str,
                        action='append',
                        help='{}World counter{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    parser.add_argument('-brain_counter',
                        type=str,
                        action='append',
                        help='{}Brain counter{}'.format(
                            Colors.OKBLUE, Colors.ENDC))
    
    parser.add_argument('-repetition_counter',
                        type=str,
                        action='append',
                        help='{}Repetition counter{}'.format(
                            Colors.OKBLUE, Colors.ENDC))

    args = parser.parse_args()

    config_data = {'config': None, 'gui': None, 'tui': None, 'script': None, 'random': False, 'world_counter': 0, 'brain_counter': 0, 'repetition_counter': 0}
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

    if args.world_counter:
        config_data['world_counter'] = args.world_counter
    
    if args.brain_counter:
        config_data['brain_counter'] = args.brain_counter

    if args.repetition_counter:
        config_data['repetition_counter'] = args.repetition_counter

    return config_data

def main():
    config_data = check_args(sys.argv)
    app_configuration = Config(config_data['config'][0])

    world_counter = int(config_data['world_counter'][0])
    brain_counter = int(config_data['brain_counter'][0])
    repetition_counter = int(config_data['repetition_counter'][0])

    logger.info(str(world_counter) + ' ' + str(brain_counter) + ' ' + str(repetition_counter))

    world = app_configuration.current_world[world_counter]
    brain = app_configuration.brain_path[brain_counter]
    experiment_model = app_configuration.experiment_model[brain_counter]

    if app_configuration.spawn_points:
        spawn_point = app_configuration.spawn_points[world_counter][repetition_counter]
        environment.launch_env(world, random_spawn_point=app_configuration.experiment_random_spawn_point, carla_simulator=True, config_spawn_point=app_configuration.spawn_points[world_counter][repetition_counter])
    else:
        environment.launch_env(world, random_spawn_point=app_configuration.experiment_random_spawn_point, carla_simulator=True)
    controller = ControllerCarla()

    # Launch control
    pilot = PilotCarla(app_configuration, controller, brain, experiment_model=experiment_model)
    pilot.daemon = True
    pilot.start()
    logger.info('Executing app')
    controller.resume_pilot()
    controller.unpause_carla_simulation()
    controller.record_metrics(app_configuration.stats_out, world_counter=world_counter, brain_counter=brain_counter, repetition_counter=repetition_counter)
    if app_configuration.use_world_timeouts:
        experiment_timeout = CARLA_TOWNS_TIMEOUTS[controller.carla_map.name]
    else:
        experiment_timeout = app_configuration.experiment_timeouts[world_counter]

    rospy.sleep(experiment_timeout)
    controller.stop_recording_metrics()
    controller.pilot.stop()
    controller.stop_pilot()
    controller.pause_carla_simulation()

    logger.info('closing all processes...')
    controller.pilot.kill()
    environment.close_ros_and_simulators()
    while not controller.pilot.execution_completed:
        time.sleep(1)


if __name__ == '__main__':
    main()
    sys.exit(0)