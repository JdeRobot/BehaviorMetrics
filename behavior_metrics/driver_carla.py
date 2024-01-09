import argparse
import os
import sys
import threading
import time
import rospy
import glob
import json


from ui.tui.main_view import TUI
from utils import environment
from utils.traffic import TrafficManager
from utils.colors import Colors
from utils.configuration import Config
from utils.controller_carla import ControllerCarla
from utils.logger import logger
from utils.tmp_world_generator import tmp_world_generator
from utils import metrics_carla
from datetime import datetime
from pilot_carla import PilotCarla

import matplotlib.pyplot as plt
import pandas as pd

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

def is_config_correct(app_configuration):
    is_correct = True
    if len(app_configuration.current_world) != len(app_configuration.experiment_timeouts):
        logger.error('Config error: Worlds number is not equal to experiment timeouts')
        is_correct = False
    if len(app_configuration.brain_path) != len(app_configuration.experiment_model):
        logger.error('Config error: Brains number is not equal to experiment models')
        is_correct = False

    return is_correct

def generate_agregated_experiments_metrics(experiments_starting_time, experiments_elapsed_times, app_configuration):
    result = metrics_carla.get_aggregated_experiments_list(experiments_starting_time)

    experiments_starting_time_dt = datetime.fromtimestamp(experiments_starting_time)
    experiments_starting_time_str = str(experiments_starting_time_dt.strftime("%Y%m%d-%H%M%S")) + '_experiments_metrics'

    os.mkdir(experiments_starting_time_str)

    experiments_metrics_and_titles = [
        {
            'metric': 'experiment_total_simulated_time',
            'title': 'Experiment total simulated time per experiment'
        },
        {
            'metric': 'position_deviation_total_err',
            'title': 'Position deviation total error per experiment'
        },
        {
            'metric': 'effective_completed_distance',
            'title': 'Effective completed distance per experiment'
        },
        {
            'metric': 'experiment_total_real_time',
            'title': 'Experiment total real time per experiment'
        },
        {
            'metric': 'suddenness_distance_control_commands',
            'title': 'Suddennes distance control commands per experiment'
        },
        {
            'metric': 'suddenness_distance_throttle',
            'title': 'Suddennes distance throttle per experiment'
        },
        {
            'metric': 'suddenness_distance_steer',
            'title': 'Suddennes distance steer per experiment'
        },
        {
            'metric': 'suddenness_distance_brake_command',
            'title': 'Suddennes distance brake per experiment'
        },
        {
            'metric': 'suddenness_distance_control_command_per_km',
            'title': 'Suddennes distance control commands per km per experiment'
        },
        {
            'metric': 'suddenness_distance_throttle_per_km',
            'title': 'Suddennes distance throttle per km per experiment'
        },
        {
            'metric': 'suddenness_distance_steer_per_km',
            'title': 'Suddennes distance steer per km per experiment'
        },
        {
            'metric': 'suddenness_distance_brake_command_per_km',
            'title': 'Suddennes distance brake per km per experiment'
        },
        {
            'metric': 'brain_iterations_frequency_simulated_time',
            'title': 'Brain itertions frequency simulated time per experiment'
        },
        {
            'metric': 'mean_brain_iterations_simulated_time',
            'title': 'Mean brain iterations simulated time per experiment'
        }, 
        {
            'metric': 'gpu_mean_inference_time',
            'title': 'GPU mean inference time per experiment'
        }, 
        {
            'metric': 'mean_brain_iterations_real_time',
            'title': 'Mean brain iterations real time per experiment'
        },
        {
            'metric': 'target_brain_iterations_real_time',
            'title': 'Target brain iterations real time per experiment'
        },
        {
            'metric': 'completed_distance',
            'title': 'Total distance per experiment'
        }, 
        {
            'metric': 'average_speed',
            'title': 'Average speed per experiment'
        },
        {
            'metric': 'collisions',
            'title': 'Total collisions per experiment'
        },
        {
            'metric': 'lane_invasions',
            'title': 'Total lane invasions per experiment'
        },
        {
            'metric': 'position_deviation_mean',
            'title': 'Mean position deviation per experiment'
        },
        {
            'metric': 'position_deviation_mean_per_km',
            'title': 'Mean position deviation per km per experiment'
        },
        {
            'metric': 'gpu_inference_frequency',
            'title': 'GPU inference frequency per experiment'
        },
        {
            'metric': 'brain_iterations_frequency_real_time',
            'title': 'Brain frequency per experiment'
        },
        {
            'metric': 'collisions_per_km',
            'title': 'Collisions per km per experiment'
        },
        {
            'metric': 'lane_invasions_per_km',
            'title': 'Lane invasions per experiment'
        },
        {
            'metric': 'suddenness_distance_speed',
            'title': 'Suddenness distance speed per experiment'
        },
        {
            'metric': 'suddenness_distance_speed_per_km',
            'title': 'Suddenness distance speed per km per experiment'
        },
        
        {
            'metric': 'completed_laps',
            'title': 'Completed laps per experiment'
        },
    ]

    if app_configuration.task == 'follow_lane_traffic':
        experiments_metrics_and_titles.append(
            {
                'metric': 'dangerous_distance_pct_km',
                'title': 'Percentage of dangerous distance per km'
            },
            {
                'metric': 'close_distance_pct_km',
                'title': 'Percentage of close distance per km'
            },
            {
                'metric': 'medium_distance_pct_km',
                'title': 'Percentage of medium distance per km'
            },
            {
                'metric': 'great_distance_pct_km',
                'title': 'Percentage of great distance per km'
            },
        )

    metrics_carla.get_all_experiments_aggregated_metrics(result, experiments_starting_time_str, experiments_metrics_and_titles)
    metrics_carla.get_per_model_aggregated_metrics(result, experiments_starting_time_str, experiments_metrics_and_titles)
    metrics_carla.get_all_experiments_aggregated_metrics_boxplot(result, experiments_starting_time_str, experiments_metrics_and_titles)

    with open(experiments_starting_time_str + '/' + 'experiment_elapsed_times.json', 'w') as f:
        json.dump(experiments_elapsed_times, f)

    df = pd.DataFrame(experiments_elapsed_times)
    fig = plt.figure(figsize=(20,10))
    df['elapsed_time'].plot.bar()
    plt.title('Experiments elapsed time || Experiments total time: ' + str(experiments_elapsed_times['total_experiments_elapsed_time']) + ' secs.')
    fig.tight_layout()
    plt.xticks(rotation=90)
    plt.savefig(experiments_starting_time_str + '/' + 'experiment_elapsed_times.png')
    plt.close()

def main():
    """Main function for the app. Handles creation and destruction of every element of the application."""

    config_data = check_args(sys.argv)
    app_configuration = Config(config_data['config'][0])
    if not config_data['script']:
        if app_configuration.task not in ['follow_lane', 'follow_lane_traffic']:
            logger.info('Selected task does not support --gui. Try use --script instead. Killing program...')
            sys.exit(-1)
        environment.launch_env(app_configuration.current_world, random_spawn_point=app_configuration.experiment_random_spawn_point, carla_simulator=True)
        controller = ControllerCarla()
        traffic_manager = TrafficManager(app_configuration.number_of_vehicle, 
                                         app_configuration.number_of_walker, 
                                         app_configuration.percentage_walker_running, 
                                         app_configuration.percentage_walker_crossing,
                                         app_configuration.async_mode)
        traffic_manager.generate_traffic()

        # Launch control
        if hasattr(app_configuration, 'experiment_model'):
            experiment_model = app_configuration.experiment_model
            pilot = PilotCarla(app_configuration, controller, app_configuration.brain_path, experiment_model=experiment_model)
        else:
            pilot = PilotCarla(app_configuration, controller, app_configuration.brain_path)
        pilot.daemon = True
        pilot.start()
        logger.info('Executing app')
        main_win(app_configuration, controller)
        logger.info('closing all processes...')
        traffic_manager.destroy()
        pilot.kill_event.set()
        environment.close_ros_and_simulators()
    else:
        if is_config_correct(app_configuration):
            if app_configuration.task == 'follow_lane':
                experiments_starting_time = time.time()
                experiment_counter = 0
                experiments_elapsed_times = {'experiment_counter': [], 'elapsed_time': []}
                experiments_information = {'world_counter': {}}
                for world_counter, world in enumerate(app_configuration.current_world):
                    experiments_information['world_counter'][world_counter] = {'brain_counter': {}}
                    for brain_counter, brain in enumerate(app_configuration.brain_path):
                        experiments_information['world_counter'][world_counter]['brain_counter'][brain_counter] = {'repetition_counter': {}}
                        for repetition_counter in range(app_configuration.experiment_repetitions):
                            success = -1
                            experiment_attempts = 0
                            while success != 0:                    
                                experiments_information['world_counter'][world_counter]['brain_counter'][brain_counter]['repetition_counter'][repetition_counter] = experiment_attempts
                                logger.info("Launching: python3 script_manager_carla.py -c " + config_data['config'][0] + " -s -world_counter " + str(world_counter) + " -brain_counter " + str(brain_counter) + " -repetition_counter " + str(repetition_counter))
                                logger.info("Experiment attempt: " + str(experiment_attempts+1))
                                current_experiment_starting_time = time.time()
                                success = os.system("python3 script_manager_carla.py -c " + config_data['config'][0] + " -s -world_counter " + str(world_counter) + " -brain_counter " + str(brain_counter) + " -repetition_counter " + str(repetition_counter))
                                if success != 0:
                                    root = './'
                                    folders = list(os.walk(root))[1:]
                                    for folder in folders:
                                        if len(folder[0].split('/')) == 2 and not folder[1] and not folder[2]:
                                            logger.info("Removing empty folder: " + folder[0])
                                            os.rmdir(folder[0])
                                if success == 2:
                                    logger.info('KeyboardInterrupt called! Killing program...')
                                    sys.exit(-1)
                                elif success != 0 and experiment_attempts < 5:
                                    experiment_attempts += 1
                                    logger.info("Python process finished with error! Repeating experiment")
                                elif success != 0 and experiment_attempts >= 5:
                                    success = 0
                                    logger.info("Too many failed attempts for this experiment.")
                                else:
                                    experiments_elapsed_times['experiment_counter'].append(experiment_counter)
                                    experiments_elapsed_times['elapsed_time'].append(time.time() - current_experiment_starting_time)
                                    experiment_counter += 1
                                logger.info("Python process finished.")

                            logger.info('Experiments information: ')
                            logger.info(experiments_information)
                            logger.info('Last experiment folder: ')
                            logger.info(max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime))
            elif app_configuration.task == 'follow_route':

                experiments_starting_time = time.time()
                experiment_counter = 0
                experiments_elapsed_times = {'experiment_counter': [], 'elapsed_time': []}
                
                for world_counter, world in enumerate(app_configuration.current_world):
                    for brain_counter, brain in enumerate(app_configuration.brain_path):
                        for route_counter in range(app_configuration.num_routes):
                            success = -1
                            experiment_attempts = 0
                            while success != 0:
                                logger.info("Launching: python3 test_suite_manager_carla.py -c " + config_data['config'][0])
                                logger.info("Experiment attempt: " + str(experiment_attempts+1))
                                current_experiment_starting_time = time.time()
                                success = os.system("python3 test_suite_manager_carla.py -c " + config_data['config'][0] + " -s -world_counter " + str(world_counter) + " -brain_counter " + str(brain_counter) + " -route_counter " + str(route_counter))
                                print('success: ', success)
                                if success != 0:
                                    root = './'
                                    folders = list(os.walk(root))[1:]
                                    for folder in folders:
                                        if len(folder[0].split('/')) == 2 and not folder[1] and not folder[2]:
                                            logger.info("Removing empty folder: " + folder[0])
                                            os.rmdir(folder[0])
                                if success == 2:
                                    logger.info('KeyboardInterrupt called! Killing program...')
                                    sys.exit(-1)
                                elif success != 0 and experiment_attempts < 5:
                                    experiment_attempts += 1
                                    logger.info("Python process finished with error! Repeating experiment")
                                elif success != 0 and experiment_attempts >= 5:
                                    success = 0
                                    logger.info("Too many failed attempts for this experiment.")
                                else:
                                    experiments_elapsed_times['experiment_counter'].append(experiment_counter)
                                    experiments_elapsed_times['elapsed_time'].append(time.time() - current_experiment_starting_time)
                                    experiment_counter += 1
                                logger.info("Python process finished.")
                            
                            logger.info('Last experiment folder: ')
                            logger.info(max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime))
            else:
                logger.info('Invalid task type. Try "follow_route", "follow_lane" or "follow_lane_traffic". Killing program...')
                sys.exit(-1)
            experiments_elapsed_times['total_experiments_elapsed_time'] = time.time() - experiments_starting_time
            generate_agregated_experiments_metrics(experiments_starting_time, experiments_elapsed_times, app_configuration)
    if app_configuration.experiment_random_spawn_point == True or app_configuration.task == 'follow_route':
        if os.path.isfile('tmp_circuit.launch'):
            os.remove('tmp_circuit.launch')
    logger.info('DONE! Bye, bye :)')
                    

if __name__ == '__main__':
    main()
    sys.exit(0)
