import argparse
import os
import sys
import threading
import time
import rospy
import glob

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

def is_config_correct(app_configuration):
    is_correct = True
    if len(app_configuration.current_world) != len(app_configuration.experiment_timeouts):
        logger.error('Config error: Worlds number is not equal to experiment timeouts')
        is_correct = False
    if len(app_configuration.brain_path) != len(app_configuration.experiment_model):
        logger.error('Config error: Brains number is not equal to experiment models')
        is_correct = False

    return is_correct

def main():
    """Main function for the app. Handles creation and destruction of every element of the application."""

    config_data = check_args(sys.argv)
    app_configuration = Config(config_data['config'][0])
    if not config_data['script']:
        environment.launch_env(app_configuration.current_world, carla_simulator=True)
        controller = ControllerCarla()

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
        pilot.kill_event.set()
        environment.close_ros_and_simulators()
    else:
        if is_config_correct(app_configuration):
            experiments_starting_time = time.time()
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
                            logger.info("Python process finished.")

                        import re
                        import pandas as pd
                        import matplotlib.pyplot as plt
                        import matplotlib.patches as mpatches
                        from datetime import datetime

                        current_experiment_folders = []
                        root = './'
                        folders = list(os.walk(root))[1:]
                        for folder in folders:
                            if len(folder[0].split('/')) == 2 and folder[2] and experiments_starting_time < os.stat(folder[0]).st_mtime:
                                current_experiment_folders.append(folder)

                        dataframes = []
                        for folder in current_experiment_folders:
                            try:
                                r = re.compile(".*\.json")
                                json_list = list(filter(r.match, folder[2])) # Read Note below
                                df = pd.read_json(folder[0] + '/' + json_list[0], orient='index').T
                                dataframes.append(df)
                            except:
                                print('Broken experiment: ' + folder[0])
                        
                        logger.info('Experiments information: ')
                        logger.info(experiments_information)
                        logger.info('Last experiment folder: ')
                        logger.info(max(glob.glob(os.path.join('./', '*/')), key=os.path.getmtime))

            result = pd.concat(dataframes)
            result.index = result['timestamp'].values.tolist()

            experiments_starting_time_dt = datetime.fromtimestamp(experiments_starting_time)
            experiments_starting_time_str = str(experiments_starting_time_dt.strftime("%Y%m%d-%H%M%S")) + '_experiments_metrics'

            os.mkdir(experiments_starting_time_str)

            maps_colors = {
                'Carla/Maps/Town01': 'red', 
                'Carla/Maps/Town02': 'green', 
                'Carla/Maps/Town03': 'blue', 
                'Carla/Maps/Town04': 'grey', 
                'Carla/Maps/Town05': 'black', 
                'Carla/Maps/Town06': 'pink', 
                'Carla/Maps/Town07': 'orange', 
            }
            colors = []
            for i in result['carla_map']:
                colors.append(maps_colors[i])

            # COMPLETED DISTANCE
            fig = plt.figure(figsize=(20,10))
            result['completed_distance'].plot.bar(color=colors)
            plt.title('Total distance per experiment')
            fig.tight_layout()
            plt.xticks(rotation=90)

            red_patch = mpatches.Patch(color='red', label='Map01')
            green_patch = mpatches.Patch(color='green', label='Map02')
            blue_patch = mpatches.Patch(color='blue',  label='Map03')
            grey_patch = mpatches.Patch(color='grey',  label='Map04')
            black_patch = mpatches.Patch(color='black',  label='Map05')
            pink_patch = mpatches.Patch(color='pink',  label='Map06')
            orange_patch = mpatches.Patch(color='orange',  label='Map07')

            plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
            plt.savefig(experiments_starting_time_str + '/' + 'completed_distance.png')

            # AVERAGE SPEED
            fig = plt.figure(figsize=(20,10))
            result['average_speed'].plot.bar(color=colors)
            plt.title('Average speed per experiment')
            fig.tight_layout()
            plt.xticks(rotation=90)

            plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
            plt.savefig(experiments_starting_time_str + '/' + 'average_speed.png')

            # TOTAL COLLISIONS
            fig = plt.figure(figsize=(20,10))
            result['collisions'].plot.bar(color=colors)
            plt.title('Total collisions per experiment')
            fig.tight_layout()
            plt.xticks(rotation=90)

            plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
            plt.savefig(experiments_starting_time_str + '/' + 'collisions.png')

            # TOTAL LANE INVASIONS
            fig = plt.figure(figsize=(20,10))
            result['lane_invasions'].plot.bar(color=colors)
            plt.title('Total lane invasions per experiment')
            fig.tight_layout()
            plt.xticks(rotation=90)

            plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
            plt.savefig(experiments_starting_time_str + '/' + 'lane_invasions.png')

            # POSITION DEVIATION
            fig = plt.figure(figsize=(20,10))
            result['position_deviation_mae'].plot.bar(color=colors)
            plt.title('Position deviation per experiment')
            fig.tight_layout()
            plt.xticks(rotation=90)

            plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
            plt.savefig(experiments_starting_time_str + '/' + 'position_deviation_mae.png')

            # GPU inference frequency
            fig = plt.figure(figsize=(20,10))
            result['gpu_inference_frequency'].plot.bar(color=colors)
            plt.title('GPU inference frequency per experiment')
            fig.tight_layout()
            plt.xticks(rotation=90)

            plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
            plt.savefig(experiments_starting_time_str + '/' + 'gpu_inference_frequency.png')

            # BRAIN frequency
            fig = plt.figure(figsize=(20,10))
            result['brain_iterations_frequency_real_time'].plot.bar(color=colors)
            plt.title('Brain frequency per experiment')
            fig.tight_layout()
            plt.xticks(rotation=90)

            plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
            plt.savefig(experiments_starting_time_str + '/' + 'brain_iterations_frequency_real_time.png')

            unique_experiment_models = result['experiment_model'].unique()
            
            for unique_experiment_model in unique_experiment_models:
                unique_model_experiments = result.loc[result['experiment_model'].eq(unique_experiment_model)]
                
                # AVERAGE SPEED
                fig = plt.figure(figsize=(20,10))
                unique_model_experiments['average_speed'].plot.bar(color=colors)
                plt.title('Average speed per experiment')
                fig.tight_layout()
                plt.xticks(rotation=90)

                plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
                plt.savefig(experiments_starting_time_str + '/' + unique_experiment_model + '_average_speed.png')

                # TOTAL COLLISIONS
                fig = plt.figure(figsize=(20,10))
                unique_model_experiments['collisions'].plot.bar(color=colors)
                plt.title('Total collisions per experiment')
                fig.tight_layout()
                plt.xticks(rotation=90)

                plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
                plt.savefig(experiments_starting_time_str + '/' + unique_experiment_model + '_collisions.png')

                # TOTAL LANE INVASIONS
                fig = plt.figure(figsize=(20,10))
                unique_model_experiments['lane_invasions'].plot.bar(color=colors)
                fig.tight_layout()
                plt.xticks(rotation=90)

                plt.title('Total lane invasions per experiment')
                plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
                plt.savefig(experiments_starting_time_str + '/' + unique_experiment_model + '_lane_invasions.png')

                # POSITION DEVIATION
                fig = plt.figure(figsize=(20,10))
                unique_model_experiments['position_deviation_mae'].plot.bar(color=colors)
                plt.title('Position deviation per experiment')
                fig.tight_layout()
                plt.xticks(rotation=90)

                plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
                plt.savefig(experiments_starting_time_str + '/' + unique_experiment_model + '_position_deviation_mae.png')

                # GPU inference frequency
                fig = plt.figure(figsize=(20,10))
                unique_model_experiments['gpu_inference_frequency'].plot.bar(color=colors)
                plt.title('GPU inference frequency per experiment')
                fig.tight_layout()
                plt.xticks(rotation=90)

                plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
                plt.savefig(experiments_starting_time_str + '/' + unique_experiment_model + '_gpu_inference_frequency.png')

                # BRAIN frequency
                fig = plt.figure(figsize=(20,10))
                unique_model_experiments['brain_iterations_frequency_real_time'].plot.bar(color=colors)
                plt.title('Brain frequency per experiment')
                fig.tight_layout()
                plt.xticks(rotation=90)

                plt.legend(handles=[red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch])
                plt.savefig(experiments_starting_time_str + '/' + unique_experiment_model + '_brain_iterations_frequency_real_time.png')


    logger.info('DONE! Bye, bye :)')
                    

if __name__ == '__main__':
    main()
    sys.exit(0)
