#!/usr/bin/env python

"""This module contains the script manager.

This module is in charge of running the application as an script, without GUI. 
It is used for experiments for different brains/worlds

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


import subprocess
import xml.etree.ElementTree as ET
import time
import os
import rospy
import random
import sys

import numpy as np

from utils import metrics
from utils import environment
from utils.logger import logger
from utils.constants import MIN_EXPERIMENT_PERCENTAGE_COMPLETED, CIRCUITS_TIMEOUTS
from pilot import Pilot
from utils.random_initializer import tmp_random_initializer
from rosgraph_msgs.msg import Clock

clock_time = None

def clock_callback(clock_data):
    global clock_time
    clock_time = clock_data.clock.to_sec()

def run_brains_worlds(app_configuration, controller, randomize=False):
    global clock_time
    # Start Behavior Metrics app
    tmp_random_initializer(app_configuration.current_world[0], app_configuration.stats_perfect_lap[0], randomize=randomize, gui=False, launch=True)
    pilot = Pilot(app_configuration, controller, app_configuration.brain_path[0])
    pilot.daemon = True
    controller.pilot.start()
    for world_counter, world in enumerate(app_configuration.current_world):
        import os
        for brain_counter, brain in enumerate(app_configuration.brain_path):
            repetition_counter = 0
            while repetition_counter < app_configuration.experiment_repetitions:
                # 1. Load world
                tmp_random_initializer(world, app_configuration.stats_perfect_lap[world_counter], gui=False, launch=True)
                controller.initialize_robot()
                controller.pilot.configuration.current_world = world
                controller.pilot.brains.brain_path = brain
                logger.info('Executing brain')
                # 2. Play
                if hasattr(app_configuration, 'experiment_model'):
                    controller.reload_brain(brain, model=app_configuration.experiment_model[brain_counter])
                else:
                    controller.reload_brain(brain)
                controller.resume_pilot()
                controller.unpause_gazebo_simulation()
                controller.record_stats(app_configuration.stats_perfect_lap[world_counter], app_configuration.stats_out, world_counter=world_counter, brain_counter=brain_counter, repetition_counter=repetition_counter)
                
                clock_subscriber = rospy.Subscriber("/clock", Clock, clock_callback)
                perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag(app_configuration.stats_perfect_lap[world_counter])
                new_point = np.array([controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().x, controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().y])
                time_start = clock_time
                
                is_finished = False
                if hasattr(app_configuration, 'experiment_timeouts'):
                    experiment_timeout = app_configuration.experiment_timeouts[world_counter]
                else:
                    experiment_timeout = CIRCUITS_TIMEOUTS[os.path.basename(world)]  * 1.1
                while (clock_time - time_start < experiment_timeout and not is_finished) or clock_time - time_start < 10:
                    rospy.sleep(10)
                    old_point = new_point
                    new_point = np.array([controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().x, controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().y])
                    if is_trapped(old_point, new_point):
                        is_finished = True
                    if metrics.is_finish_line(new_point, perfect_lap_checkpoints[0]):
                        is_finished = True

                logger.info('--------------')
                logger.info('--------------')
                logger.info('--------------')
                logger.info('--------------')
                logger.info('--- END TIME ----------------')
                time_end = clock_time
                clock_subscriber.unregister()
                logger.info(time_end - time_start)
                controller.stop_record_stats()
                # 3. Stop
                controller.pause_pilot()
                controller.pause_gazebo_simulation()
                logger.info('--- WORLD ---')
                logger.info(world)
                logger.info('--- BRAIN ---')
                logger.info(brain)
                if hasattr(app_configuration, 'experiment_model'):
                    logger.info('--- MODEL ---')
                    logger.info(app_configuration.experiment_model[brain_counter])
                logger.info('--- STATS ---')
                logger.info(controller.lap_statistics)
                if controller.lap_statistics['percentage_completed'] < MIN_EXPERIMENT_PERCENTAGE_COMPLETED:
                    logger.info('--- DELETE STATS and RETRY EXPERIMENT ---')
                    os.remove(controller.stats_filename)
                else:
                    repetition_counter += 1
                logger.info('--------------')
                logger.info('--------------')
                logger.info('--------------')
                logger.info('--------------')
        os.remove('tmp_circuit.launch')
        os.remove('tmp_world.launch')
        
        
def is_trapped(old_point, new_point):
    dist = (old_point - new_point) ** 2
    dist = np.sum(dist, axis=0)
    dist = np.sqrt(dist)
    if dist < 0.5:
        return True
    return False