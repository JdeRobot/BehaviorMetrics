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
import time
import os
import rospy
import random
import sys

import numpy as np

from utils import metrics
from utils.logger import logger
from utils.constants import MIN_EXPERIMENT_PERCENTAGE_COMPLETED, CIRCUITS_TIMEOUTS
from pilot import Pilot
from utils.tmp_world_generator import tmp_world_generator
from rosgraph_msgs.msg import Clock


def run_brains_worlds(app_configuration, controller, randomize=False):
    # Start Behavior Metrics app
    for world_counter, world in enumerate(app_configuration.current_world):
        for brain_counter, brain in enumerate(app_configuration.brain_path):
            repetition_counter = 0
            while repetition_counter < app_configuration.experiment_repetitions:
                tmp_world_generator(world, app_configuration.stats_perfect_lap[world_counter],
                                       app_configuration.real_time_update_rate, randomize=randomize, gui=False,
                                       launch=True)
                pilot = Pilot(app_configuration, controller, app_configuration.brain_path[brain_counter])
                pilot.daemon = True
                pilot.real_time_update_rate = app_configuration.real_time_update_rate
                controller.pilot.start()
                # 1. Load world
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
                controller.record_metrics(app_configuration.stats_perfect_lap[world_counter],
                                          app_configuration.stats_out,
                                          world_counter=world_counter, brain_counter=brain_counter,
                                          repetition_counter=repetition_counter)

                perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag(
                    app_configuration.stats_perfect_lap[world_counter])
                new_point = np.array([controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().x,
                                      controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().y])
                start_point = new_point
                time_start = controller.pilot.ros_clock_time
                previous_pitch = 0
                is_finished = False
                pitch_error = False
                if hasattr(app_configuration, 'experiment_timeouts'):
                    experiment_timeout = app_configuration.experiment_timeouts[world_counter]
                else:
                    experiment_timeout = CIRCUITS_TIMEOUTS[os.path.basename(world)] * 1.1
                while (controller.pilot.ros_clock_time - time_start < experiment_timeout and not is_finished) \
                        or controller.pilot.ros_clock_time - time_start < 20:
                    rospy.sleep(10)
                    old_point = new_point
                    new_point = np.array([controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().x,
                                          controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().y])
                    if is_trapped(old_point, new_point):
                        is_finished = True
                    elif metrics.is_finish_line(new_point, start_point):
                        is_finished = True
                    elif previous_pitch != 0 and abs(controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().pitch
                                                     - previous_pitch) > 0.2:
                        is_finished = True
                        pitch_error = True
                    else:
                        previous_pitch = controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().pitch
                time_end = controller.pilot.ros_clock_time
                logger.info('* Experiment end time ---> ' + str(time_end - time_start))
                controller.stop_recording_metrics(pitch_error)
                # 3. Stop
                controller.stop_pilot()
                controller.pause_gazebo_simulation()
                logger.info('* World ---> ' + world)
                logger.info('* Brain ---> ' + brain)
                if hasattr(app_configuration, 'experiment_model'):
                    logger.info('* Model ---> ' + app_configuration.experiment_model[brain_counter])
                if not pitch_error:
                    logger.info('* Metrics ---> ' + str(controller.experiment_metrics))
                repetition_counter += 1
                os.remove('tmp_circuit.launch')
                os.remove('tmp_world.launch')
                while not controller.pilot.execution_completed:
                    time.sleep(1)
    controller.stop_pilot()


def is_trapped(old_point, new_point):
    dist = (old_point - new_point) ** 2
    dist = np.sum(dist, axis=0)
    dist = np.sqrt(dist)
    if dist < 0.5:
        return True
    return False
