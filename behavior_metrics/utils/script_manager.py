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
from utils.constants import MIN_EXPERIMENT_PERCENTAGE_COMPLETED
from pilot import Pilot


    
def launch_gazebo_no_gui(current_world, stats_perfect_lap):
    environment.close_gazebo()
    tree = ET.parse(current_world)
    root = tree.getroot()
    for child in root[0]:
        if child.attrib['name'] == 'gui':
            child.attrib['value'] = 'false'
        if child.attrib['name'] == 'world_name':
            world_name = child.attrib['value']
            child.attrib['value'] = os.getcwd() + '/tmp_world.launch'

    tree.write('tmp_circuit.launch')
    tree = ET.parse(os.path.dirname(os.path.dirname(current_world)) + '/worlds/' + world_name)
    root = tree.getroot()
    
    perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag(stats_perfect_lap)
    random_index = random.randint(0,len(perfect_lap_checkpoints))
    random_point = perfect_lap_checkpoints[random_index]
    
    random_orientation = random.randint(0, 1)
    if random_orientation == 1:
        orientation_z = -random_point['pose.pose.orientation.z']
    else:
        orientation_z = random_point['pose.pose.orientation.z']
        
    random_start_point = np.array([round(random_point['pose.pose.position.x'], 3), round(random_point['pose.pose.position.y'], 3) , round(random_point['pose.pose.position.z'], 3), round(random_point['pose.pose.orientation.x'], 3), round(random_point['pose.pose.orientation.y'], 3), round(orientation_z, 3)*2.22])
    
    for child_1 in root[0]:
        if child_1.tag == 'include':
            next = False
            for child_2 in child_1:
                if next:
                    child_2.text = str(random_start_point[0]) + " " + str(random_start_point[1]) + " " + str(random_start_point[2]) + " " + str(random_start_point[3]) + " " + str(random_start_point[4]) + " " + str(random_start_point[5])
                    next = False
                elif child_2.text == 'model://f1_renault':
                    next = True
                    
    tree.write('tmp_world.launch')
    try:
        with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
            subprocess.Popen(["roslaunch", 'tmp_circuit.launch'], stdout=out, stderr=err)
            logger.info("GazeboEnv: launching gzserver.")
    except OSError as oe:
        logger.error("GazeboEnv: exception raised launching gzserver. {}".format(oe))
        environment.close_gazebo()
        sys.exit(-1)
    
    # give gazebo some time to initialize
    time.sleep(5)


def run_brains_worlds(app_configuration, controller):
    # Start Behavior Metrics app
    launch_gazebo_no_gui(app_configuration.current_world[0], app_configuration.stats_perfect_lap[0])
    pilot = Pilot(app_configuration, controller, app_configuration.brain_path[0])
    pilot.daemon = True
    controller.pilot.start()
    for world_counter, world in enumerate(app_configuration.current_world):
        for brain_counter, brain in enumerate(app_configuration.brain_path):
            repetition_counter = 0
            while repetition_counter < app_configuration.experiment_repetitions:
                # 1. Load world
                launch_gazebo_no_gui(world, app_configuration.stats_perfect_lap[world_counter])
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

                time_start = rospy.get_time()
                perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag(app_configuration.stats_perfect_lap[world_counter])
                new_point = np.array([controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().x, controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().y])

                is_finished = False
                while (rospy.get_time() - time_start < app_configuration.experiment_timeouts[world_counter] and not is_finished) or rospy.get_time() - time_start < 10:
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
                time_end = rospy.get_time()
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