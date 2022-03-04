#!/usr/bin/env python

"""This module contains the random initializer.

This module is used if you want to use random initialization of your robot
in the GUI mode. It creates a tmp launch file with the same configuration
as the original one but with a different initial position of the robot.

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
import math

import numpy as np

from utils import metrics
from utils import environment
from utils.logger import logger


def tmp_world_generator(current_world, stats_perfect_lap, real_time_update_rate, randomize=False, gui=False,
                           launch=False):
    environment.close_gazebo()
    tree = ET.parse(current_world)
    root = tree.getroot()
    for child in root[0]:
        if child.attrib['name'] == 'gui':
            if gui:
                child.attrib['value'] = 'true'
            else:
                child.attrib['value'] = 'false'
        if child.attrib['name'] == 'world_name':
            world_name = child.attrib['value']
            child.attrib['value'] = os.getcwd() + '/tmp_world.launch'

    tree.write('tmp_circuit.launch')
    tree = ET.parse(os.path.dirname(os.path.dirname(current_world)) + '/worlds/' + world_name)
    root = tree.getroot()

    perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag(stats_perfect_lap)
    check_point_offset = 5
    if randomize:
        random_index = random.randint(0, int(len(perfect_lap_checkpoints)- check_point_offset - 1))
        try:
            random_point = perfect_lap_checkpoints[random_index]
        except Exception as exc:
            print("Error to review {}".format(exc))
            random_index = 2
            random_point = perfect_lap_checkpoints[random_index]


        p1 = perfect_lap_checkpoints[random_index]
        p2 = perfect_lap_checkpoints[random_index + check_point_offset]
        delta_y = p2['pose.pose.position.y'] - p1['pose.pose.position.y']
        delta_x = p2['pose.pose.position.x'] - p1['pose.pose.position.x']
        result = math.atan2(delta_y, delta_x)
        result = math.degrees(result)
        if result < 0:
            result = 360 + result

        # Half chances of orientating the car to the exactly opposite direction
        random_orientation = random.randint(0, 1)
        if random_orientation == 1:
            result = (result + 180) % 360

        radians = math.radians(result)
        orientation_z = radians

        random_start_point = np.array(
            [round(random_point['pose.pose.position.x'], 3), round(random_point['pose.pose.position.y'], 3),
             round(random_point['pose.pose.position.z'], 3), round(random_point['pose.pose.orientation.x'], 3),
             round(random_point['pose.pose.orientation.y'], 3), round(orientation_z, 3)])

        for child_1 in root[0]:
            if child_1.tag == 'include':
                next = False
                for child_2 in child_1:
                    if next:
                        child_2.text = str(random_start_point[0]) + " " + str(random_start_point[1]) + " " + str(
                            random_start_point[2]) + " " + str(random_start_point[3]) + " " + str(
                            random_start_point[4]) + " " + str(random_start_point[5])
                        next = False
                    elif child_2.text == 'model://f1_renault':
                        next = True

    # Add physics real time update rate value
    physics_element = ET.SubElement(root[0], 'physics')
    physics_element.set("type", "ode")
    real_time_update_rate_element = ET.SubElement(physics_element, 'real_time_update_rate')
    real_time_update_rate_element.text = str(real_time_update_rate)  # 1000 is the default value

    tree.write('tmp_world.launch')
    if launch:
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
