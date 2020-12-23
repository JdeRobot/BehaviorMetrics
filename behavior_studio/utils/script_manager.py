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

import numpy as np

from utils import metrics
from utils import environment
from utils.logger import logger
from pilot import Pilot

    
def launch_gazebo_no_gui(current_world):
    environment.close_gazebo()
    tree = ET.parse(current_world)
    root = tree.getroot()
    for child in root[0]:
        if child.attrib['name'] == 'gui':
            child.attrib['value'] = 'false'

    tree.write('tmp_circuit.launch')
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
    # Start Behavior Studio app
    launch_gazebo_no_gui(app_configuration.current_world[0])
    pilot = Pilot(app_configuration, controller, app_configuration.brain_path[0])
    pilot.daemon = True
    controller.pilot.start()
    for i, world in enumerate(app_configuration.current_world):
        for brain in app_configuration.brain_path:
            # 1. Load world
            launch_gazebo_no_gui(world)
            controller.initialize_robot()
            controller.pilot.configuration.current_world = world
            controller.pilot.brains.brain_path = brain
            logger.info('Executing brain')
            # 2. Play
            controller.reload_brain(brain)
            controller.resume_pilot()
            controller.pilot.configuration.brain_path = app_configuration.brain_path
            controller.unpause_gazebo_simulation()
            controller.record_stats(app_configuration.stats_perfect_lap[i], app_configuration.stats_out)

            time_start = rospy.get_time()
            perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag('lap-simple-circuit.bag')
            point = np.array([controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().x, controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().y])
            
            finish_line = False
            while (rospy.get_time() - time_start < app_configuration.experiment_timeout and not finish_line) or rospy.get_time() - time_start < 10:
                rospy.sleep(10)
                point = np.array([controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().x, controller.pilot.sensors.get_pose3d('pose3d_0').getPose3d().y])
                if metrics.is_finish_line(point, perfect_lap_checkpoints[0]):
                    finish_line = True
            
            logger.info('--------------')
            logger.info('--- END TIME ----------------')
            time_end = rospy.get_time()
            logger.info(time_end - time_start)
            controller.stop_record_stats()
            # 3. Stop
            controller.pause_pilot()
            controller.pause_gazebo_simulation()
            logger.info('--- BRAIN ---')
            logger.info(brain)
            logger.info('--- STATS ---')
            logger.info(controller.lap_statistics)
            logger.info('--------------')
        os.remove('tmp_circuit.launch')
        