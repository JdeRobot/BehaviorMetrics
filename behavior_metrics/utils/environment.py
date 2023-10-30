#!/usr/bin/env python

"""This module contains the environment handler.
This module is in charge of loading and stopping gazebo and ros processes such as gazebo and ros launch files.
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
import sys
import time
import os
import random

from utils.logger import logger
from utils.constants import ROOT_PATH, CARLA_TOWNS_SPAWN_POINTS

import xml.etree.ElementTree as ET

# TODO: remove absolute paths

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


def launch_env(launch_file, random_spawn_point=False, carla_simulator=False, config_spawn_point=None, config_town=None):
    """Launch the environmet specified by the launch_file given in command line at launch time.
    Arguments:
        launch_file {str} -- path of the launch file to be executed
    """

    # close previous instances of ROS and simulators if hanged.
    close_ros_and_simulators()
    try:
        if carla_simulator:
            if random_spawn_point:
                tree = ET.parse(ROOT_PATH + '/' + launch_file)
                root = tree.getroot()
                town = root.find(".//*[@name=\"town\"]")
                if town is None:
                    town=root.find(".//*[@name='town']")
                spawn_point = root.find(".//*[@name=\"spawn_point\"]")
                if spawn_point is None:
                    spawn_point=root.find(".//*[@name='spawn_point']")
                if config_spawn_point:
                    spawn_point.attrib['default'] = config_spawn_point
                else:
                    spawn_point.attrib['default'] = random.choice(CARLA_TOWNS_SPAWN_POINTS[town.attrib['default']])
                tree.write('tmp_circuit.launch')
            elif config_spawn_point is not None and config_town is not None:
                tree = ET.parse(ROOT_PATH + '/' + launch_file)
                root = tree.getroot()
                town = root.find(".//*[@name=\"town\"]")
                if town is None:
                    town=root.find(".//*[@name='town']")
                town.attrib['default'] = config_town
                spawn_point = root.find(".//*[@name=\"spawn_point\"]")
                if spawn_point is None:
                    spawn_point=root.find(".//*[@name='spawn_point']")
                spawn_point.attrib['default'] = config_spawn_point
                tree.write('tmp_circuit.launch')
            else:
                spawn_point = None
            with open("/tmp/.carlalaunch_stdout.log", "w") as out, open("/tmp/.carlalaunch_stderr.log", "w") as err:
                tree = ET.parse(ROOT_PATH + '/' + launch_file)
                root = tree.getroot()
                quality = root.find(".//*[@name=\"quality\"]")
                if quality == None:
                    subprocess.Popen([os.environ["CARLA_ROOT"] + "CarlaUE4.sh", "-RenderOffScreen"], stdout=out, stderr=err)
                elif quality.attrib['default'] == 'Low':
                    subprocess.Popen([os.environ["CARLA_ROOT"] + "CarlaUE4.sh", "-RenderOffScreen", "-quality-level=Low"], stdout=out, stderr=err)                
                #subprocess.Popen(["/home/jderobot/Documents/Projects/carla_simulator_0_9_13/CarlaUE4.sh", "-RenderOffScreen", "-quality-level=Low"], stdout=out, stderr=err)
            logger.info("SimulatorEnv: launching simulator server.")
            time.sleep(5)
            with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
                if random_spawn_point or (spawn_point is not None and town is not None):
                    child = subprocess.Popen(["roslaunch", ROOT_PATH + '/tmp_circuit.launch'], stdout=out, stderr=err)
                else:
                    child = subprocess.Popen(["roslaunch", ROOT_PATH + '/' + launch_file], stdout=out, stderr=err)
        else:
            with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
                child = subprocess.Popen(["roslaunch", launch_file], stdout=out, stderr=err)
        logger.info("SimulatorEnv: launching simulator server.")
    except OSError as oe:
        logger.error("SimulatorEnv: exception raised launching simulator server. {}".format(oe))
        close_ros_and_simulators()
        sys.exit(-1)

    # give simulator some time to initialize
    time.sleep(5)


def close_ros_and_simulators(close_ros_resources=True):
    """Kill all the simulators and ROS processes."""
    try:
        ps_output = subprocess.check_output(["ps", "-Af"]).decode('utf-8').strip("\n")
    except subprocess.CalledProcessError as ce:
        logger.error("SimulatorEnv: exception raised executing ps command {}".format(ce))
        sys.exit(-1)

    if ps_output.count('gzclient') > 0:
        try:
            subprocess.check_call(["killall", "-9", "gzclient"])
            logger.debug("SimulatorEnv: gzclient killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for gzclient {}".format(ce))

    if ps_output.count('gzserver') > 0:
        try:
            subprocess.check_call(["killall", "-9", "gzserver"])
            logger.debug("SimulatorEnv: gzserver killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for gzserver {}".format(ce))

    if ps_output.count('CarlaUE4.sh') > 0:
        try:
            subprocess.check_call(["killall", "-9", "CarlaUE4.sh"])
            logger.debug("SimulatorEnv: CARLA server killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for CARLA server {}".format(ce))

    if ps_output.count('CarlaUE4-Linux-Shipping') > 0:
        try:
            subprocess.check_call(["killall", "-9", "CarlaUE4-Linux-Shipping"])
            logger.debug("SimulatorEnv: CarlaUE4-Linux-Shipping killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for CarlaUE4-Linux-Shipping {}".format(ce))

    if ps_output.count('rosout') > 0 and close_ros_resources:
        try:
            import rosnode
            for node in rosnode.get_node_names():
                if node != '/carla_ros_bridge':
                    subprocess.check_call(["rosnode", "kill", node])

            logger.debug("SimulatorEnv:rosout killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for rosout {}".format(ce))

    if ps_output.count('bridge.py') > 0:
        try:
            os.system("ps -ef | grep 'bridge.py' | awk '{print $2}' | xargs kill -9")
            logger.debug("SimulatorEnv:bridge.py killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for bridge.py {}".format(ce))
        except FileNotFoundError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for bridge.py {}".format(ce))

    if ps_output.count('rosmaster') > 0 and close_ros_resources:
        try:
            subprocess.check_call(["killall", "-9", "rosmaster"])
            logger.debug("SimulatorEnv: rosmaster killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for rosmaster {}".format(ce))

    if ps_output.count('roscore') > 0 and close_ros_resources:
        try:
            subprocess.check_call(["killall", "-9", "roscore"])
            logger.debug("SimulatorEnv: roscore killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for roscore {}".format(ce))

    if ps_output.count('px4') > 0:
        try:
            subprocess.check_call(["killall", "-9", "px4"])
            logger.debug("SimulatorEnv: px4 killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for px4 {}".format(ce))

    if ps_output.count('roslaunch') > 0 and close_ros_resources:
        try:
            subprocess.check_call(["killall", "-9", "roslaunch"])
            logger.debug("SimulatorEnv: roslaunch killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for roslaunch {}".format(ce))
    
    if ps_output.count('rosout') > 0 and close_ros_resources:
        try:
            subprocess.check_call(["killall", "-9", "rosout"])
            logger.debug("SimulatorEnv:rosout killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for rosout {}".format(ce))


def is_gzclient_open():
    """Determine if there is an instance of Gazebo GUI running
    Returns:
        bool -- True if there is an instance running, False otherwise
    """

    try:
        ps_output = subprocess.check_output(["ps", "-Af"], encoding='utf8').strip("\n")
    except subprocess.CalledProcessError as ce:
        logger.error("SimulatorEnv: exception raised executing ps command {}".format(ce))
        sys.exit(-1)

    return ps_output.count('gzclient') > 0


def close_gzclient():
    """Close the Gazebo GUI if opened."""

    if is_gzclient_open():
        try:
            subprocess.check_call(["killall", "-9", "gzclient"])
            logger.debug("SimulatorEnv: gzclient killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for gzclient {}".format(ce))


def open_gzclient():
    """Open the Gazebo GUI if not running"""

    if not is_gzclient_open():
        try:
            with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
                subprocess.Popen(["gzclient"], stdout=out, stderr=err)
            logger.debug("SimulatorEnv: gzclient started.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing gzclient {}".format(ce))
