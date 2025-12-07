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


# Import ROS only if needed
ROS_VERSION = os.environ.get('ROS_VERSION', 'None')
USE_ROS = ROS_VERSION in ('1', '2')

def launch_env(launch_file, random_spawn_point=False, carla_simulator=False,
               config_spawn_point=None, config_town=None):
    """Launch the environment according to ROS version or Python API mode."""
    close_ros_and_simulators()

    try:
        # Detect current mode
        ROS_VERSION = os.environ.get('ROS_VERSION', 'None')
        USE_ROS = ROS_VERSION in ('1', '2')

        # --- PYTHON API MODE (sin ROS) ---
        if USE_ROS or ROS_VERSION == 'None':
            logger.info("Launching environment in Python API mode (ROS_VERSION=None)")
            
            carla_bin = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
            carla_root = os.environ.get("CARLA_ROOT")
            
            # comprobar si CARLA ya corre
            ps_output = subprocess.check_output(["ps", "-Af"], encoding="utf-8")
            carla_running = ("CarlaUE4.sh" in ps_output) or ("CarlaUE4-Linux-Shipping" in ps_output)
            print(f"Carla running: {carla_running}")

            # iniciar servidor CARLA si no está activo
            if not carla_running:
                if not os.path.exists(carla_bin):
                    raise FileNotFoundError(f"Carla not found {carla_bin}")
                
                # carla_bin = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
                with open("/tmp/.carla_stdout.log", "w") as out, open("/tmp/.carla_stderr.log", "w") as err:
                    subprocess.Popen([carla_bin, "-RenderOffScreen", "-prefernvidia"],  # "/bin/bash", 
                                     cwd=carla_root,
                                     stdout=out, stderr=err,
                                     shell=False,
                                     env=os.environ
                                     )
                logger.info("CARLA server started (Python API)")
                time.sleep(10)

            # lanzar generador del mundo

            launch_file = os.path.join(ROOT_PATH,launch_file)
            world_gen_path = os.path.join(ROOT_PATH, "utils/carla_world_generator.py")
            
            if not os.path.exists(world_gen_path):
                raise FileNotFoundError(f"carla_world_generator.py not found at {world_gen_path}")

            logger.info(f"Running world generator: {world_gen_path}")
            with open("/tmp/.worldgen_stdout.log", "w") as out, open("/tmp/.worldgen_stderr.log", "w") as err:
                subprocess.Popen(["python3", world_gen_path, launch_file], stdout=out, stderr=err,)
            logger.info("World generator launched successfully (Python API)")
            time.sleep(5)
            return  # no seguir al flujo ROS

        # --- ROS 1 / ROS 2 MODES ---
        spawn_point = None
        town = None
        tree = None
        root = None
        launch_file_path = None

        if carla_simulator:
            # analizar archivo launch para ROS1 (.launch) o ROS2 (.launch.py)
            if launch_file.endswith('.launch') and ROS_VERSION == '1':
                xml_path = os.path.join(ROOT_PATH, launch_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
            elif launch_file.endswith('.launch.py') and ROS_VERSION == '2':
                xml_path = os.path.join(ROOT_PATH, launch_file.replace('.launch.py', '.launch'))
                if os.path.exists(xml_path):
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                else:
                    logger.warning(f"No XML 'twins' file found for {launch_file}, town/spawn not readable.")
            else:
                logger.warning("Launch file format not supported for CARLA simulator.")

            # extraer town/spawn si hay XML
            if tree is not None and root is not None:
                town = root.find(".//*[@name='town']")
                spawn_point = root.find(".//*[@name='spawn_point']")
                if town is None or spawn_point is None:
                    raise ValueError("Missing 'town' or 'spawn_point' in launch XML.")

                if random_spawn_point:
                    town_default = town.attrib.get('default', '')
                    spawn_list = CARLA_TOWNS_SPAWN_POINTS.get(town_default, [])
                    spawn_point.attrib['default'] = config_spawn_point or random.choice(spawn_list)
                elif config_spawn_point and config_town:
                    spawn_point.attrib['default'] = config_spawn_point
                    town.attrib['default'] = config_town

                tmp_launch = os.path.join(ROOT_PATH, 'tmp_circuit.launch')
                tree.write(tmp_launch)
                launch_file_path = tmp_launch if ROS_VERSION == '1' else os.path.join(ROOT_PATH, launch_file)
            else:
                launch_file_path = os.path.join(ROOT_PATH, launch_file)

            # lanzar servidor CARLA
            ps_output = subprocess.check_output(["ps", "-Af"], encoding="utf-8")
            if "CarlaUE4" not in ps_output:
                carla_bin = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
                quality = None
                if root is not None:
                    qnode = root.find(".//*[@name='quality']")
                    quality = qnode.attrib.get('default', '').lower() if qnode is not None else None

                args = [carla_bin, "-RenderOffScreen"]
                if quality == "low":
                    args += ["-quality-level=Low"]

                with open("/tmp/.carla_stdout.log", "w") as out, open("/tmp/.carla_stderr.log", "w") as err:
                    subprocess.Popen(args, stdout=out, stderr=err)
                logger.info(f"CARLA server started (ROS mode, quality={quality or 'default'})")
                time.sleep(5)

            # lanzar roslaunch / ros2 launch
            with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
                if ROS_VERSION == '2':
                    launch_path = os.path.abspath(launch_file_path)
                    ros_cmd = ['ros2', 'launch', launch_path]
                else:
                    ros_cmd = ['roslaunch', launch_file_path]
                subprocess.Popen(ros_cmd, stdout=out, stderr=err)
                logger.info(f"ROS ({ROS_VERSION}) launch started: {' '.join(ros_cmd)}")

        else:
            # lanzar ROS sin simulador
            with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
                if ROS_VERSION == '2':
                    launch_path = os.path.join(ROOT_PATH, launch_file)
                    ros_cmd = ['ros2', 'launch', launch_path]
                else:
                    ros_cmd = ['roslaunch', launch_file]
                subprocess.Popen(ros_cmd, stdout=out, stderr=err, preexec_fn=os.setsid)
                logger.info(f"ROS ({ROS_VERSION}) launch started: {' '.join(ros_cmd)}")

    except OSError as oe:
        logger.error(f"SimulatorEnv: exception raised launching simulator server. {oe}")
        close_ros_and_simulators()
        sys.exit(-1)

    # tiempo para inicializar
    time.sleep(10)



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
        # kill zombies processes -> nohup ./CarlaUE4.sh > /dev/null 2>&1 &
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
    
    if ps_output.count('carla_manual_control') > 0:
        try:
            subprocess.check_call(["killall", "-9", "carla_manual_control"])
            logger.debug("SimulatorEnv: carla_manual_control killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for carla_manual_control {}".format(ce))


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