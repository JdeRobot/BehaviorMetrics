import subprocess
import sys
import time
from logger import logger

# TODO: quitar paths absolutos


def launch_env(launch_file):
    close_gazebo()
    try:
        with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
            subprocess.Popen(["roslaunch", launch_file], stdout=out, stderr=err)
        logger.info("GazeboEnv: launching gzserver.")
    except OSError as oe:
        logger.error("GazeboEnv: exception raised launching gzserver. {}".format(oe))
        close_gazebo()
        sys.exit(-1)

    time.sleep(5)


def close_gazebo():
    try:
        ps_output = subprocess.check_output(["ps", "-Af"]).strip("\n")
    except subprocess.CalledProcessError as ce:
        logger.error("GazeboEnv: exception raised executing ps command {}".format(ce))
        sys.exit(-1)

    if ps_output.count('gzclient') > 0:
        try:
            subprocess.check_call(["killall", "-9", "gzclient"])
            logger.debug("GazeboEnv: gzclient killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("GazeboEnv: exception raised executing killall command for gzclient {}".format(ce))

    if ps_output.count('gzserver') > 0:
        try:
            subprocess.check_call(["killall", "-9", "gzserver"])
            logger.debug("GazeboEnv: gzserver killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("GazeboEnv: exception raised executing killall command for gzserver {}".format(ce))

    if ps_output.count('rosmaster') > 0:
        try:
            subprocess.check_call(["killall", "-9", "rosmaster"])
            logger.debug("GazeboEnv: rosmaster killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("GazeboEnv: exception raised executing killall command for rosmaster {}".format(ce))

    if ps_output.count('roscore') > 0:
        try:
            subprocess.check_call(["killall", "-9", "roscore"])
            logger.debug("GazeboEnv: roscore killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("GazeboEnv: exception raised executing killall command for roscore {}".format(ce))
