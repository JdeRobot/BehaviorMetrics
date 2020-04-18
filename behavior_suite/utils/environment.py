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
    pass

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

    if ps_output.count('px4') > 0:
        try:
            subprocess.check_call(["killall", "-9", "px4"])
            logger.debug("GazeboEnv: px4 killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("GazeboEnv: exception raised executing killall command for px4 {}".format(ce))


def is_gzclient_open():
    try:
        ps_output = subprocess.check_output(["ps", "-Af"]).strip("\n")
    except subprocess.CalledProcessError as ce:
        logger.error("GazeboEnv: exception raised executing ps command {}".format(ce))
        sys.exit(-1)

    return ps_output.count('gzclient') > 0

def close_gzclient():
    if is_gzclient_open():
        try:
            subprocess.check_call(["killall", "-9", "gzclient"])
            logger.debug("GazeboEnv: gzclient killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("GazeboEnv: exception raised executing killall command for gzclient {}".format(ce))

def open_gzclient():
    if not is_gzclient_open():
        try:
            with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
                subprocess.Popen(["gzclient"], stdout=out, stderr=err)
            logger.debug("GazeboEnv: gzclient started.")
        except subprocess.CalledProcessError as ce:
            logger.error("GazeboEnv: exception raised executing gzclient {}".format(ce))
