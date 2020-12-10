from utils import environment
import subprocess
import xml.etree.ElementTree as ET
import time
import os
from utils.logger import logger
from pilot import Pilot

    
def launch_gazebo_no_gui_worlds(current_world):
    environment.close_gazebo()
    tree = ET.parse(current_world)
    root = tree.getroot()
    for child in root[0]:
        if child.attrib['name'] == 'gui':
            # child.attrib['value'] = 'false'
            child.attrib['value'] = 'true'

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
    launch_gazebo_no_gui_worlds(app_configuration.current_world[0])
    pilot = Pilot(app_configuration, controller, app_configuration.brain_path[0])
    pilot.daemon = True
    controller.pilot.start()
    for i, world in enumerate(app_configuration.current_world):
        for x, brain in enumerate(app_configuration.brain_path):
            # 1. Load world
            launch_gazebo_no_gui_worlds(world)
            controller.initialize_robot()
            controller.pilot.configuration.current_world = world
            logger.info('Executing brain')
            # 2. Play
            controller.reload_brain(brain)
            controller.resume_pilot()
            controller.pilot.configuration.brain_path = app_configuration.brain_path
            controller.unpause_gazebo_simulation()
            controller.record_stats(app_configuration.stats_perfect_lap[i], app_configuration.stats_out)
            time.sleep(app_configuration.experiment_timeout)
            controller.stop_record_stats()
            # 3. Stop
            controller.pause_pilot()
            controller.pause_gazebo_simulation()
            print(controller.lap_statistics)
        os.remove('tmp_circuit.launch')
        