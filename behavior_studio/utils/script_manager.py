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
    time.sleep(10)


def run_brains_worlds(app_configuration, controller):
    for world in app_configuration.current_world:
        launch_gazebo_no_gui_worlds(world)
        # controller.pilot.configuration.current_world = world
        for brain in app_configuration.brain_path:
            logger.info('Executing brain')
            pilot = Pilot(app_configuration, controller, brain)
            pilot.daemon = True
            controller.pilot.start()
            controller.pilot.configuration.current_world = world
            # controller.reload_brain(brain)
            # controller.pilot.brains.brain_path = brain
            # controller.pilot.start()
            controller.resume_pilot()
            controller.unpause_gazebo_simulation()
            controller.record_stats(app_configuration.stats_perfect_lap, app_configuration.stats_out)
            time.sleep(20)
            controller.stop_record_stats()
            print(controller.lap_statistics)
            # controller.pilot.kill()
            controller.pause_pilot()
            controller.stop_pilot()
            pilot.stop_interfaces()
            pilot.kill()
            # controller.reset_gazebo_simulation()
            # self.configuration.current_world = world
        os.remove('tmp_circuit.launch')
        