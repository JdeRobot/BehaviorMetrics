import threading
import time
from datetime import datetime

from brains.brains_handler import Brains
from robot.actuators import Actuators
from robot.sensors import Sensors
from utils.logger import logger

TIME_CYCLE = 60


class Pilot(threading.Thread):

    def __init__(self, configuration, controller):
        self.controller = controller
        self.controller.set_pilot(self)
        self.configuration = configuration

        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        threading.Thread.__init__(self, args=self.stop_event)

        self.actuators = Actuators(self.configuration.actuators)
        self.sensors = Sensors(self.configuration.sensors)
        self.brains = Brains(self.sensors, self.actuators, self.configuration.brain_path, self.controller)

        # thread_ui_comm = threading.Thread(target=self.ui_listener)
        # thread_ui_comm.daemon = True
        # thread_ui_comm.start()

        # self.ros_handler = RosSrvHandler()
        # self.ros_handler.pause_gazebo_simulation()  # start the simulation paused
        # TODO: improve for real robots, not only simulation
        gazebo_ready = False
        while not gazebo_ready:
            try:
                # self.controller.pause_gazebo_simulation()
                gazebo_ready = True
            except Exception:
                pass

    def run(self):
        it = 0
        ss = time.time()
        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():

                self.brains.active_brain.execute()

            dt = datetime.now() - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            elapsed = time.time() - ss
            if elapsed < 1:
                it += 1
            else:
                ss = time.time()
                # print(it)
                it = 0

            if (ms < TIME_CYCLE):
                time.sleep((TIME_CYCLE - ms) / 1000.0)
        logger.debug('Pilot: pilot killed.')

    def stop(self):
        self.stop_event.set()

    def play(self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill(self):
        self.actuators.kill()
        self.kill_event.set()

    def reload_brain(self, brain_path):
        self.brains.load_brain(brain_path)

    # def callback(self, data):
    #     if data.data == env.PAUSE_SIMULATION:
    #         self.ros_handler.pause_gazebo_simulation()
    #     elif data.data == env.RESUME_SIMULATION:
    #         self.ros_handler.unpause_gazebo_simulation()
    #     elif data.data == env.CHANGE_BRAIN:
    #         self.ros_handler.pause_gazebo_simulation()
    #         self.brains.load_brain('brains/f1/brain_f1_opencv2.py')
    #         self.ros_handler.unpause_gazebo_simulation()
    #     elif data.data == env.RECORD_DATASET:
    #         self.ros_handler.record_rosbag(None, None)
    #     elif data.data == env.STOP_RECORD_DATASET:
    #         self.ros_handler.stop_record()
    #     elif data.data == 'quit':
    #         self.kill()

    # def callback_topics(self, data):
    #     pass

    # def ui_listener(self):
    #     rospy.Subscriber("/behavior/ui_comm", String, self.callback)
    #     rospy.Subscriber("/behavior/ui_comm", String, self.callback_topics)
    #     rospy.spin()
