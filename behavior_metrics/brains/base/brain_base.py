
from abc import abstractmethod
import time
import numpy as np
from threading import Lock

class BrainBase():
    IT_LIST_SIZE = 100

    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        self.sensors = sensors
        self.actuators = actuators
        self.model = model
        self.handler = handler
        self.config = config
        self.it_counter = []
        self.time_it_mutex = Lock()


    @abstractmethod
    def execute_imp(self):
        pass

    @abstractmethod
    def update_frame(self, frame_id, data):
        pass

    def execute(self):
        start_time = time.time()
        self.execute_imp()
        end_time = time.time() - start_time
        self.time_it_mutex.acquire()
        if len(self.it_counter) > self.IT_LIST_SIZE:
            self.it_counter.pop()
        self.it_counter.append(end_time)
        self.time_it_mutex.release()

    def get_iteration_time(self):
        self.time_it_mutex.acquire()
        current_time = np.mean(self.it_counter)
        self.time_it_mutex.release()
        return current_time

