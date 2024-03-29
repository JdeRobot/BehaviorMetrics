
import threading
import time
from datetime import datetime

time_cycle = 80


class ThreadPublisher(threading.Thread):

    def __init__(self, pub, kill_event):
        self.pub = pub
        self.kill_event = kill_event
        threading.Thread.__init__(self, args=kill_event)

    def run(self):
        while not self.kill_event.is_set():
            start_time = datetime.now()

            self.pub.publish()

            finish_time = datetime.now()
            dt = finish_time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            if ms < time_cycle:
                time.sleep((time_cycle - ms) / 1000.0)
