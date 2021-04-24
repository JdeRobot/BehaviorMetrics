#!/usr/bin/env python

from drone_wrapper import DroneWrapper


class Brain:

    def __init__(self, sensors=None, actuators=None, handler=None):
        self.drone = DroneWrapper()
        self.handler = handler
        # self.drone.takeoff()

    def update_frame(self, frame_id, data):
        self.handler.update_frame(frame_id, data)

    def execute(self):

        img_frontal = self.drone.get_frontal_image()
        img_ventral = self.drone.get_ventral_image()
        # Both the above images are cv2 images

        self.update_frame('frame_0', img_frontal)
        self.update_frame('frame_1', img_ventral)
