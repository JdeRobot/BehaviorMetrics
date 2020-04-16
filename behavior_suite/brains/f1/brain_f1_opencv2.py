#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils.logger import logger

class Brain:

    def __init__(self, sensors, actuators, handler):
        # super(Brain, self).__init__(sensors, actuators, brain_path=None)
        self.camera = sensors.get_camera('camera_0')
        # self.laser = sensors.get_laser('laser_0')
        self.pose = sensors.get_pose3d('pose3d_0')
        self.motors = actuators.get_motor('motors_0')
        # self.viewer = viewer
        self.handler = handler

    def update_frame(self, frame_id, data):
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        self.handler.update_pose3d(pose_data)

    def execute(self):
        self.update_pose(self.pose.getPose3d())
        # logger.info('asdfasdfas')
        v = 0
        w = 0.8
        self.motors.sendV(v)
        self.motors.sendW(w)
        image = self.camera.getImage().data
        # laser_data = self.laser.getLaserData()
        # self.update_frame('frame_0', image)
        # self.update_frame('frame_0', laser_data)
        self.update_frame('frame_0', image)
        # frame = None
        # try:
        #     frame = self.viewer.main_view.get_frame('frame_0')
        #     frame1 = self.viewer.main_view.get_frame('frame_1')
        #     frame2 = self.viewer.main_view.get_frame('frame_2')
        #     frame3 = self.viewer.main_view.get_frame('frame_3')
        #     frame4 = self.viewer.main_view.get_frame('frame_4')
        # except:
        #     pass
        # if frame:
        #     frame.set_data(image)
        #     frame1.set_data(image)
        #     frame2.set_data(image)
        #     frame3.set_data(image)
        #     frame4.set_data(image)
