#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils.logger import logger

class Brain:

    def __init__(self, sensors, actuators, handler):
        # super(Brain, self).__init__(sensors, actuators, brain_path=None)
        self.camera_c = sensors.get_camera('camera_c')
        self.camera_l = sensors.get_camera('camera_l')
        self.camera_r = sensors.get_camera('camera_r')
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
        image_c = self.camera_c.getImage().data
        image_l = self.camera_l.getImage().data
        image_r = self.camera_r.getImage().data
        # self.update_frame('frame_0', image)
        self.update_frame('frame_0', image_c)
        self.update_frame('frame_1', image_l)
        self.update_frame('frame_2', image_r)
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
