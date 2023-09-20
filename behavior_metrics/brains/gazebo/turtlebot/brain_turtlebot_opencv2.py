#!/usr/bin/python
# -*- coding: utf-8 -*-


class Brain:

    def __init__(self, sensors, actuators, handler=None):
        self.camera_l = sensors.get_camera('camera_0')
        self.camera_r = sensors.get_camera('camera_1')
        self.laser = sensors.get_laser('laser_0')
        self.pose = sensors.get_pose3d('pose3d_0')
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler

    def update_frame(self, frame_id, data):
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        self.handler.update_pose3d(pose_data)

    def execute(self):
        self.update_pose(self.pose.getPose3d())
        v = 0
        w = 0.8
        self.motors.sendV(v)
        self.motors.sendW(w)
        image_l = self.camera_l.getImage().data
        image_r = self.camera_r.getImage().data
        laser_data = self.laser.getLaserData()
        self.update_frame('frame_0', image_l)
        self.update_frame('frame_1', image_r)
        self.update_frame('frame_3', laser_data)
