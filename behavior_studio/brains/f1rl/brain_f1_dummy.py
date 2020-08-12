#!/usr/bin/python
# -*- coding: utf-8 -*-


class Brain:
    """Specific dummy brain for the f1 robot."""

    def __init__(self, sensors, actuators, handler=None):
        """Constructor of the class.

        Arguments:
            sensors {robot.sensors.Sensors} -- Sensors instance of the robot
            actuators {robot.actuators.Actuators} -- Actuators instance of the robot

        Keyword Arguments:
            handler {brains.brain_handler.Brains} -- Handler of the current brain. Communication with the controller
            (default: {None})
        """
        self.camera = sensors.get_camera('camera_0')
        # self.laser = sensors.get_laser('laser_0')
        self.pose = sensors.get_pose3d('pose3d_0')
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        """Update the pose 3D information obtained from the robot.

        Arguments:
            data {*} -- Data to be updated, will be retrieved later by the UI.
        """
        self.handler.update_pose3d(pose_data)

    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""

        self.update_pose(self.pose.getPose3d())
        v = 0
        w = 0.8
        self.motors.sendV(v)
        self.motors.sendW(w)
        image = self.camera.getImage().data
        self.update_frame('frame_0', image)
        # laser_data = self.laser.getLaserData()
        # self.update_frame('frame_0', laser_data)
