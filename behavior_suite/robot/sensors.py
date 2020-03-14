
from robot.interfaces.camera import ListenerCamera
from robot.interfaces.laser import ListenerLaser


class Sensors:

    def __init__(self, sensors_config):

        # Load cameras
        cameras_conf = sensors_config.get('Cameras', None)
        self.cameras = None
        if cameras_conf:
            self.cameras = self.__create_sensor(cameras_conf, 'camera')

        # Load lasers
        lasers_conf = sensors_config.get('Lasers', None)
        self.lasers = None
        if lasers_conf:
            self.lasers = self.__create_sensor(lasers_conf, 'laser')

    def __create_sensor(self, sensor_config, sensor_type):

        sensor_dict = {}

        for elem in sensor_config:
            name = sensor_config[elem]['Name']
            topic = sensor_config[elem]['Topic']
            if sensor_type == 'camera':
                sensor_dict[name] = ListenerCamera(topic)
            elif sensor_type == 'laser':
                sensor_dict[name] = ListenerLaser(topic)

        return sensor_dict

    def __get_sensor(self, sensor_name, sensor_type):

        sensor = None
        try:
            if sensor_type == 'camera':
                sensor = self.cameras[sensor_name]
            elif sensor_type == 'laser':
                sensor = self.laser[sensor_name]
        except KeyError:
            return "[ERROR] No existing camera with {} name.".format(sensor_name)

        return sensor

    def get_camera(self, camera_name):
        return self.__get_sensor(camera_name, 'camera')

    def get_laser(self, laser_name):
        return self.__get_sensor(laser_name, 'laser')
