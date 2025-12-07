import os
import math
import numpy as np
import carla
import cv2

_client = _world = _ego = None

def get_carla():
    """Obtener cliente, mundo y vehículo ego."""
    global _client, _world, _ego
    if _client is None:
        host = os.getenv("CARLA_HOST", "127.0.0.1")
        port = int(os.getenv("CARLA_PORT", "2000"))
        _client = carla.Client(host, port)
        _client.set_timeout(10)
        _world = _client.get_world()
        _ego = _find_ego(_world, os.getenv("EGO_ROLE_NAME", "ego_vehicle"))
    return _client, _world, _ego


def _find_ego(world, role_name):
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name") == role_name:
            return a
    raise RuntimeError(f"Ego vehicle with role_name={role_name} not found")


def _str_pos_to_transform(pos_str):
    vals = list(map(float, pos_str.split()))
    x, y, z, yaw, pitch, roll = vals
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=yaw, pitch=pitch, roll=roll)
    )


class CarlaApiCamera:
    def __init__(self, cfg):
        self._is_seg = False
        if "Type" in cfg and "semantic_segmentation" in cfg["Type"].lower():
            self._is_seg = True
            
        _, self.world, self.ego = get_carla()
        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find(f"sensor.camera.{cfg['Type']}")

        # Aplicar atributos del YAML si están definidos
        if camera_bp.has_attribute('image_size_x'):
            camera_bp.set_attribute('image_size_x', str(cfg.get('Width', '800')))
            camera_bp.set_attribute('image_size_y', str(cfg.get('Height', '600')))
        if 'FOV' in cfg:
            camera_bp.set_attribute('fov', str(cfg['FOV']))
        
        camera_bp.set_attribute('role_name', cfg['Topic'])
        transform = _str_pos_to_transform(cfg.get('Position', '1.5 0 2.4 0 0 0'))
        self.sensor = self.world.spawn_actor(camera_bp, transform, attach_to=self.ego)
        self.image_data = None

        # Callback para recibir imágenes
        self.sensor.listen(lambda image: self._callback(image))

    def _callback(self, image):
        
        try:
            is_seg = "semantic_segmentation" in self.sensor.type_id
            if is_seg:
                image.convert(carla.ColorConverter.CityScapesPalette)  # coloreada

            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            if is_seg:
                self.image_data = arr[:, :, :3] 
                self.image_data = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)    
            else:
                self.image_data = arr[:, :, :3]
                self.image_data = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)     
        except Exception as e:
            print(f"[ERROR] Camera callback failed: {e}")
            self.image_data = None

    def getImage(self):
        """Devuelve la última imagen recibida (numpy.ndarray)"""
        return self.image_data

    def stop(self):
        self.destroy()

    def destroy(self):
        if hasattr(self, "sensor") and self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass
            self.sensor.destroy()
            self.sensor = None

class CarlaApiLidar:
    def __init__(self, cfg):
        _, world, ego = get_carla()
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")

        bp.set_attribute('range', str(cfg.get('Range', 50)))
        bp.set_attribute('rotation_frequency', str(cfg.get('RotationHz', 10)))
        bp.set_attribute('channels', str(cfg.get('Channels', 32)))
        bp.set_attribute('points_per_second', str(cfg.get('PointsPerSecond', 100000)))

        transform = _str_pos_to_transform(cfg.get('Position', '0 0 2 0 0 0'))
        self.sensor = world.spawn_actor(bp, transform, attach_to=ego)
        self.scan_data = None
        self.sensor.listen(lambda data: self._callback(data))

    def _callback(self, data):
        self.scan_data = data

    def getScan(self):
        return self.scan_data

    def stop(self):
        self.destroy()

    def destroy(self):
        if hasattr(self, "sensor") and self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass
            self.sensor.destroy()
            self.sensor = None

class CarlaApiPose3D:
    def __init__(self, cfg):
        _, world, ego = get_carla()
        self._world = world
        self._ego = ego

    def getPose3d(self):
        t: carla.Transform = self._ego.get_transform()
        v = self._ego.get_velocity()
        pose = type("Pose3D", (), {})()
        pose.x, pose.y, pose.z = t.location.x, t.location.y, t.location.z
        pose.roll, pose.pitch, pose.yaw = t.rotation.roll, t.rotation.pitch, t.rotation.yaw
        pose.vx, pose.vy, pose.vz = v.x, v.y, v.z
        return pose

    def stop(self):
        pass

    def destroy(self):
        pass

class CarlaApiSpeedometer:
    def __init__(self, cfg):
        _, world, ego = get_carla()
        self._ego = ego

    def getSpeed(self):
        v = self._ego.get_velocity()
        speed_m_s = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        return 3.6 * speed_m_s  # km/h

    def stop(self):
        pass

    def destroy(self):
        pass
