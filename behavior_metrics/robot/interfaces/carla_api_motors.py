# robot/interfaces/motors.py

import numpy as np
import carla

from robot.interfaces.carla_api_sensors import get_carla

class CarlaApiMotors:
    def __init__(self, vmax=3.0, wmax=0.3):
        _, world, ego = get_carla()  # ← obtiene el ego por role_name
        self._vehicle = ego  # ← ***esto faltaba***
        self._control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.maxV = vmax
        self.maxW = wmax

    def _apply(self):
        self._vehicle.apply_control(self._control)

    def sendThrottle(self, throttle: float):
        self._control.throttle = float(np.clip(throttle, 0.0, 1.0))
        self._apply()

    def sendSteer(self, steer: float):
        self._control.steer = float(np.clip(steer, -1.0, 1.0))
        self._apply()

    def sendBrake(self, brake: float):
        self._control.brake = float(np.clip(brake, 0.0, 1.0))
        self._apply()

    # API de compatibilidad
    def getMaxV(self):
        return self.maxV

    def getMaxW(self):
        return self.maxW

    def stop(self):
        pass