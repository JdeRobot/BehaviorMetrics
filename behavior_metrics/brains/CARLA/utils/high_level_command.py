import numpy as np
import carla

class HighLevelCommandLoader:
    def __init__(self, vehicle, map, route=None):
        self.vehicle = vehicle
        self.map = map
        self.prev_hlc = 0
        self.route = route

    def _get_junction(self):
        """determine whether vehicle is at junction"""
        junction = None
        vehicle_location = self.vehicle.get_transform().location
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)

        # check whether vehicle is at junction
        for j in range(1, 11):
            next_waypoint = vehicle_waypoint.next(j * 1.0)[0]
            if next_waypoint.is_junction:
                junction = next_waypoint.get_junction()
                break
        if vehicle_waypoint.is_junction:
            junction = vehicle_waypoint.get_junction()
        return junction

    def _command_to_int(self, command):
        commands = {
            'Left': 1,
            'Right': 2,
            'Straight': 3
        }
        return commands[command]

    def get_random_hlc(self):
        """select a random turn at junction"""
        junction = self._get_junction()
        vehicle_location = self.vehicle.get_transform().location
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        
        # randomly select a turning direction
        if junction is not None:
            if self.prev_hlc == 0:
                valid_turns = []
                waypoints = junction.get_waypoints(carla.LaneType.Driving)
                for next_wp, _ in waypoints:
                    yaw_diff = next_wp.transform.rotation.yaw - vehicle_waypoint.transform.rotation.yaw
                    yaw_diff = (yaw_diff + 180) % 360 - 180 # convert to [-180, 180]
                    if -15 < yaw_diff < 15:
                        valid_turns.append(3)  # Go Straight
                    elif 15 < yaw_diff < 165: 
                        valid_turns.append(1)  # Turn Left
                    elif -165 < yaw_diff < -15:
                        valid_turns.append(2)  # Turn Right
                hlc = np.random.choice(valid_turns)
            else:
                hlc = self.prev_hlc
        else:
            hlc = 0

        self.prev_hlc = hlc

        return hlc

    
    def get_next_hlc(self):
        if self.route is not None and len(self.route) > 0:
            return self.load_next_hlc()
        return self.get_random_hlc()

    def load_next_hlc(self):
        """load the next high level command from pre-defined route"""
        if self.prev_hlc is None:
            return None

        junction = self._get_junction()
        
        if junction is not None:
            if self.prev_hlc == 0:
                if len(self.route) == 0:
                    hlc = None
                hlc = self._command_to_int(self.route.pop(0))
            else:
                hlc = self.prev_hlc
        else:
            hlc = 0
        
        self.prev_hlc = hlc
        
        return hlc
        