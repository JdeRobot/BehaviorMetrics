import numpy as np
import carla

class HighLevelCommandLoader:
    def __init__(self, vehicle, map):
        self.vehicle = vehicle
        self.map = map
        self.prev_hlc = 0
    
    def get_random_hlc(self):
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

