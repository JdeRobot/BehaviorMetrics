import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time

print(carla.__file__)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
world = client.get_world()
print(world)
time.sleep(2)

traffic_lights = world.get_actors().filter('traffic.traffic_light')
traffic_speed_limits = world.get_actors().filter('traffic.speed_limit*')
print(traffic_speed_limits)
for traffic_light in traffic_lights:
    #success = traffic_light.destroy()
    traffic_light.set_green_time(20000)
    traffic_light.set_state(carla.TrafficLightState.Green)
    #print(success)

for speed_limit in traffic_speed_limits:
   success = speed_limit.destroy()
   print(success)

print(world.get_actors().filter('vehicle.*'))
car = world.get_actors().filter('vehicle.*')[0]


#settings = world.get_settings()
#settings.synchronous_mode = True # Enables synchronous mode
#world.apply_settings(settings)
traffic_manager = client.get_trafficmanager()
#random.seed(0)
#car.set_autopilot(True)
car.set_autopilot(True)

'''
# ROUTE 0
route = ["Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight"]

# ROUTE 1
route = ["Left", "Straight", "Right", "Right", "Straight", "Left", "Left",
"Right", "Right", "Left", "Straight", "Left"]
'''
# ROUTE 2
route = ["Left", "Right", "Straight", "Right", "Right", "Straight", "Straight", "Right", "Left", "Left", "Right", "Right"]

traffic_manager.set_route(car, route)
iterator = 0
previous_waypoint_no_junction = True

while True:
    #traffic_manager.set_route(car, route)
    #print(traffic_manager.get_next_action(car))
    #print(traffic_manager.get_all_actions(car))

    waypoint = world.get_map().get_waypoint(car.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
    #print(waypoint.is_junction)
    if waypoint.is_junction and previous_waypoint_no_junction:
        # first point in junction
        previous_waypoint_no_junction = False
        print('---------------------')
        print('ENTRAMOS EN JUNCTION')
        print(iterator)
        print('Accion -> ', route[iterator])
        print()
    elif not waypoint.is_junction and not previous_waypoint_no_junction:
        # last point in junction
        previous_waypoint_no_junction = True
        iterator += 1
        print('SALIMOS DE JUNCTION')
        print(iterator)
        print('Siguiente junction -> ', route[iterator])
        print('---------------------')
        print()

    #print(iterator)

'''
    print("Current lane type: " + str(waypoint.lane_type))
    # Check current lane change allowed
    print("Current Lane change:  " + str(waypoint.lane_change))
    # Left and Right lane markings
    print("L lane marking type: " + str(waypoint.left_lane_marking.type))
    print("L lane marking change: " + str(waypoint.left_lane_marking.lane_change))
    print("R lane marking type: " + str(waypoint.right_lane_marking.type))
    print("R lane marking change: " + str(waypoint.right_lane_marking.lane_change))
    #location = car.get_location()
    #print(type(location))
    #print(location.is_junction())
'''

'''
import math
while True:
    speed = car.get_velocity()
    vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)
    if (abs(vehicle_speed) > 5):
        print('SLOW DOWN!')
        car.apply_control(carla.VehicleControl(throttle=float(0), steer=float(0), brake=float(1.0)))


'''
'''
# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
#traffic_manager.set_synchronous_mode(True)

# Set a seed so behaviour can be repeated if necessary
#traffic_manager.set_random_device_seed(0)
#random.seed(0)
#car.set_autopilot(True)
car.set_autopilot(True)
route = ["Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight"]
traffic_manager.set_route(car, route)
#time.sleep(3)
#car.set_autopilot(False)
#print('autopilot false!')
'''
'''
car.apply_control(carla.VehicleControl(throttle=float(0.75), steer=-0.1, brake=float(0.0)))
time.sleep(3)
car.set_autopilot(True)
#traffic_manager.set_route(car, route)
'''


'''
while True:
    print('entra')
    world.tick()

    #traffic_manager.update_vehicle_lights(car, True)
    #traffic_manager.random_left_lanechange_percentage(car, 0)
    #traffic_manager.random_right_lanechange_percentage(car, 0)
    #traffic_manager.auto_lane_change(car, False)

    world_map =world.get_map()
    spawn_points = world_map.get_spawn_points()

    # Create route 1 from the chosen spawn points
    route_1_indices = [129, 28, 124, 33, 97, 119, 58, 154, 147]
    route_1 = []
    for ind in route_1_indices:
        route_1.append(spawn_points[ind].location)
    print(route_1)
    traffic_manager.set_path(car, route_1)
'''


