import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
world = client.get_world()
print(world)
time.sleep(2)
car = world.get_actors().filter('vehicle.*')[0]
car.set_autopilot(True)
print(car.is_at_traffic_light())

'''
if vehicle_actor.is_at_traffic_light():
    traffic_light = vehicle_actor.get_traffic_light()
    if traffic_light.get_state() == carla.TrafficLightState.Red:
       # world.hud.notification("Traffic light changed! Good to go!")
        traffic_light.set_state(carla.TrafficLightState.Green)
'''

'''
my_vehicles = [car]

tm = client.get_trafficmanager(port)
tm_port = tm.get_port()
for v in my_vehicles:
  v.set_autopilot(True,tm_port)
danger_car = my_vehicles[0]
tm.ignore_lights_percentage(danger_car,100)
tm.distance_to_leading_vehicle(danger_car,0)
tm.vehicle_percentage_speed_difference(danger_car,-20)

'''
