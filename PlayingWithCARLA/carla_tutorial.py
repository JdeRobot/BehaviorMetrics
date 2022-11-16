import glob
import os
import sys

try:
    sys.path.append(glob.glob('PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import cv2
import skimage.measure as measure

#in synchronous mode, sensor data must be added to a queue
import queue

client = carla.Client('localhost', 2000)
client.set_timeout(11.0)

#print(client.get_available_maps())
#world = client.load_world('Town03')
#settings = world.get_settings()
#settings.fixed_delta_seconds = 0.05 #must be less than 0.1, or else physics will be noisy
#must use fixed delta seconds and synchronous mode for python api controlled sim, or else
#camera and sensor data may not match simulation properly and will be noisy
#settings.synchronous_mode = True
#world.apply_settings(settings)

actor_list = []

blueprint_library = client.get_world().get_blueprint_library()
bp = random.choice(blueprint_library.filter('vehicle')) # lets choose a vehicle at random

# lets choose a random spawn point
transform = random.choice(client.get_world().get_map().get_spawn_points()) 

#spawn a vehicle
vehicle =  client.get_world().try_spawn_actor(bp, transform) 
actor_list.append(vehicle)

vehicle.set_autopilot(True)



print()






