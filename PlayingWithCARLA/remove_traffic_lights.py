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

#print(world.get_actors())

#traffic_lights = world.get_actors().filter('traffic.traffic_light')

#for traffic_light in traffic_lights:
#	traffic_light.destroy()

#print('*****************')
#print(world.get_actors())

###################################################################################################
#print(world.get_actors())

#actors = world.get_actors()
#for actor in actors:
#    print(actor)    

#print('*****************')
#print(world.get_actors())


meshes = world.get_actors().filter('static.prop.mesh')
print(meshes)
for mesh in meshes:
    mesh.destroy()
print(world.get_actors())


traffic_lights = world.get_actors().filter('traffic.traffic_light')

for traffic_light in traffic_lights:
    traffic_light.destroy()
print(world.get_actors())


speed_limits = world.get_actors().filter('traffic.speed_limit.30')
print(speed_limits)
for speed_limit in speed_limits:
    speed_limit.destroy()
print(world.get_actors())

speed_limits = world.get_actors().filter('traffic.speed_limit.60')
print(speed_limits)
for speed_limit in speed_limits:
    speed_limit.destroy()
print(world.get_actors())

speed_limits = world.get_actors().filter('traffic.speed_limit.90')
print(speed_limits)
for speed_limit in speed_limits:
    speed_limit.destroy()
print(world.get_actors())



'''
print(traffic_lights[0].state)
print(type(traffic_lights[0]))
print(type(traffic_lights[0].state))
print('*****************')
print(dir(traffic_lights[0]))
print('*****************')
print(dir(traffic_lights[0].state))

carla.TrafficLightState('Red')

traffic_lights[0].set_state('Green')
traffic_lights[0].state = 'Green'
print(traffic_lights[0].state)
'''


