from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time
import csv
from os import listdir
from os.path import isfile, join
import pandas
import time
import math

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
world = client.get_world()

time.sleep(2)


'''
birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=100, height=300),
    pixels_per_meter=10,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
)
'''
birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=100, height=300),
    pixels_per_meter=10,
    crop_type=BirdViewCropType.FRONT_AREA_ONLY
)

world.get_actors()
car = world.get_actors().filter('vehicle.*')[0]



# open the file in the write mode
f = open('../../carla_dataset_control/carla_dataset_test_09_11_anticlockwise_town_01/dataset.csv', 'a')
# create the csv writer
writer = csv.writer(f)

try: 
    df = pandas.read_csv('../../carla_dataset_control/carla_dataset_test_09_11_anticlockwise_town_01/dataset.csv')
    batch_number = int(df.iloc[-1]['batch']) + 1
except Exception as ex:
    #fkjfbdjkhfsd
    batch_number = 0
    header = ['batch', 'image_id', 'timestamp', 'throttle', 'steer', 'brake', 'location_x', 'location_y', 'previous_velocity', 'current_velocity', 'control_command']
    writer.writerow(header)

print(batch_number)

'''
car.apply_control(carla.VehicleControl(throttle=float(1.0), steer=0.0, brake=float(0.0)))
time.sleep(2)
car.apply_control(carla.VehicleControl(throttle=float(0.0), steer=0.0, brake=float(1.0)))
time.sleep(5)
car.apply_control(carla.VehicleControl(throttle=float(0.5), steer=0.1, brake=float(0.0)))
time.sleep(3)
car.set_autopilot(True)
'''
frame_number = 0
previous_speed = 0
#previous_image = 0

route = ["Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight",
"Straight", "Straight", "Straight", "Straight", "Straight"]
'''
route = ["Left", "Straight", "Right", "Right", "Straight", "Left", "Left",
"Right", "Right", "Left", "Straight", "Left"]
'''

iterator = 0
previous_waypoint_no_junction = True

#import numpy as np
try:
    while world.wait_for_tick():
        vehicle_control = car.get_control()
        vehicle_location = car.get_location()
        #print(car.get_location())
        #print(vehicle_control.steer)
        
        #if (vehicle_control.throttle > 0.1 and vehicle_control.throttle < 0.8 ) or (vehicle_control.brake > 0.2):
        #if (vehicle_control.steer < -0.25 or vehicle_control.steer > 0.25):
        if (vehicle_control.throttle > 0.0 or vehicle_control.steer > 0.0 or vehicle_control.brake > 0.0):
            #print(frame_number)
            #print(frame_number)
            # Input for your model - call it every simulation step
            # returned result is np.ndarray with ones and zeros of shape (8, height, width)
            birdview = birdview_producer.produce(
                agent_vehicle=car  # carla.Actor (spawned vehicle)
            )
            image = BirdViewProducer.as_rgb(birdview)
            #if np.array_equal(previous_image, image):
            #    print(True)
            #previous_image = image
            #print(image.shape)
            #print(type(image))
            cv2.imwrite('../../carla_dataset_control/carla_dataset_test_09_11_anticlockwise_town_01/' + str(batch_number) + '_' +  str(frame_number) + '.png', image)
            #image.save_to_disk('../../carla_dataset_21_09/' + str(batch_number) + '_' +  str(frame_number) + '.png', cc)
            #vehicle_control = car.get_vehicle_control()
            #vehicle_control = car.get_control()
            
            
            #print(vehicle_control)
            
            
            # write a row to the csv file
            speed = car.get_velocity()
            vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)
            
            waypoint = world.get_map().get_waypoint(car.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
            if waypoint.is_junction and previous_waypoint_no_junction:
                # first point in junction
                previous_waypoint_no_junction = False
                #print('---------------------')
                #print('ENTRAMOS EN JUNCTION')
                #print(iterator)
                #print('Accion -> ', route[iterator])
                #print()
            elif not waypoint.is_junction and not previous_waypoint_no_junction:
                # last point in junction
                previous_waypoint_no_junction = True
                iterator += 1
                #print('SALIMOS DE JUNCTION')
                #print(iterator)
                #print('Siguiente junction -> ', route[iterator])
                #print('---------------------')
                #print()
            #print('Siguiente junction -> ', route[iterator])
            row = [batch_number, str(batch_number) + '_' + str(frame_number) + '.png', time.time(), vehicle_control.throttle, vehicle_control.steer, vehicle_control.brake, vehicle_location.x, vehicle_location.y, previous_speed, vehicle_speed, route[iterator]]
            writer.writerow(row)
            previous_speed = vehicle_speed
            #print('----')
            #print(previous_speed)
            #print('----')
            frame_number += 1
        else:
            pass
            #print('nop')
        
except KeyboardInterrupt:
    pass

# close the file
f.close()


