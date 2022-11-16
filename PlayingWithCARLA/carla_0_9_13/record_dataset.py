import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time
import csv
from os import listdir
from os.path import isfile, join
import pandas
import math

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
world = client.get_world()

time.sleep(2)

world.get_actors()
car = world.get_actors().filter('vehicle.*')[0]

camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
#camera_transform = carla.Transform(carla.Location(x=0, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=car)
#image_queue = queue.Queue()
#camera.listen(image_queue.put)
last_image = ''
def listener(image):
    global last_image
    last_image = image
camera.listen(listener)

#rgb camera
#image = image_queue.get()
# open the file in the write mode
f = open('../../carla_dataset_test_26_10_anticlockwise_full_image_segmentation/dataset.csv', 'a')

# create the csv writer
writer = csv.writer(f)

#mypath = '../carla_dataset_14_09/'
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#onlyfiles = sorted(onlyfiles)
#print(onlyfiles)
#print(onlyfiles[len(onlyfiles)-2])
#print(onlyfiles[len(onlyfiles)-2][0])
#print(type(onlyfiles[len(onlyfiles)-2][0]))

try: 
    df = pandas.read_csv('../../carla_dataset_test_26_10_anticlockwise_full_image_segmentation/dataset.csv')
    batch_number = int(df.iloc[-1]['batch']) + 1
except Exception as ex:
    #ddddd
    batch_number = 0
    header = ['batch', 'image_id', 'timestamp', 'throttle', 'steer', 'brake', 'location_x', 'location_y']
    writer.writerow(header)

print(batch_number)
frame_number = 0
try:
    while True:        
        vehicle_control = car.get_control()
        vehicle_location = car.get_location()
        speed = car.get_velocity()
        vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)
        '''
        print(type(image_queue))
        print(image_queue)
        print(image_queue.qsize())
        '''
        if (vehicle_speed > 0.0 and last_image != ''):
            #image = image_queue.get()
            image = last_image
            cc_raw = carla.ColorConverter.Raw
            cc_seg = carla.ColorConverter.CityScapesPalette
            image.save_to_disk('../../carla_dataset_test_26_10_anticlockwise_full_image_segmentation/raw/' + str(batch_number) + '_' +  str(frame_number) + '.png', cc_raw)
            image.save_to_disk('../../carla_dataset_test_26_10_anticlockwise_full_image_segmentation/mask/' + str(batch_number) + '_' +  str(frame_number) + '.png', cc_seg)
            # write a row to the csv file
            row = [batch_number, str(batch_number) + '_' + str(frame_number) + '.png', time.time(), vehicle_control.throttle, vehicle_control.steer, vehicle_control.brake, vehicle_location.x, vehicle_location.y]
            writer.writerow(row)
            frame_number += 1
            print(vehicle_control)
        else:
            pass
            #print('nop')
except KeyboardInterrupt:
    pass

# close the file
f.close()


