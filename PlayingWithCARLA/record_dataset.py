import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time
import csv
from os import listdir
from os.path import isfile, join
import pandas

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
image_queue = queue.Queue()
camera.listen(image_queue.put)

#rgb camera
image = image_queue.get()
# open the file in the write mode
f = open('../carla_dataset_20_09/dataset.csv', 'a')

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
    df = pandas.read_csv('../carla_dataset_20_09/dataset.csv')
    batch_number = df.iloc[-1]['batch'] + 1
except:
    batch_number = 0
    header = ['batch', 'image_id', 'throttle', 'steer']
    writer.writerow(header)

print(batch_number)

try:
    while True:
        image = image_queue.get()
        cc = carla.ColorConverter.Raw
        image.save_to_disk('../carla_dataset_20_09/' + str(batch_number) + '_' +  str(image.frame_number) + '.png', cc)
        vehicle_control = car.get_vehicle_control()
        # write a row to the csv file
        row = [batch_number, str(batch_number) + '_' + str(image.frame_number) + '.png', vehicle_control.throttle, vehicle_control.steer]
        writer.writerow(row)

except KeyboardInterrupt:
    pass

# close the file
f.close()


