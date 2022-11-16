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

world.get_actors()

car = world.get_actors().filter('vehicle.*')[0]
print(car)
print(car.get_velocity())
#print(car.get_vehicle_control())
#vehicle_control =car.get_vehicle_control()
#vehicle_control.steer = 10000


import tensorflow as tf
import numpy as np

from albumentations import (
    Compose, Normalize
)

# LOAD CAMERA
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
#camera_transform = carla.Transform(carla.Location(x=0, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=car)
image_queue = queue.Queue()
camera.listen(image_queue.put)
#rgb camera
image = image_queue.get()

# LOAD TF MODEL
PRETRAINED_MODELS = "/home/jderobot/Documents/Projects/"
#model = "20220905-164806_pilotnet_CARLA_cp.h5"
#model = "20220906-114029_pilotnet_CARLA_extreme_cases_cp.h5"
#model = "20220906-165331_pilotnet_CARLA_extreme_cases_cp.h5"
#model = "20220907-105158_pilotnet_CARLA_extreme_cases_07_09_cp.h5"
#model = "20220907-140021_pilotnet_CARLA_extreme_cases_07_09_cp.h5"
#model = "20220914-124554_pilotnet_CARLA_extreme_cases_14_09_dataset_cp.h5"
#model = "20220914-130834_pilotnet_CARLA_extreme_cases_14_09_dataset_cp.h5"
#model = "20220914-134708_pilotnet_CARLA_extreme_cases_14_09_dataset_new_crop_cp.h5"
#model = "20220914-140016_pilotnet_CARLA_extreme_cases_14_09_dataset_new_crop_cp.h5"
#model = "20220916-140706_pilotnet_CARLA_extreme_cases_16_09_dataset_new_crop_cp.h5"
#model = "20220916-154943_pilotnet_CARLA_extreme_cases_16_09_dataset_new_crop_cp.h5"
#model = "20220916-164609_pilotnet_CARLA_extreme_cases_16_09_dataset_new_crop_extreme_cases.h5"
#model = "20220919-172247_pilotnet_CARLA_extreme_cases_16_09_dataset_new_crop_extreme_cases_simplified_images_cp.h5"
#model = "20220919-184652_pilotnet_CARLA_extreme_cases_16_09_dataset_new_crop_extreme_cases_simplified_images_cp.h5"
model = "20220929-181325_pilotnet_CARLA_extreme_cases_29_09_dataset_bird_eye_random_start_point_retrained_double_cp.h5"


print('***********************************************************************************************')
net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
#net = tf.saved_model.load(PRETRAINED_MODELS + model)
print('***********************************************************************************************')


while True:
	# Predict
	image = image_queue.get()
	'''
	print(image)
	print(image.raw_data)
	'''
	array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image.height, image.width, 4)) # RGBA format
	image = array[:, :, :3] #  Take only RGB
	image = image[:, :, ::-1] # BGR

	#print(image.shape)

	image = image[325:600, 0:800]
	#plt.imshow(image)
	#plt.show()

	img = cv2.resize(image, (200, 66))
	frame = img
	hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
	# Threshold of blue in HSV space
	lower_blue = np.array([10, 100, 20])
	upper_blue = np.array([25, 255, 255])


	# preparing the mask to overlay
	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	# The black region in the mask has the value of 0,
	# so when multiplied with original image removes all non-blue regions
	result = cv2.bitwise_and(frame, frame, mask = mask)
	#new_images_carla_dataset.append(result)

	img = result


	AUGMENTATIONS_TEST = Compose([
	    Normalize()
	])
	image = AUGMENTATIONS_TEST(image=img)
	img = image["image"]


	img = np.expand_dims(img, axis=0)
	prediction = net.predict(img)
	prediction_w = prediction[0][1] * (1 - (-1)) + (-1)
	'''
	print(prediction)
	print(prediction_w)
	'''

	throttle = prediction[0][0]
	steer = prediction_w
	'''
	print('----')
	print(throttle)
	print(type(throttle))
	print(float(throttle))
	print(steer)
	print(float(steer))
	print(type(steer))
	print('----')
	'''
	print(float(throttle), steer)

	#car.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer))
	car.apply_control(carla.VehicleControl(throttle=0.2, steer=steer))
	#print(car.get_velocity())
	#print(car.get_vehicle_control())
	#time.sleep(5)


'''
car.apply_control(carla.VehicleControl(throttle = 1, brake = 0))
print(car.get_velocity())
print(car.get_vehicle_control())
time.sleep(5)
print(car.get_velocity())
print(car.get_vehicle_control())
time.sleep(5)
print(car.get_velocity())
print(car.get_vehicle_control())
car.apply_control(carla.VehicleControl(throttle = 0, brake = 0))
time.sleep(5)
print(car.get_velocity())
print(car.get_vehicle_control())
'''
