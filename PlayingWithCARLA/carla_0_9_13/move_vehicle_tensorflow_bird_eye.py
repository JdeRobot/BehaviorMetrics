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
import tensorflow as tf
import numpy as np

from albumentations import (
    Compose, Normalize
)


#import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
world = client.get_world()

time.sleep(2)

birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=150, height=300),
    pixels_per_meter=4,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
)

birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=100, height=300),
    pixels_per_meter=10,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
)

world.get_actors()
car = world.get_actors().filter('vehicle.*')[0]

PRETRAINED_MODELS = "../../../"
#model = "20220920-175541_pilotnet_CARLA_extreme_cases_20_09_dataset_bird_eye_cp.h5"
#model = "20220921-130758_pilotnet_CARLA_extreme_cases_20_09_dataset_bird_eye_only_extreme.h5"
#model = "20220921-154633_pilotnet_CARLA_extreme_cases_20_09_dataset_bird_eye_only_extreme_cp.h5"
#model = "20220921-173038_pilotnet_CARLA_extreme_cases_20_09_dataset_bird_eye_only_extreme_only_extreme_cp.h5"
#model = "20220928-144619_pilotnet_CARLA_extreme_cases_20_09_dataset_bird_eye_random_start_point.h5"
#model = "20220928-162449_pilotnet_CARLA_extreme_cases_28_09_dataset_bird_eye_random_start_point_300_epochs_cp.h5"
#model = "20220929-164843_pilotnet_CARLA_extreme_cases_29_09_dataset_bird_eye_random_start_point_retrained_cp.h5"
#model = "20220930-105720_pilotnet_CARLA_28_09_dataset_bird_eye_random_start_point_300_epochs_no_flip_cp.h5"
#model = "20220930-130349_pilotnet_CARLA_28_09_dataset_bird_eye_random_start_point_300_epochs_no_flip_retrained.h5"
#model = "20220930-153914_pilotnet_CARLA_28_09_dataset_bird_eye_random_start_point_300_epochs_no_flip_1_output_cp.h5"
#model = "20221003-131905_pilotnet_CARLA_28_09_dataset_bird_eye_random_start_point_300_epochs_no_flip_1_output_cp.h5"
'''
model = "20221003-160817_pilotnet_CARLA_28_09_dataset_bird_eye_random_start_point_300_epochs_no_flip_3_output_cp.h5"
model = "20221004-180428_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_3_output_more_data_cp.h5"
model = "20221005-091607_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_3_output_more_data_cp.h5"
model_w = "20221005-105821_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_1_output_more_data_cp.h5"
model = "20221005-120932_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_2_output_more_data_cp.h5"

model_v = "20221005-150027_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_1_output_V_more_data_cp.h5"

model = "20221005-184041_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_3_output_more_more_data_cp.h5"

model = "20221007-175421_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_3_output_more_more_data_extreme_cases_cp.h5"
'''
model = "20221007-182407_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_3_output_more_more_data_extreme_cases_cp.h5"
#model = "20221010-090523_xception_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_3_outputs_more_more_data_cp.h5"

model = "20221017-110327_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_cp.h5"
model = "20221017-111655_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_cp.h5"
model = "20221017-113220_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_cp.h5"
model = "20221017-134410_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_extreme_cases_cp.h5"
model = "20221017-144828_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_more_extreme_cases_cp.h5"
model = "20221021-154936_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_new_extreme_more_more_cp.h5"


#model_b = "20221010-102610_pilotnet_CARLA_04_10_dataset_bird_eye_random_start_point_300_epochs_no_flip_1_output_B_more_more_data_extreme_cases_cp.h5"
print('***********************************************************************************************')
#net_v = tf.keras.models.load_model(PRETRAINED_MODELS + model_v)
#net = tf.keras.models.load_model(PRETRAINED_MODELS + model_w)
#net_b = tf.keras.models.load_model(PRETRAINED_MODELS + model_b)


net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
print('***********************************************************************************************')

try:
    while True:
        #start = time.time()

        # Input for your model - call it every simulation step
        # returned result is np.ndarray with ones and zeros of shape (8, height, width)
        birdview = birdview_producer.produce(
            agent_vehicle=car  # carla.Actor (spawned vehicle)
        )

        image = BirdViewProducer.as_rgb(birdview)
        #image_shape=(66, 200)
        #image_shape=(200, 66)
        image_shape=(50, 150)
        #image_shape=(71, 150)
        img = cv2.resize(image, image_shape)

        AUGMENTATIONS_TEST = Compose([
            Normalize()
        ])
        image = AUGMENTATIONS_TEST(image=img)
        img = image["image"]


        img = np.expand_dims(img, axis=0)
        prediction = net.predict(img)
        #print(prediction)
        #prediction_v = net_v.predict(img)
        #prediction_b = net_b.predict(img)
        #print(prediction_v, prediction)
        '''
        prediction_w = prediction[0][1] * (1 - (-1)) + (-1)
        throttle = prediction[0][0]
        steer = prediction_w
        print(throttle, steer)
        '''
        prediction_w = prediction[0][1] * (1 - (-1)) + (-1)
        throttle = prediction[0][0]
        break_command = prediction[0][2]
        #if throttle < 0.3:
        #    throttle = 0.5
        #break_command = prediction[0][2]
        steer = prediction_w
        #Sprint(throttle, steer, break_command)
        #print(throttle, steer)
        #print(car.get_velocity().steer)
        vehicle_control = car.get_control()
        #print(vehicle_control.steer)
        speed = car.get_velocity()
        
        #print(speed)
        
        if (abs(speed.x) > 6 or abs(speed.y) > 6):
            #print('SLOW DOWN!')
            car.apply_control(carla.VehicleControl(brake=float(1.0)))
        else:
            if (speed.x > 0.01 or speed.y > 0.01) and break_command > float(0.1):
                #print('1')
                car.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer, brake=float(break_command)))
            else:
                car.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer, brake=float(0.0)))
        
        #car.apply_control(carla.VehicleControl(throttle=0.5, steer=steer, brake=float(0.0)))
        #print(steer)
        #car.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer, brake=float(break_command)))
        #car.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer))
        #car.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer, brake=float(0.0)))
        ##car.apply_control(carla.VehicleControl(throttle=1.0, steer=steer, brake=float(0.01)))

        #end = time.time()
        #print(end)
        #print(end - start)
except KeyboardInterrupt:
    pass

# close the file
f.close()


