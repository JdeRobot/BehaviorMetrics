#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
import random
import math
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import cv2
from albumentations import (
    Compose, Normalize
)
from gradcam import GradCAM

mean_step_time = []

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick(10)
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False, location=(0,0), is_black_space=False):
    try:
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
    except:
        if is_black_space:
            array = image
        else:
            array = image
            array = cv2.resize(array, (200, 600))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, location)


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    actor_list = []
    pygame.init()

    #display = pygame.display.set_mode((800*2+200, 600),pygame.HWSURFACE | pygame.DOUBLEBUF)
    display = pygame.display.set_mode((800+400, 600),pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())

        # Town 1
        # anticlockwise
        #start_pose = carla.Transform(carla.Location(x=-2.0, y=307, z=1.37), carla.Rotation(pitch=0, yaw=90, roll=0))
        #start_pose = carla.Transform(carla.Location(x=-2.0, y=5, z=1.37), carla.Rotation(pitch=0, yaw=90, roll=0))
        #start_pose = carla.Transform(carla.Location(x=-2.0, y=280, z=1.37), carla.Rotation(pitch=0, yaw=90, roll=0))
        # clockwise
        #start_pose = carla.Transform(carla.Location(x=2.0, y=200, z=1.37), carla.Rotation(pitch=0, yaw=-90, roll=0))

        # Town 2
        # anticlockwise
        #start_pose = carla.Transform(carla.Location(x=-7.5, y=200, z=1.37), carla.Rotation(pitch=0, yaw=90, roll=0))
        # clockwise
        #start_pose = carla.Transform(carla.Location(x=-4, y=240, z=1.37), carla.Rotation(pitch=0, yaw=-90, roll=0))

        # Town 3
        # anticlockwise
        #start_pose = carla.Transform(carla.Location(x=13.2, y=208, z=1.37), carla.Rotation(pitch=0, yaw=0, roll=0))
        # clockwise
        #start_pose = carla.Transform(carla.Location(x=13.2, y=194, z=1.37), carla.Rotation(pitch=0, yaw=180, roll=0))

        # Town 04
        # anticlockwise
        #start_pose = carla.Transform(carla.Location(x=16.6, y=195.4, z=1.37), carla.Rotation(pitch=0, yaw=-90, roll=0))
        # clockwise
        #start_pose = carla.Transform(carla.Location(x=14.5, y=-209.4, z=1.37), carla.Rotation(pitch=0, yaw=-90, roll=0))

        # Town 05
        #start_pose = carla.Transform(carla.Location(x=51.2, y=-186.3, z=1.37), carla.Rotation(pitch=0, yaw=0, roll=0))

        # Town 06
        #start_pose = carla.Transform(carla.Location(x=672.4, y=112.6, z=1.37), carla.Rotation(pitch=0, yaw=-90, roll=0))

        # Town 7
        # anticlockwise
        #start_pose = carla.Transform(carla.Location(x=14.0, y=63.0, z=1.37), carla.Rotation(pitch=0, yaw=0, roll=0))
        #start_pose = carla.Transform(carla.Location(x=72.3, y=-7.2, z=1.37), carla.Rotation(pitch=0, yaw=-60, roll=0))
        #start_pose = carla.Transform(carla.Location(x=72.3, y=-7.2, z=1.37), carla.Rotation(pitch=0, yaw=-60, roll=0))
        start_pose = carla.Transform(carla.Location(x=14.3, y=-237.3, z=1.37), carla.Rotation(pitch=0, yaw=0, roll=0))


        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        blueprint = world.get_blueprint_library().filter('vehicle')[0]
        

        vehicle = world.spawn_actor(
            blueprint,
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(True)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        #birdview_producer = BirdViewProducer(
        #    client,  # carla.Client
        #    target_size=PixelDimensions(width=100, height=300),
        #    pixels_per_meter=10,
        #    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
        #)

        birdview_producer = BirdViewProducer(
            client,  # carla.Client
            target_size=PixelDimensions(width=100, height=300),
            pixels_per_meter=10,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY
        )
        PRETRAINED_MODELS = "../../../"
        #model = "20221017-144828_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_more_extreme_cases_cp.h5"
        model = "20221019-140458_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_more_extreme_cases_both_directions_cp.h5"
        #model = "20221019-155549_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_more_extreme_cases_both_directions_more_cp.h5"
        model = "20221021-095950_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_cp.h5"
        #model = "20221021-114545_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_extreme_cp.h5"
        model = "20221021-140015_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_new_extreme_cp.h5"
        model = "20221021-143934_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_new_extreme_more_cp.h5"
        model = "20221021-150901_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_new_extreme_more_more_cp.h5"
        model = "20221021-154936_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_new_extreme_more_more_cp.h5"  # BEST MODEL!


        model = "20221031-093726_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_new_extreme_more_more_towns_cp.h5"
        model = "20221031-171738_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_new_extreme_more_more_towns_3_5_cp.h5"

        model = "20221031-175645_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_both_directions_new_bird_eye_view_new_extreme_more_more_towns_3_5_7_cp.h5"

        net = tf.keras.models.load_model(PRETRAINED_MODELS + model)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=10) as sync_mode:
            while True:
                if should_quit():
                    return
                start = time.time()
                clock.tick()
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=10.0)

                birdview = birdview_producer.produce(
                    agent_vehicle=vehicle  # carla.Actor (spawned vehicle)
                )

                image = BirdViewProducer.as_rgb(birdview)
                image_shape=(50, 150)
                img_base = cv2.resize(image, image_shape)
                AUGMENTATIONS_TEST = Compose([
                    Normalize()
                ])
                image = AUGMENTATIONS_TEST(image=img_base)
                img = image["image"]


                img = np.expand_dims(img, axis=0)
                prediction = net.predict(img)
                throttle = prediction[0][0]
                steer = prediction[0][1] * (1 - (-1)) + (-1)
                break_command = prediction[0][2]
                speed = vehicle.get_velocity()
                vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)
                acceleration = vehicle.get_acceleration()
                vehicle_acceleration = 3.6 * math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
                vehicle_location = vehicle.get_location()
                #print(prediction)
                
                #vehicle.set_autopilot(True)

                '''
                if vehicle_speed > 20:
                    #print('SLOW DOWN!')
                    vehicle.apply_control(carla.VehicleControl(throttle=float(0), steer=steer, brake=float(1.0)))
                else:
                    if vehicle_speed > 20 and break_command > float(0.01):
                        #print('1')
                        vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer, brake=float(break_command)))
                    else:
                        vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer, brake=float(0.0)))
                '''
                if vehicle_speed > 20:
                    #print('SLOW DOWN!')
                    vehicle.apply_control(carla.VehicleControl(throttle=float(0), steer=steer, brake=float(1.0)))
                else:
                    if vehicle_speed < 5:
                        vehicle.apply_control(carla.VehicleControl(throttle=float(1.0), steer=steer, brake=float(0.0)))
                    elif vehicle_speed > 20 and break_command > float(0.01):
                        #print('1')
                        vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer, brake=float(break_command)))
                    else:
                        vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer, brake=float(0.0)))
                

                #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=float(0.0)))
                #print(throttle, steer, break_command)
                #print(vehicle.get_control())

                i = np.argmax(prediction[0])
                cam = GradCAM(net, i)
                heatmap = cam.compute_heatmap(img)
                heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
                
                #print(original_image.shape)
                #print(resized_image.shape)
                #print(heatmap.shape)
                (heatmap, output) = cam.overlay_heatmap(heatmap, img_base, alpha=0.5)

                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                #draw_image(display, image_semseg, blend=True, location=(800,0))
                #draw_image(display, img_base, blend=False, location=(1600,0))
                draw_image(display, img_base, blend=False, location=(800,0))
                draw_image(display, output, blend=False, location=(1000,0))
                draw_image(display, np.zeros((160,300, 3)), blend=False, location=(0,0), is_black_space=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                
                if vehicle_speed > 20:
                    display.blit(
                        font.render('Speed: ' + str(round(vehicle_speed, 2)) + ' m/s', True, (255, 0, 0)),
                        (22, 46))
                else:
                    display.blit(
                        font.render('Speed: ' + str(round(vehicle_speed, 2)) + ' m/s', True, (255, 255, 255)),
                        (22, 46))
                if mean_step_time != []:
                    display.blit(
                        font.render('Mean step time: ' + str(round(sum(mean_step_time) / len(mean_step_time), 3)), True, (255, 255, 255)),
                        (22, 64))
                if vehicle_acceleration > 10.0:
                    display.blit(
                        font.render('Acceleration: ' + str(round(vehicle_acceleration, 2)) + ' m/s^2', True, (255, 0, 0)),
                        (22, 82))
                else:
                    display.blit(
                        font.render('Acceleration: ' + str(round(vehicle_acceleration, 2)) + ' m/s^2', True, (255, 255, 255)),
                        (22, 82))
                
                display.blit(
                    font.render('Throttle: ' + str(round(throttle, 2)) + ' Steer: ' + str(round(steer, 2)) + ' Break: ' + str(round(break_command, 2)), True, (255, 255, 255)),
                    (22, 100))

                display.blit(
                    font.render('Position: X=' + str(round(vehicle_location.x, 2)) + ' Y= ' + str(round(vehicle_location.y, 2)), True, (255, 255, 255)),
                    (22, 118))

                display.blit(
                    font.render('World: ' + str(m.name), True, (255, 255, 255)),
                    (22, 136))

                pygame.display.flip()
                end = time.time()
                mean_step_time.append(end - start)
                #print(sum(mean_step_time) / len(mean_step_time))
                #print(end - start)
                #time.sleep(0.5)
    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')