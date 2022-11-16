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

import random

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
import csv
import math
import time

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


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


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

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    f = open('../../carla_dataset_test_24_10_anticlockwise_full_image_2/dataset.csv', 'a')
    # create the csv writer
    writer = csv.writer(f)
    try: 
        df = pandas.read_csv('../../carla_dataset_test_24_10_anticlockwise_full_image_2/dataset.csv')
        batch_number = int(df.iloc[-1]['batch']) + 1
    except Exception as ex:
        #fkjfbdjkhfsd
        batch_number = 0
        header = ['batch', 'image_id', 'timestamp', 'throttle', 'steer', 'brake', 'location_x', 'location_y']
        writer.writerow(header)

    print(batch_number)
    frame_number = 0
    start = time.time()


    try:
        m = world.get_map()
        #start_pose = random.choice(m.get_spawn_points())
        start_pose = carla.Transform(carla.Location(x=-2.0, y=307, z=0.1), carla.Rotation(pitch=0, yaw=90, roll=0))
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
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        # Remove traffic lights and traffic limits
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

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return

                #print('----------------')
                #print(vehicle.get_transform())
                traffic_manager = client.get_trafficmanager()
                vehicle.set_autopilot(True)
                route = ["Straight", "Straight", "Straight", "Straight", "Straight",
                "Straight", "Straight", "Straight", "Straight", "Straight",
                "Straight", "Straight", "Straight", "Straight", "Straight",
                "Straight", "Straight", "Straight", "Straight", "Straight",
                "Straight", "Straight", "Straight", "Straight", "Straight",
                "Straight", "Straight", "Straight", "Straight", "Straight"]
                traffic_manager.set_route(vehicle, route)
                clock.tick()
                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=10.0)

                #print(vehicle.get_transform())
                #print(vehicle.get_control())
                #print('----------------')

                ############################################################
                vehicle_control = vehicle.get_control()
                vehicle_location = vehicle.get_location()
                speed = vehicle.get_velocity()
                vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)

                end = time.time()
                
                if (vehicle_speed > 0.0 and vehicle.get_location().z < 0.01 and (end - start) > 6):
                    print('entra')
                    #print(vehicle_speed)
                    #print(vehicle.get_transform())
                    #image = image_queue.get()
                    image = image_rgb
                    cc = carla.ColorConverter.Raw
                    image.save_to_disk('../../carla_dataset_test_24_10_anticlockwise_full_image_2/' + str(batch_number) + '_' +  str(frame_number) + '.png', cc)
                    # write a row to the csv file
                    row = [batch_number, str(batch_number) + '_' + str(frame_number) + '.png', time.time(), vehicle_control.throttle, vehicle_control.steer, vehicle_control.brake, vehicle_location.x, vehicle_location.y]
                    writer.writerow(row)
                    frame_number += 1
                    #print(vehicle_control)
                else:
                    #pass
                    print('nop')
                ############################################################3

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                #draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

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