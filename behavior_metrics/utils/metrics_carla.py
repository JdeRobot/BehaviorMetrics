#!/usr/bin/env python

"""This module contains the metrics manager.
This module is in charge of generating metrics for a brain execution.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np
import shutil
import time
import os
import rosbag

from datetime import datetime
from bagpy import bagreader
from utils.logger import logger


def circuit_distance_completed(checkpoints, lap_point):
    previous_point = []
    diameter = 0
    for i, point in enumerate(checkpoints):
        current_point = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        if i != 0:
            dist = (previous_point - current_point) ** 2
            dist = np.sum(dist, axis=0)
            dist = np.sqrt(dist)
            diameter += dist
        if point is lap_point:
            break
        previous_point = current_point
    return diameter


def get_metrics(stats_filename):
    empty_metrics = {
        "completed_distance": 0, 
        "average_speed": 0,
        "experiment_total_simulated_time": 0
    }
    experiment_metrics = {}
    
    time_counter = 5
    while not os.path.exists(stats_filename):
        time.sleep(1)
        time_counter -= 1
        if time_counter <= 0:
            ValueError(f"{stats_filename} isn't a file!")
            return empty_metrics

    try:
        bag_reader = bagreader(stats_filename)
    except rosbag.bag.ROSBagException:
        return empty_metrics

    csv_files = []
    for topic in bag_reader.topics:
        data = bag_reader.message_by_topic(topic)
        csv_files.append(data)

    data_file = stats_filename.split('.bag')[0] + '/carla-ego_vehicle-odometry.csv'
    dataframe_pose = pd.read_csv(data_file)
    checkpoints = []
    for index, row in dataframe_pose.iterrows():
        checkpoints.append(row)

    data_file = stats_filename.split('.bag')[0] + '/clock.csv'
    dataframe_pose = pd.read_csv(data_file)
    clock_points = []
    for index, row in dataframe_pose.iterrows():
        clock_points.append(row)
    start_clock = clock_points[0]
    seconds_start = start_clock['clock.secs']
    seconds_end = clock_points[len(clock_points) - 1]['clock.secs']

    collision_points = []
    if '/carla/ego_vehicle/collision' in bag_reader.topics:
        data_file = stats_filename.split('.bag')[0] + '/carla-ego_vehicle-collision.csv'
        dataframe_collision = pd.read_csv(data_file)
        for index, row in dataframe_collision.iterrows():
            collision_points.append(row)

    lane_invasion_points = []
    if '/carla/ego_vehicle/lane_invasion' in bag_reader.topics:
        data_file = stats_filename.split('.bag')[0] + '/carla-ego_vehicle-lane_invasion.csv'
        dataframe_lane_invasion = pd.read_csv(data_file, on_bad_lines='skip')
        for index, row in dataframe_lane_invasion.iterrows():
            lane_invasion_points.append(row)

    if len(checkpoints) > 1:
        experiment_metrics = get_distance_completed(experiment_metrics, checkpoints)
        experiment_metrics = get_average_speed(experiment_metrics, seconds_start, seconds_end)
        experiment_metrics = get_collisions(experiment_metrics, collision_points)
        experiment_metrics = get_lane_invasions(experiment_metrics, lane_invasion_points)
        experiment_metrics = get_position_deviation(experiment_metrics, checkpoints)
        experiment_metrics['experiment_total_simulated_time'] = seconds_end - seconds_start
        logger.info('* Experiment total simulated time ---> ' + str(experiment_metrics['experiment_total_simulated_time']) + ' s')
        shutil.rmtree(stats_filename.split('.bag')[0])
        return experiment_metrics
    else:
        return empty_metrics


def get_distance_completed(experiment_metrics, checkpoints):
    end_point = checkpoints[len(checkpoints) - 1]
    experiment_metrics['completed_distance'] = circuit_distance_completed(checkpoints, end_point)
    logger.info('* Completed distance ---> ' + str(experiment_metrics['completed_distance']) + ' m')
    return experiment_metrics


def get_average_speed(experiment_metrics, seconds_start, seconds_end):
    if (seconds_end - seconds_start):
        experiment_metrics['average_speed'] = (experiment_metrics['completed_distance'] / (seconds_end - seconds_start))* 3.6
    else:
        experiment_metrics['average_speed'] = 0
    logger.info('* Average speed ---> ' + str(experiment_metrics['average_speed']) + ' km/h')
    return experiment_metrics

def get_collisions(experiment_metrics, collision_points):
    experiment_metrics['collisions'] = len(collision_points)
    logger.info('* Collisions ---> ' + str(experiment_metrics['collisions']))
    return experiment_metrics

def get_lane_invasions(experiment_metrics, lane_invasion_points):
    experiment_metrics['lane_invasions'] = len(lane_invasion_points)
    logger.info('* Lane invasions ---> ' + str(experiment_metrics['lane_invasions']))
    return experiment_metrics

def get_position_deviation(experiment_metrics, checkpoints):
    '''
        checkpoints --> POSE


    '''
    import carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) # seconds
    world = client.get_world()
    
    print('---------------------------------------------------------------------------------------------------')
    import random
    m = world.get_map()
    print(m.get_spawn_points())
    waypoint = m.get_waypoint(m.get_spawn_points()[0].location)
    
    print('---------------------------------------------------------------------------------------------------')
    print(waypoint)
    print('---------------------------------------------------------------------------------------------------')
    print(waypoint.next(1.5))
    print('---------------------------------------------------------------------------------------------------')
    
    waypoint = random.choice(waypoint.next(1.5))
    print(waypoint)
    print('---------------------------------------------------------------------------------------------------')
    map_waypoints = m.generate_waypoints(1.5)
    
    print(len(map_waypoints))

    new_checkpoints = []
    for i, point in enumerate(checkpoints):
        current_checkpoint = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        new_checkpoint = np.array([(396.32-1.96)-current_checkpoint[0], point['pose.pose.position.y']*280/-280])
        new_checkpoints.append(new_checkpoint)

    print(new_checkpoints)

    return experiment_metrics
