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


def get_metrics(stats_filename, map_waypoints):
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
        experiment_metrics = get_position_deviation(experiment_metrics, checkpoints, map_waypoints)
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

def get_position_deviation(experiment_metrics, checkpoints, map_waypoints):
    waypoints_x = []
    waypoints_y = []
    for waypoint in map_waypoints:
        waypoints_x.append(waypoint.transform.location.x)
        waypoints_y.append(waypoint.transform.location.y)

    new_checkpoints = []
    new_checkpoints_x = []
    new_checkpoints_y= []
    for i, point in enumerate(checkpoints):
        current_checkpoint = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])    
        new_checkpoints_x.append((max(waypoints_x) + min(waypoints_x))-current_checkpoint[0])
        new_checkpoints_y.append(-point['pose.pose.position.y'])

    print('--------------------------------------------')
    print(type(map_waypoints))
    print(type(checkpoints))
    print('--------------------------------------------')
    df = pd.DataFrame([waypoints_x, waypoints_y]).transpose()
    df.columns = ['waypoints_x', 'waypoints_y']
    df.to_csv('waypoints.csv')

    df = pd.DataFrame([new_checkpoints_x, new_checkpoints_y]).transpose()
    df.columns = ['new_checkpoints_x', 'new_checkpoints_x']
    df.to_csv('new_checkpoints.csv')
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#0000FF", "#00FF00", "#FF0066"]
    ax.scatter(waypoints_x, waypoints_y, s=10, c='b', marker="s", label='Circuit waypoints')
    ax.scatter(new_checkpoints_x[0], new_checkpoints_y[0], s=200, marker="o", color=colors[1])
    ax.scatter(new_checkpoints_x, new_checkpoints_y, s=10, c='r', marker="o", label='Experiment')
    plt.legend(loc='upper left', prop={'size': 25})
    plt.show()
    '''
    ###############################################3

    from scipy.optimize import fmin, dual_annealing
    from scipy.interpolate import CubicSpline

    min_dists = []

    # Get list of points
    point_x = []
    point_y = []
    point_t = []
    for x, checkpoint in enumerate(checkpoints):
        point_x.append(checkpoint['pose.pose.position.x'])
        point_y.append(checkpoint['pose.pose.position.y'])
        point_t.append(x)

    # Generate a natural spline from points
    spline_x = CubicSpline(point_t, point_x, bc_type='natural')
    spline_y = CubicSpline(point_t, point_y, bc_type='natural')

    #print(len(spline_x))
    #print(len(spline_y))

    # Rotate the x and y to start according to checkpoints
    min_dist = 100
    index_t = -1
    perfect_x = []
    perfect_y = []
    for i, waypoint in enumerate(map_waypoints):
        x = waypoint.transform.location.x
        y = waypoint.transform.location.y
        #x = waypoint.transform.location.x
        #y = waypoint.transform.location.y
        perfect_x.append(x)
        perfect_y.append(y)

        dist = np.sqrt((point_x[0] - x) ** 2 + (point_y[0] - y) ** 2)
        if min_dist > dist:
            min_dist = dist
            index_t = i

    perfect_x = perfect_x[index_t:] + perfect_x[:index_t]
    perfect_y = perfect_y[index_t:] + perfect_y[:index_t]


    print(len(perfect_x))
    print(len(perfect_y))


    # Iterate through checkpoints and calculate minimum distance
    previous_t = 0
    perfect_index = 0
    count_same_t = 0
    while True:
        x = perfect_x[perfect_index]
        y = perfect_y[perfect_index]

        point = np.array([x, y])
        distance_function = lambda t: np.sqrt((point[0] - spline_x(t)) ** 2 + (point[1] - spline_y(t)) ** 2)

        # Local Optimization for minimum distance
        current_t = fmin(distance_function, np.array([previous_t]), disp=False)[0]
        min_dist = distance_function(current_t)

        # Global Optimization if minimum distance is greater than expected
        # OR
        # at start checkpoints, since the car may be at start position during initialization (NN Brain)
        if min_dist > 1 or perfect_index in [0, 1, 2]:
            min_bound = previous_t
            max_bound = previous_t + 100
            current_t = dual_annealing(distance_function, bounds=[(min_bound, max_bound)]).x[0]
            min_dist = distance_function(current_t)

        # Two termination conditions:
        # 1. Loop only till all the available points
        if current_t > point_t[-1] - 1:
            break

        # 2. Terminate when converging to same point on spline
        if abs(current_t - previous_t) < 0.01:
            count_same_t += 1
            if count_same_t > 3:
                logger.info("Unexpected Behavior: Converging to same point")
                break
        else:
            count_same_t = 0

        previous_t = current_t
        min_dists.append(1000 ** min_dist)
        perfect_index = (perfect_index + 1) % len(perfect_x)

    if len(min_dists):
        experiment_metrics['position_deviation_mae'] = sum(min_dists) / len(min_dists)
    else:
        experiment_metrics['position_deviation_mae'] = 0

    print(min_dists)
        
    experiment_metrics['position_deviation_total_err'] = sum(min_dists)
    logger.info('* Position deviation MAE ---> ' + str(experiment_metrics['position_deviation_mae']))
    logger.info('* Position deviation total error ---> ' + str(experiment_metrics['position_deviation_total_err']))

    return experiment_metrics