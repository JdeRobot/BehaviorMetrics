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
import matplotlib.pyplot as plt
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


def get_metrics(experiment_metrics, experiment_metrics_bag_filename, map_waypoints, experiment_metrics_filename):
    time_counter = 5
    while not os.path.exists(experiment_metrics_bag_filename):
        time.sleep(1)
        time_counter -= 1
        if time_counter <= 0:
            ValueError(f"{experiment_metrics_bag_filename} isn't a file!")
            return {}

    try:
        bag_reader = bagreader(experiment_metrics_bag_filename)
    except rosbag.bag.ROSBagException:
        return {}

    csv_files = []
    for topic in bag_reader.topics:
        data = bag_reader.message_by_topic(topic)
        csv_files.append(data)

    data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-odometry.csv'
    dataframe_pose = pd.read_csv(data_file)
    checkpoints = []
    for index, row in dataframe_pose.iterrows():
        checkpoints.append(row)

    data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/clock.csv'
    dataframe_pose = pd.read_csv(data_file)
    clock_points = []
    for index, row in dataframe_pose.iterrows():
        clock_points.append(row)
    start_clock = clock_points[0]
    seconds_start = start_clock['clock.secs']
    seconds_end = clock_points[len(clock_points) - 1]['clock.secs']

    collision_points = []
    if '/carla/ego_vehicle/collision' in bag_reader.topics:
        data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-collision.csv'
        dataframe_collision = pd.read_csv(data_file)
        for index, row in dataframe_collision.iterrows():
            collision_points.append(row)

    lane_invasion_points = []
    if '/carla/ego_vehicle/lane_invasion' in bag_reader.topics:
        data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-lane_invasion.csv'
        dataframe_lane_invasion = pd.read_csv(data_file, on_bad_lines='skip')
        for index, row in dataframe_lane_invasion.iterrows():
            lane_invasion_points.append(row)

    if len(checkpoints) > 1:
        experiment_metrics = get_distance_completed(experiment_metrics, checkpoints)
        experiment_metrics = get_average_speed(experiment_metrics, seconds_start, seconds_end)
        experiment_metrics = get_collisions(experiment_metrics, collision_points)
        experiment_metrics = get_lane_invasions(experiment_metrics, lane_invasion_points)
        experiment_metrics = get_position_deviation(experiment_metrics, checkpoints, map_waypoints, experiment_metrics_filename)
        experiment_metrics['experiment_total_simulated_time'] = seconds_end - seconds_start
        shutil.rmtree(experiment_metrics_bag_filename.split('.bag')[0])
        return experiment_metrics
    else:
        return {}


def get_distance_completed(experiment_metrics, checkpoints):
    end_point = checkpoints[len(checkpoints) - 1]
    experiment_metrics['completed_distance'] = circuit_distance_completed(checkpoints, end_point)
    return experiment_metrics


def get_average_speed(experiment_metrics, seconds_start, seconds_end):
    if (seconds_end - seconds_start):
        experiment_metrics['average_speed'] = (experiment_metrics['completed_distance'] / (seconds_end - seconds_start))* 3.6
    else:
        experiment_metrics['average_speed'] = 0
    return experiment_metrics

def get_collisions(experiment_metrics, collision_points):
    experiment_metrics['collisions'] = len(collision_points)
    return experiment_metrics

def get_lane_invasions(experiment_metrics, lane_invasion_points):
    experiment_metrics['lane_invasions'] = len(lane_invasion_points)
    return experiment_metrics

def get_position_deviation(experiment_metrics, checkpoints, map_waypoints, experiment_metrics_filename):
    map_waypoints_tuples = []
    map_waypoints_tuples_x = []
    map_waypoints_tuples_y = []
    for waypoint in map_waypoints:
        map_waypoints_tuples_x.append(waypoint.transform.location.x)
        map_waypoints_tuples_y.append(waypoint.transform.location.y)
        map_waypoints_tuples.append((waypoint.transform.location.x, waypoint.transform.location.y))

    checkpoints_tuples = []
    checkpoints_tuples_x = []
    checkpoints_tuples_y= []
    for i, point in enumerate(checkpoints):
        current_checkpoint = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        checkpoint_x = (max(map_waypoints_tuples_x) + min(map_waypoints_tuples_x))-current_checkpoint[0]
        checkpoint_y = -point['pose.pose.position.y']
        checkpoints_tuples_x.append(checkpoint_x)
        checkpoints_tuples_y.append(checkpoint_y)
        checkpoints_tuples.append((checkpoint_x, checkpoint_y))
    
    min_dists = []
    best_checkpoint_points_x = []
    best_checkpoint_points_y = []
    for error_counter, checkpoint in enumerate(checkpoints_tuples):
        min_dist = 100
        for x, perfect_checkpoint in enumerate(map_waypoints_tuples):
            point_1 = np.array([checkpoint[0], checkpoint[1]])
            point_2 = np.array([perfect_checkpoint[0], perfect_checkpoint[1]])
            dist = (point_2 - point_1) ** 2
            dist = np.sum(dist, axis=0)
            dist = np.sqrt(dist)
            if dist < min_dist:
                min_dist = dist
                best_checkpoint = x
                best_checkpoint_point_x = point_2[0]
                best_checkpoint_point_y = point_2[1]
        best_checkpoint_points_x.append(best_checkpoint_point_x)
        best_checkpoint_points_y.append(best_checkpoint_point_y)
        if min_dist < 100:
            min_dists.append(min_dist)

    experiment_metrics['position_deviation_mae'] = sum(min_dists) / len(min_dists)  
    experiment_metrics['position_deviation_total_err'] = sum(min_dists)
    
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#00FF00", "#FF0000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='b', marker="s", label='Map waypoints')
    ax.scatter(best_checkpoint_points_x, best_checkpoint_points_y, s=10, c='g', marker="o", label='Map waypoints for position deviation')
    ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c='r', marker="o", label='Experiment waypoints')
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x)-1], checkpoints_tuples_y[len(checkpoints_tuples_x)-1], s=200, marker="o", color=colors[1], label='Experiment finish point')
    plt.legend(loc='upper left', prop={'size': 25})
    fig.savefig(experiment_metrics_filename + '.png', dpi=fig.dpi)

    return experiment_metrics