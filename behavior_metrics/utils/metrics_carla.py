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
import re

from bagpy import bagreader
from utils.logger import logger

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

    data_file = experiment_metrics_bag_filename.split('.bag')[0] + '/carla-ego_vehicle-speedometer.csv'
    dataframe_speedometer = pd.read_csv(data_file)
    speedometer_points = []
    for index, row in dataframe_speedometer.iterrows():
        speedometer_points.append(row)

    if len(checkpoints) > 1:
        starting_point = checkpoints[0]
        starting_point = (starting_point['pose.pose.position.x'], starting_point['pose.pose.position.y'])
        experiment_metrics['starting_point'] = starting_point
        experiment_metrics = get_distance_completed(experiment_metrics, checkpoints)
        experiment_metrics = get_average_speed(experiment_metrics, speedometer_points)
        experiment_metrics = get_collisions(experiment_metrics, collision_points)
        experiment_metrics = get_lane_invasions(experiment_metrics, lane_invasion_points)
        experiment_metrics['experiment_total_simulated_time'] = seconds_end - seconds_start
        experiment_metrics = get_position_deviation_and_effective_completed_distance(experiment_metrics, checkpoints, map_waypoints, experiment_metrics_filename, speedometer_points)
        shutil.rmtree(experiment_metrics_bag_filename.split('.bag')[0])
        return experiment_metrics
    else:
        return {}


def get_distance_completed(experiment_metrics, checkpoints):
    end_point = checkpoints[len(checkpoints) - 1]
    experiment_metrics['completed_distance'] = circuit_distance_completed(checkpoints, end_point)
    return experiment_metrics


def get_average_speed(experiment_metrics, speedometer_points):
    speedometer_points_sum = 0
    for point in speedometer_points:
        speedometer_points_sum += point.data

    experiment_metrics['average_speed'] = (speedometer_points_sum/len(speedometer_points))*3.6
    return experiment_metrics

def get_collisions(experiment_metrics, collision_points):
    experiment_metrics['collisions'] = len(collision_points)
    return experiment_metrics

def get_lane_invasions(experiment_metrics, lane_invasion_points):
    experiment_metrics['lane_invasions'] = len(lane_invasion_points)
    return experiment_metrics

def get_position_deviation_and_effective_completed_distance(experiment_metrics, checkpoints, map_waypoints, experiment_metrics_filename, speedometer):
    map_waypoints_tuples = []
    map_waypoints_tuples_x = []
    map_waypoints_tuples_y = []
    for waypoint in map_waypoints:
        if (experiment_metrics['carla_map'] == 'Carla/Maps/Town04'):
            map_waypoints_tuples_x.append(-waypoint.transform.location.x)
            map_waypoints_tuples_y.append(waypoint.transform.location.y)
            map_waypoints_tuples.append((-waypoint.transform.location.x, waypoint.transform.location.y))
        elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town06'):
            map_waypoints_tuples_x.append(waypoint.transform.location.x)
            map_waypoints_tuples_y.append(-waypoint.transform.location.y)
            map_waypoints_tuples.append((waypoint.transform.location.x, -waypoint.transform.location.y))
        else:
            map_waypoints_tuples_x.append(waypoint.transform.location.x)
            map_waypoints_tuples_y.append(waypoint.transform.location.y)
            map_waypoints_tuples.append((waypoint.transform.location.x, waypoint.transform.location.y))
            
    checkpoints_tuples = []
    checkpoints_tuples_x = []
    checkpoints_tuples_y = []
    checkpoints_speeds = []
    for i, point in enumerate(checkpoints):
        current_checkpoint = np.array([point['pose.pose.position.x'], point['pose.pose.position.y'], speedometer[i]['data']*3.6])
        if (experiment_metrics['carla_map'] == 'Carla/Maps/Town01' or experiment_metrics['carla_map'] == 'Carla/Maps/Town02'):
            checkpoint_x = (max(map_waypoints_tuples_x) + min(map_waypoints_tuples_x))-current_checkpoint[0]
            checkpoint_y = -point['pose.pose.position.y']
        elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town03' or experiment_metrics['carla_map'] == 'Carla/Maps/Town07'):
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]
        elif (experiment_metrics['carla_map'] == 'Carla/Maps/Town04'):
            checkpoint_x = -current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]
        else:
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = current_checkpoint[1]
        checkpoints_tuples_x.append(checkpoint_x)
        checkpoints_tuples_y.append(checkpoint_y)
        checkpoints_speeds.append(current_checkpoint[2])
        checkpoints_tuples.append((checkpoint_x, checkpoint_y, current_checkpoint[2]))
    
    min_dists = []
    best_checkpoint_points_x = []
    best_checkpoint_points_y = []

    covered_checkpoints = []
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
            if len(covered_checkpoints) == 0 or (len(covered_checkpoints) > 0 and covered_checkpoints[len(covered_checkpoints)-1][0] != best_checkpoint_point_x and covered_checkpoints[len(covered_checkpoints)-1][1] != best_checkpoint_point_y):
                if min_dist < 1:
                    covered_checkpoints.append((best_checkpoint_point_x, best_checkpoint_point_y))

    experiment_metrics['effective_completed_distance'] = len(covered_checkpoints)*0.5
    experiment_metrics['position_deviation_mae'] = sum(min_dists) / len(min_dists)  
    experiment_metrics['position_deviation_total_err'] = sum(min_dists)
    starting_point_map = (checkpoints_tuples_x[0], checkpoints_tuples_y[0])
    experiment_metrics['starting_point_map'] = starting_point_map
    
    create_experiment_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, best_checkpoint_points_x, best_checkpoint_points_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds)
    return experiment_metrics

def create_experiment_map(experiment_metrics, experiment_metrics_filename, map_waypoints_tuples_x, map_waypoints_tuples_y, best_checkpoint_points_x, best_checkpoint_points_y, checkpoints_tuples_x, checkpoints_tuples_y, checkpoints_speeds):
    difference_x = 0
    difference_y = 0
    starting_point_landmark = 0
    while difference_x < 1 and difference_y < 1 and starting_point_landmark < len(checkpoints_tuples_x)-1:
        difference_x = abs(checkpoints_tuples_x[starting_point_landmark] - checkpoints_tuples_x[0])
        difference_y = abs(checkpoints_tuples_y[starting_point_landmark] - checkpoints_tuples_y[0])
        if difference_x < 1 and difference_y < 1:
            starting_point_landmark += 1

    difference_x = 0
    difference_y = 0
    finish_point_landmark = len(checkpoints_tuples_x)-1
    while difference_x < 1 and difference_y < 1 and finish_point_landmark > 0:
        difference_x = abs(checkpoints_tuples_x[finish_point_landmark] - checkpoints_tuples_x[len(checkpoints_tuples_x)-1])
        difference_y = abs(checkpoints_tuples_y[finish_point_landmark] - checkpoints_tuples_y[len(checkpoints_tuples_x)-1])
        if difference_x < 1 and difference_y < 1:
            finish_point_landmark -= 1

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot()
    colors=["#00FF00", "#FF0000", "#000000"]
    ax.scatter(map_waypoints_tuples_x, map_waypoints_tuples_y, s=10, c='b', marker="s", label='Map waypoints')
    ax.scatter(best_checkpoint_points_x, best_checkpoint_points_y, s=10, c='g', marker="o", label='Map waypoints for position deviation')
    plot = ax.scatter(checkpoints_tuples_x, checkpoints_tuples_y, s=10, c=checkpoints_speeds, cmap='hot_r', marker="o", label='Experiment waypoints', vmin=0, vmax=30)
    ax.scatter(checkpoints_tuples_x[0], checkpoints_tuples_y[0], s=200, marker="o", color=colors[0], label='Experiment starting point')
    ax.scatter(checkpoints_tuples_x[starting_point_landmark], checkpoints_tuples_y[starting_point_landmark], s=100, marker="o", color=colors[2])
    ax.scatter(checkpoints_tuples_x[len(checkpoints_tuples_x)-1], checkpoints_tuples_y[len(checkpoints_tuples_x)-1], s=200, marker="o", color=colors[1], label='Experiment finish point')
    ax.scatter(checkpoints_tuples_x[finish_point_landmark], checkpoints_tuples_y[finish_point_landmark], s=100, marker="o", color=colors[2])
    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', prop={'size': 20})
    
    full_text = ''
    for key, value in experiment_metrics.items():
        print(key, value)
        full_text += ' * ' + str(key) + ' : ' + str(value) + '\n'
    plt.figtext(0.1, 0.01, full_text, wrap=True, horizontalalignment='left', fontsize=12)

    plt.grid(True)
    plt.subplots_adjust(bottom=0.4)
    plt.title(experiment_metrics['experiment_model'], fontsize=20)
    fig.savefig(experiment_metrics_filename + '.png', dpi=fig.dpi)


def get_aggregated_experiments_list(experiments_starting_time):
    current_experiment_folders = []
    root = './'
    folders = list(os.walk(root))[1:]
    for folder in folders:
        if len(folder[0].split('/')) == 2 and folder[2] and experiments_starting_time < os.stat(folder[0]).st_mtime and re.search("\./[0-9]+-[0-9]+", folder[0]):
            current_experiment_folders.append(folder)
    current_experiment_folders.sort()

    dataframes = []
    for folder in current_experiment_folders:
        try:
            r = re.compile(".*\.json")
            json_list = list(filter(r.match, folder[2])) # Read Note below
            df = pd.read_json(folder[0] + '/' + json_list[0], orient='index').T
            dataframes.append(df)
        except:
            print('Broken experiment: ' + folder[0])
            shutil.rmtree(folder[0])

    result = pd.concat(dataframes)
    result.index = result['timestamp'].values.tolist()

    return result

def get_maps_colors():
    maps_colors = {
        'Carla/Maps/Town01': 'red', 
        'Carla/Maps/Town02': 'green', 
        'Carla/Maps/Town03': 'blue', 
        'Carla/Maps/Town04': 'grey', 
        'Carla/Maps/Town05': 'black', 
        'Carla/Maps/Town06': 'pink', 
        'Carla/Maps/Town07': 'orange', 
    }
    return maps_colors

def get_color_handles():
    red_patch = mpatches.Patch(color='red', label='Map01')
    green_patch = mpatches.Patch(color='green', label='Map02')
    blue_patch = mpatches.Patch(color='blue',  label='Map03')
    grey_patch = mpatches.Patch(color='grey',  label='Map04')
    black_patch = mpatches.Patch(color='black',  label='Map05')
    pink_patch = mpatches.Patch(color='pink',  label='Map06')
    orange_patch = mpatches.Patch(color='orange',  label='Map07')
    color_handles = [red_patch, green_patch, blue_patch, grey_patch, black_patch, pink_patch, orange_patch]

    return color_handles


def get_all_experiments_aggregated_metrics(result, experiments_starting_time_str, experiments_metrics_and_titles):
    maps_colors = get_maps_colors()
    color_handles = get_color_handles()
    colors = []
    for i in result['carla_map']:
        colors.append(maps_colors[i])

    for experiment_metric_and_title in experiments_metrics_and_titles:
        fig = plt.figure(figsize=(20,10))
        result[experiment_metric_and_title['metric']].plot.bar(color=colors)
        plt.title(experiment_metric_and_title['title'])
        fig.tight_layout()
        plt.xticks(rotation=90)
        plt.legend(handles=color_handles)
        plt.savefig(experiments_starting_time_str + '/' + experiment_metric_and_title['metric'] + '.png')
        plt.close()

def get_per_model_aggregated_metrics(result, experiments_starting_time_str, experiments_metrics_and_titles):
    maps_colors = get_maps_colors()
    color_handles = get_color_handles()
    unique_experiment_models = result['experiment_model'].unique()
            
    for unique_experiment_model in unique_experiment_models:
        unique_model_experiments = result.loc[result['experiment_model'].eq(unique_experiment_model)]
        colors = []
        for i in unique_model_experiments['carla_map']:
            colors.append(maps_colors[i])

        for experiment_metric_and_title in experiments_metrics_and_titles:
            fig = plt.figure(figsize=(20,10))
            unique_model_experiments[experiment_metric_and_title['metric']].plot.bar(color=colors)
            plt.title(experiment_metric_and_title['title'] + ' with ' + unique_experiment_model)
            fig.tight_layout()
            plt.xticks(rotation=90)
            plt.legend(handles=color_handles)
            plt.savefig(experiments_starting_time_str + '/' + unique_experiment_model + '_ ' + experiment_metric_and_title['metric'] + '.png')
            plt.close()