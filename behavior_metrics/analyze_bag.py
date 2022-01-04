import os
import json
import yaml
import rosbag
import sys
import argparse
import cv2
from cv_bridge import CvBridge
import numpy as np
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Analyze Rosbags and Generate Plots', epilog='Enjoy the program! :)')

    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help='Path to ROS Bag file directory.')

    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=True,
                        help='Output to plots directory.')

    args = parser.parse_args()

    bridge = CvBridge()

    baginput = args.input
    output = args.output

    all_data = {}

    for root, dirs, files in os.walk(baginput):
        print("Total Number of Bags to Read from {}: {}".format(root, len(files)))
        for name in files:

            bag_file = os.path.join(root, name)

            print('Reading bag: ' + bag_file)

            bag_name = (bag_file.split('/')[-1]).split('.')[0]

            try:
                bag = rosbag.Bag(bag_file)
                x_points = []
                y_points = []
                for topic, point, t in bag.read_messages(topics=['/F1ROS/odom']):
                    point_yml = yaml.load(str(point), Loader=yaml.FullLoader)
                    x_points.append(point_yml['pose']['pose']['position']['x'])
                    y_points.append(point_yml['pose']['pose']['position']['y'])

                for topic, point, t in bag.read_messages(topics=['/metadata']):
                    y = yaml.load(str(point), Loader=yaml.FullLoader)
                    h = json.dumps(y, indent=4)
                    data = json.loads(h)
                    metadata = json.loads(data['data'])

                for topic, point, t in bag.read_messages(topics=['/lap_metrics']):
                    y = yaml.load(str(point), Loader=yaml.FullLoader)
                    h = json.dumps(y, indent=4)
                    data = json.loads(h)
                    lapdata = json.loads(data['data'])

                for topic, point, t in bag.read_messages(topics=['/time_stats']):
                    y = yaml.load(str(point), Loader=yaml.FullLoader)
                    h = json.dumps(y, indent=4)
                    data = json.loads(h)
                    time_stats_metadata = json.loads(data['data'])

                for topic, point, t in bag.read_messages(topics=['/first_image']):
                    first_image = bridge.imgmsg_to_cv2(point, desired_encoding='passthrough')

                world = metadata['world'].split('.')[0]

                if world not in all_data.keys():
                    all_data[world] = {}
                    all_data[world]['percentage_completed'] = []
                    all_data[world]['completed_distance'] = []
                    all_data[world]['lap_seconds'] = []
                    all_data[world]['circuit_diameter'] = []
                    all_data[world]['average_speed'] = []
                    all_data[world]['image'] = {}
                    all_data[world]['image']['first_images'] = []
                    all_data[world]['image']['path_x'] = []
                    all_data[world]['image']['path_y'] = []

                all_data[world]['completed_distance'].append(lapdata['completed_distance'])
                all_data[world]['percentage_completed'].append(lapdata['percentage_completed'])
                all_data[world]['image']['first_images'].append(first_image)
                all_data[world]['image']['path_x'].append(x_points)
                all_data[world]['image']['path_y'].append(y_points)

                if 'lap_seconds' in lapdata:
                    all_data[world]['lap_seconds'].append(lapdata['lap_seconds'])
                    all_data[world]['circuit_diameter'].append(lapdata['circuit_diameter'])
                    all_data[world]['average_speed'].append(lapdata['average_speed'])
                else:
                    all_data[world]['lap_seconds'].append(0.0)
                    all_data[world]['circuit_diameter'].append(0.0)
                    all_data[world]['average_speed'].append(0.0)

                bag.close()

            except Exception as excep:
                print(excep)
                print('Error in bag')

    for world in all_data.keys():
        directory = output + 'bag_analysis_plots/' + world
        if not os.path.exists(directory):
            os.makedirs(directory + '/' + 'first_images')
            os.makedirs(directory + '/' + 'performances')
            os.makedirs(directory + '/' + 'path_followed')

        for key in all_data[world].keys():

            if key == 'image':
                images = all_data[world][key]['first_images']
                all_path_x = all_data[world][key]['path_x']
                all_path_y = all_data[world][key]['path_y']
                for it in range(len(images)):
                    cv2.imwrite(directory + '/' + 'first_images/Run_' + str(it + 1) + '.png', images[it])

                    fig = plt.figure(figsize=(10, 5))
                    plt.scatter(all_path_x[it], all_path_y[it], zorder=3)
                    plt.ylabel(key)
                    plt.title('Path followed in "{}" circuit'.format(world))
                    plt.savefig(directory + '/' + 'path_followed/Run_' + str(it + 1) + '.png')
                    plt.close()

            else:
                plotData = all_data[world][key]
                labels = []
                for it in range(len(plotData)):
                    labels.append('Run_' + str(it + 1))
                fig = plt.figure(figsize=(10, 5))
                plt.bar(labels, plotData, color='maroon', width=0.4)
                plt.ylabel(key)
                plt.title('Performance in "{}" circuit with metric "{}"'.format(world, key))
                plt.savefig(directory + '/' + 'performances/' + key + '.png')
                plt.close()
