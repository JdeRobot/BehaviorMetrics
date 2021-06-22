import json
import yaml
import rosbag
import sys
import argparse
import cv2
from cv_bridge import CvBridge
import numpy as np

from utils import metrics
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QGridLayout
from matplotlib.figure import Figure
from utils.colors import Colors

bridge = CvBridge()

class MetricsWindow(QtWidgets.QMainWindow):
    def __init__(self, bag_file, x_points, y_points, first_image, bag_metadata, time_stats_metadata, lap_statistics, circuit_diameter):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QGridLayout(self._main)
        
        self.setWindowTitle("Metrics for: " + bag_file)
        self.x_points = x_points
        self.y_points = y_points
        self.first_image = first_image
        self.bag_metadata = bag_metadata
        self.time_stats_metadata = time_stats_metadata
        self.lap_statistics = lap_statistics
        self.circuit_diameter = circuit_diameter
        
        self.setup_plot()
        self.setup_image()
        self.add_labels()
        

    def setup_plot(self):
        self.fig_plot = Figure()
        self.canvas_plot = FigureCanvas(self.fig_plot)
        self.layout.addWidget(self.canvas_plot, 0, 0)
        self.addToolBar(NavigationToolbar(self.canvas_plot, self))
        self.ax_plot = self.fig_plot.subplots()
        self.scat_plot = self.ax_plot.scatter(self.x_points, self.y_points, zorder=3)
        
    def setup_image(self):
        self.fig_image = Figure()
        self.canvas_image = FigureCanvas(self.fig_image)
        self.layout.addWidget(self.canvas_image, 0, 1)
        self.addToolBar(NavigationToolbar(self.canvas_image, self))
        
        self.ax_image = self.fig_image.subplots()
        self.ax_image.imshow(self.first_image)
        self.ax_image.set_axis_off()
        
    def add_labels(self):
        label_world=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">World: </span>' + self.bag_metadata['world'])
        self.layout.addWidget(label_world)
        label_brain_path=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Brain path: </span>' + self.bag_metadata['brain_path'])
        self.layout.addWidget(label_brain_path)
        label_robot_type=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Robot type: </span>' + self.bag_metadata['robot_type'])
        self.layout.addWidget(label_robot_type)
        
        label_mean_iteration_time=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Mean iteration time: </span>' + str(self.time_stats_metadata['mean_iteration_time']))
        self.layout.addWidget(label_mean_iteration_time)
        label_mean_inference_time=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Mean inference time: </span>' + str(self.time_stats_metadata['mean_inference_time']))
        self.layout.addWidget(label_mean_inference_time)
        label_gpu_inferencing=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">GPU inferencing: </span>' + str(self.time_stats_metadata['gpu_inferencing']))
        self.layout.addWidget(label_gpu_inferencing)
        label_frame_rate=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Frame rate: </span>' + str(self.time_stats_metadata['frame_rate']))
        self.layout.addWidget(label_frame_rate)
        
        label_circuit_diameter=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Circuit diameter: </span>' + str(self.circuit_diameter))
        self.layout.addWidget(label_circuit_diameter)
        label_completed_distance=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Completed distance: </span>' + str(self.lap_statistics['completed_distance']))
        self.layout.addWidget(label_completed_distance)
        label_percentage_completed=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Percentage completed: </span>' + str(self.lap_statistics['percentage_completed']))
        self.layout.addWidget(label_percentage_completed)
        label_orientation_mae=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Orientation MAE: </span>' + str(self.lap_statistics['orientation_mae']))
        self.layout.addWidget(label_orientation_mae)
        label_orientation_total_err=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Orientation total ERROR: </span>' + str(self.lap_statistics['orientation_total_err']))
        self.layout.addWidget(label_orientation_total_err)
        
        if 'lap_seconds' in self.lap_statistics:
            label_lap_seconds=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Lap seconds: </span>' + str(self.lap_statistics['lap_seconds']))
            self.layout.addWidget(label_lap_seconds)
            label_circuit_diameter=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Circuit diameter: </span>' + str(self.lap_statistics['circuit_diameter']))
            self.layout.addWidget(label_circuit_diameter)
            label_average_speed=QLabel('<span style=" font-size:10pt; font-weight:600; color:#000000;">Average speed: </span>' + str(self.lap_statistics['average_speed']))
            self.layout.addWidget(label_average_speed)


def read_bags(bags):
    bags_checkpoints = []
    bags_metadata = []
    bags_lapdata = []
    time_stats = []
    correct_bags = 0
    for bag_file in bags:
        print('Reading bag: ' + bag_file)
        try:
            bag = rosbag.Bag(bag_file)
            bag_checkpoints = []
            for topic, point, t in bag.read_messages(topics=['/F1ROS/odom']):
                bag_checkpoints.append(point)
            bags_checkpoints.append(bag_checkpoints)

            for topic, point, t in bag.read_messages(topics=['/metadata']):

                y = yaml.load(str(point), Loader=yaml.FullLoader)
                h = json.dumps(y,indent=4)
                data = json.loads(h)
                metadata = json.loads(data['data'])
                bags_metadata.append(metadata)

            for topic, point, t in bag.read_messages(topics=['/lap_stats']):

                y = yaml.load(str(point), Loader=yaml.FullLoader)
                h = json.dumps(y,indent=4)
                data = json.loads(h)
                lapdata = json.loads(data['data'])
                bags_lapdata.append(lapdata)

            for topic, point, t in bag.read_messages(topics=['/time_stats']):

                y = yaml.load(str(point), Loader=yaml.FullLoader)
                h = json.dumps(y,indent=4)
                data = json.loads(h)
                time_stats_metadata = json.loads(data['data'])
                # first_image = np.array(time_stats_metadata['first_image'])
                time_stats.append(time_stats_metadata)

            for topic, point, t in bag.read_messages(topics=['/first_image']):
                first_image = bridge.imgmsg_to_cv2(point, desired_encoding='passthrough')

            bag.close()
            correct_bags += 1
        except Exception as excep:
            print(excep)
            print('Error in bag')

    print('Correct bags: ' + str(correct_bags))
    
    return bags_checkpoints, bags_metadata, bags_lapdata, time_stats, first_image


def show_metrics(bags, bags_checkpoints, bags_metadata, bags_lapdata, time_stats, first_image):
    experiments_statistics = []
    world_completed = {}

    for x, checkpoints in enumerate(bags_checkpoints):
        x_points = []
        y_points = []

        experiment_statistics = {'world': bags_metadata[x]['world'], 'brain_path': bags_metadata[x]['brain_path'],
                                 'robot_type': bags_metadata[x]['robot_type']}
        if bags_metadata[x]['world'] == 'simple_circuit.launch':
            perfect_lap_path = 'lap-simple-circuit.bag'
        elif bags_metadata[x]['world'] == 'many_curves.launch':
            perfect_lap_path = 'lap-many-curves.bag'
        elif bags_metadata[x]['world'] == 'montmelo_line.launch':
            perfect_lap_path = 'lap-montmelo.bag'

        perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag(perfect_lap_path)
        lap_statistics = bags_lapdata[x]
        experiment_statistics['lap_statistics'] = lap_statistics
        experiments_statistics.append(experiment_statistics)
        if lap_statistics['percentage_completed'] > 100:
            if bags_metadata[x]['world'] in world_completed and bags_metadata[x]['brain_path'] in world_completed[bags_metadata[x]['world']]:
                world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] + 1
            elif bags_metadata[x]['world'] in world_completed:
                world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = 1
            else:
                world_completed[bags_metadata[x]['world']] = {} 
                world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = 1
        if 'lap_seconds' in lap_statistics:
            print('LAP SECONDS -> ' + str(lap_statistics['lap_seconds']))
            print('CIRCUIT DIAMETER -> ' + str(lap_statistics['circuit_diameter']))
            print('AVERAGE SPEED -> ' + str(lap_statistics['average_speed']))

        for point in checkpoints:
            point_yml = yaml.load(str(point), Loader=yaml.FullLoader)
            x_points.append(point_yml['pose']['pose']['position']['x'])
            y_points.append(point_yml['pose']['pose']['position']['y'])

        qapp = QtWidgets.QApplication(sys.argv)
        app = MetricsWindow(bags[x], x_points, y_points, first_image, bags_metadata[x], time_stats[x], lap_statistics, circuit_diameter)
        app.show()
        qapp.exec_()


def main():
    parser = argparse.ArgumentParser(description='Show plots', epilog='Enjoy the program! :)')

    parser.add_argument('-b',
                        '--bags',
                        action='append',
                        type=str,
                        required=False,
                        help='{}Path to ROS Bag file.{}'.format(
                            Colors.OKBLUE, Colors.ENDC))
    
    args = parser.parse_args()
    config_data = {'bags': None}
    if args.bags:
        config_data['bags'] = args.bags
    
    bags_checkpoints, bags_metadata, bags_lapdata, time_stats, first_image = read_bags(config_data['bags'])
    show_metrics(config_data['bags'], bags_checkpoints, bags_metadata, bags_lapdata, time_stats, first_image)
    
if __name__ == "__main__":
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    main()
