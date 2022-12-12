import json
import yaml
import rosbag
import sys
import argparse
import cv2
from cv_bridge import CvBridge
import numpy as np

from utils import metrics_gazebo
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QGridLayout
from matplotlib.figure import Figure
from utils.colors import Colors

bridge = CvBridge()


class MetricsWindow(QtWidgets.QMainWindow):
    def __init__(self, bag_file, x_points, y_points, first_image, bag_metadata, experiment_metrics,
                 circuit_diameter):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QGridLayout(self._main)

        self.setWindowTitle("Metrics for: " + bag_file)
        self.x_points = x_points
        self.y_points = y_points
        self.first_image = first_image
        self.bag_metadata = bag_metadata
        self.experiment_metrics = experiment_metrics
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
        label_world = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">World: </span>' + self.bag_metadata[
                'world'])
        self.layout.addWidget(label_world)
        label_brain_path = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Brain path: </span>' + self.bag_metadata[
                'brain_path'])
        self.layout.addWidget(label_brain_path)
        label_robot_type = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Robot type: </span>' + self.bag_metadata[
                'robot_type'])
        self.layout.addWidget(label_robot_type)
        label_mean_brain_iterations_real_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Mean brain iterations real time: </span>' + str(
                self.experiment_metrics['mean_brain_iterations_real_time']) + 's')
        self.layout.addWidget(label_mean_brain_iterations_real_time)
        label_brain_iterations_frequency_real_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Brain iterations frequency simulated time: </span>' + str(
                self.experiment_metrics['brain_iterations_frequency_real_time']) + 's')
        self.layout.addWidget(label_brain_iterations_frequency_real_time)
        label_target_brain_iterations_real_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Target brain iterations real time: </span>' + str(
                self.experiment_metrics['target_brain_iterations_real_time']) + 'it/s')
        self.layout.addWidget(label_target_brain_iterations_real_time)
        label_brain_iterations_frequency_simulated_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Brain iterations frequency simulated time: </span>' + str(
                self.experiment_metrics['brain_iterations_frequency_simulated_time']) + 's')
        self.layout.addWidget(label_brain_iterations_frequency_simulated_time)
        label_target_brain_iterations_simulated_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Target brain iterations simulated time: </span>' + str(
                self.experiment_metrics['target_brain_iterations_simulated_time']) + 'it/s')
        self.layout.addWidget(label_target_brain_iterations_simulated_time)

        label_mean_inference_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Mean inference time: </span>' + str(
                self.experiment_metrics['mean_inference_time']) + 's')
        self.layout.addWidget(label_mean_inference_time)
        label_gpu_inference = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">GPU inferencing: </span>' + str(
                self.experiment_metrics['gpu_inference']))
        self.layout.addWidget(label_gpu_inference)
        label_suddenness_distance = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Suddenness distance: </span>' + str(
                self.experiment_metrics['suddenness_distance']))
        self.layout.addWidget(label_suddenness_distance)
        label_frame_rate = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Frame rate: </span>' + str(
                self.experiment_metrics['frame_rate']) + 'fps')
        self.layout.addWidget(label_frame_rate)
        label_mean_brain_iterations_simulated_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Mean brain iterations simulated time: </span>' + str(
                self.experiment_metrics['mean_brain_iterations_simulated_time']))
        self.layout.addWidget(label_mean_brain_iterations_simulated_time)
        label_real_time_factor = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Real time factor: </span>' + str(
                self.experiment_metrics['real_time_factor']))
        self.layout.addWidget(label_real_time_factor)
        label_real_time_update_rate = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Real time update rate: </span>' + str(
                self.experiment_metrics['real_time_update_rate']))
        self.layout.addWidget(label_real_time_update_rate)
        label_experiment_total_simulated_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Experiment total simulated time: </span>' + str(
                self.experiment_metrics['experiment_total_simulated_time']) + 's')
        self.layout.addWidget(label_experiment_total_simulated_time)
        label_experiment_total_real_time = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Experiment total real time: </span>' + str(
                self.experiment_metrics['experiment_total_real_time']) + 's')
        self.layout.addWidget(label_experiment_total_real_time)
        label_circuit_diameter = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Circuit diameter: </span>' + str(
                self.circuit_diameter))
        self.layout.addWidget(label_circuit_diameter)
        label_completed_distance = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Completed distance: </span>' + str(
                self.experiment_metrics['completed_distance']) + "m")
        self.layout.addWidget(label_completed_distance)
        label_percentage_completed = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Percentage completed: </span>' + str(
                self.experiment_metrics['percentage_completed']) + "%")
        self.layout.addWidget(label_percentage_completed)
        label_average_speed = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Average speed: </span>' + str(
                self.experiment_metrics['average_speed']) + "m/s")
        self.layout.addWidget(label_average_speed)
        label_position_deviation_mae = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Position deviation MAE: </span>' + str(
                self.experiment_metrics['position_deviation_mae']))
        self.layout.addWidget(label_position_deviation_mae)
        label_position_deviation_total_err = QLabel(
            '<span style=" font-size:10pt; font-weight:600; color:#000000;">Position deviation total error: </span>' + str(
                self.experiment_metrics['position_deviation_total_err']))
        self.layout.addWidget(label_position_deviation_total_err)

        if 'lap_seconds' in self.experiment_metrics:
            label_lap_seconds = QLabel(
                '<span style=" font-size:10pt; font-weight:600; color:#000000;">Lap seconds: </span>' + str(
                    self.experiment_metrics['lap_seconds']))
            self.layout.addWidget(label_lap_seconds)
            label_circuit_diameter = QLabel(
                '<span style=" font-size:10pt; font-weight:600; color:#000000;">Circuit diameter: </span>' + str(
                    self.experiment_metrics['circuit_diameter']))
            self.layout.addWidget(label_circuit_diameter)


def read_bags(bags):
    bags_checkpoints = []
    bags_metadata = []
    bags_experiment_data = []
    first_images = []
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
                h = json.dumps(y, indent=4)
                data = json.loads(h)
                metadata = json.loads(data['data'])
                bags_metadata.append(metadata)

            for topic, point, t in bag.read_messages(topics=['/experiment_metrics']):
                y = yaml.load(str(point), Loader=yaml.FullLoader)
                h = json.dumps(y, indent=4)
                data = json.loads(h)
                lapdata = json.loads(data['data'])
                bags_experiment_data.append(lapdata)
            first_image = np.zeros((1, 1))
            for topic, point, t in bag.read_messages(topics=['/first_image']):
                first_image = bridge.imgmsg_to_cv2(point, desired_encoding='passthrough')
            first_images.append(first_image)

            bag.close()
            correct_bags += 1
        except Exception as excep:
            print(excep)
            print('Error in bag')

    print('Correct bags: ' + str(correct_bags))

    return bags_checkpoints, bags_metadata, bags_experiment_data, first_images


def show_metrics(bags, bags_checkpoints, bags_metadata, bags_experiment_data, first_images):
    experiments_metrics = []
    world_completed = {}

    for x, checkpoints in enumerate(bags_checkpoints):
        x_points = []
        y_points = []

        experiment_metrics = {'world': bags_metadata[x]['world'], 'brain_path': bags_metadata[x]['brain_path'],
                                 'robot_type': bags_metadata[x]['robot_type']}
        if bags_metadata[x]['world'] == 'simple_circuit.launch':
            perfect_lap_path = 'perfect_bags/lap-simple-circuit.bag'
        elif bags_metadata[x]['world'] == 'many_curves.launch':
            perfect_lap_path = 'perfect_bags/lap-many-curves.bag'
        elif bags_metadata[x]['world'] == 'montmelo_line.launch':
            perfect_lap_path = 'perfect_bags/lap-montmelo.bag'

        perfect_lap_checkpoints, circuit_diameter = metrics_gazebo.read_perfect_lap_rosbag(perfect_lap_path)
        experiment_metrics = bags_experiment_data[x]
        experiment_metrics['experiment_metrics'] = experiment_metrics
        experiments_metrics.append(experiment_metrics)
        if experiment_metrics['percentage_completed'] > 100:
            if bags_metadata[x]['world'] in world_completed and \
                    bags_metadata[x]['brain_path'] in world_completed[bags_metadata[x]['world']]:
                world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = \
                    world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] + 1
            elif bags_metadata[x]['world'] in world_completed:
                world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = 1
            else:
                world_completed[bags_metadata[x]['world']] = {}
                world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = 1
        if 'lap_seconds' in experiment_metrics:
            print('LAP SECONDS -> ' + str(experiment_metrics['lap_seconds']))
            print('CIRCUIT DIAMETER -> ' + str(experiment_metrics['circuit_diameter']))
            print('AVERAGE SPEED -> ' + str(experiment_metrics['average_speed']))

        for point in checkpoints:
            point_yml = yaml.load(str(point), Loader=yaml.FullLoader)
            x_points.append(point_yml['pose']['pose']['position']['x'])
            y_points.append(point_yml['pose']['pose']['position']['y'])

        qapp = QtWidgets.QApplication(sys.argv)
        app = MetricsWindow(bags[x], x_points, y_points, first_images[x], bags_metadata[x], experiment_metrics,
                            circuit_diameter)
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

    bags_checkpoints, bags_metadata, bags_lapdata, first_images = read_bags(config_data['bags'])
    show_metrics(config_data['bags'], bags_checkpoints, bags_metadata, bags_lapdata, first_images)


if __name__ == "__main__":
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    main()
