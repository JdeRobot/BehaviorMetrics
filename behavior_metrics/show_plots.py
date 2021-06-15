import sys
import json
import yaml
import rosbag


bags = []

bag_1 = '20210528-143935.bag'
bags.append(bag_1)

bags_checkpoints = []
bags_metadata = []
time_stats = []
correct_bags = 0

for bag_file in bags:
    print('------------------BAG----------------')
    print(bag_file)
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
            print(metadata)
            print(metadata['world'])
            print(metadata['brain_path'])
            print(metadata['robot_type'])
            #print(metadata)
            bags_metadata.append(metadata)
            print(len(bag_checkpoints))
        for topic, point, t in bag.read_messages(topics=['/time_stats']):

            y = yaml.load(str(point), Loader=yaml.FullLoader)
            h = json.dumps(y,indent=4)
            data = json.loads(h)
            time_stats_metadata = json.loads(data['data'])
            #a_new = np.array(time_stats_metadata['first_image'])
            #plt.imshow(a_new)
            #plt.show()
            
            print('mean_iteration_time -> ' + str(time_stats_metadata['mean_iteration_time']))
            print('mean_inference_time -> ' + str(time_stats_metadata['mean_inference_time']))
            print('gpu_inferencing -> ' + str(time_stats_metadata['gpu_inferencing']))
            print('frame_rate -> ' + str(time_stats_metadata['frame_rate']))
            time_stats.append(time_stats_metadata)
            print(len(bag_checkpoints))    

        bag.close()
        correct_bags += 1
    except Exception as excep:
        print(excep)
        print('Error in bag')
    print('----------------------------------')

print('CORRECT BAGS : ' + str(correct_bags))


#####



import numpy as np
import matplotlib.pyplot as plt
from utils import metrics

experiments_statistics = []
world_completed = {}

for x, checkpoints in enumerate(bags_checkpoints):
    #try:
    print('------- ' + str(x) + ' -----')
    #try:
    x_points = []
    y_points = []
    print('-------------------------------------------')
    print(bags_metadata[x])
    print('WORLD -> ' + bags_metadata[x]['world'])
    print('BRAIN PATH -> ' + bags_metadata[x]['brain_path'])
    print('ROBOT TYPE -> ' + bags_metadata[x]['robot_type'])
    try:
        print('MEAN ITERATION TIME ->' + str(time_stats[x]['mean_iteration_time']))
        print('MEAN INFERENCE TIME ->' + str(time_stats[x]['mean_inference_time']))
        print('FRAME RATE ->' + str(time_stats[x]['frame_rate']))
        print('GPU INFERENCING ->' + str(time_stats[x]['gpu_inferencing']))
    except Exception as err:
        print(err)
    
    experiment_statistics = {}
    experiment_statistics['world'] = bags_metadata[x]['world']
    experiment_statistics['brain_path'] = bags_metadata[x]['brain_path']
    experiment_statistics['robot_type'] = bags_metadata[x]['robot_type']
    if bags_metadata[x]['world'] == 'simple_circuit.launch':
        perfect_lap_path = 'lap-simple-circuit.bag'
    elif bags_metadata[x]['world'] == 'many_curves.launch':
        perfect_lap_path = 'lap-many-curves.bag'
    elif bags_metadata[x]['world'] == 'montmelo_line.launch':
        perfect_lap_path = 'lap-montmelo.bag'

    perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag(perfect_lap_path)
    print('CIRCUIT DIAMETER -> ' + str(circuit_diameter))
    lap_statistics = metrics.lap_percentage_completed(bags[x], perfect_lap_checkpoints, circuit_diameter)
    experiment_statistics['lap_statistics'] = lap_statistics
    experiments_statistics.append(experiment_statistics)
    print('COMPLETED DISTANCE -> ' + str(lap_statistics['completed_distance']))
    print('PERCENTAGE COMPLETED -> ' + str(lap_statistics['percentage_completed']))
    if lap_statistics['percentage_completed'] > 100:
        if bags_metadata[x]['world'] in world_completed and bags_metadata[x]['brain_path'] in world_completed[bags_metadata[x]['world']]:
            world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] + 1
        elif bags_metadata[x]['world'] in world_completed:
            world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = 1
        else:
            world_completed[bags_metadata[x]['world']] = {} 
            world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = 1
            # world_completed[bags_metadata[x]['world']][bags_metadata[x]['brain_path']] = 1
    print('ORIENTATION MAE -> ' + str(lap_statistics['orientation_mae']))
    print('ORIENTATION TOTAL ERROR -> ' + str(lap_statistics['orientation_total_err']))
    print(lap_statistics)
    if 'lap_seconds' in lap_statistics:
        print('LAP SECONDS -> ' + str(lap_statistics['lap_seconds']))
        print('CIRCUIT DIAMETER -> ' + str(lap_statistics['circuit_diameter']))
        print('AVERAGE SPEED -> ' + str(lap_statistics['average_speed']))

    for point in checkpoints:
        point_yml = yaml.load(str(point), Loader=yaml.FullLoader)
        # print(point)
        x_points.append(point_yml['pose']['pose']['position']['x'])
        y_points.append(point_yml['pose']['pose']['position']['y'])


    print('************ SHOW PLOT *************')
    #plt.scatter(x_points, y_points)
    txt="I need the caption to be present a little below X-axis \n fdsafdasfadsfdas"
    #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    #plt.show()
    print('-------------------------------------------')
    print('************ END *************')
    #except Exception as err:
    #    print('error in bag')
    #    print(err)
    
print(world_completed) 
print(experiments_statistics)



print('************ PYQT5 *************')


import sys
import numpy as np

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton

from matplotlib.figure import Figure
from matplotlib import animation


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        self.fig = Figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.addToolBar(NavigationToolbar(self.canvas, self))

        self.setup()
        
        label=QLabel('Text related to the plot 1')
        layout.addWidget(label)
        label=QLabel('Text related to the plot 2')
        layout.addWidget(label)
        label=QLabel('Text related to the plot 3')
        layout.addWidget(label)
        label=QLabel('Text related to the plot 4')
        layout.addWidget(label)

    def setup(self):
        self.ax = self.fig.subplots()
        self.scat = self.ax.scatter(x_points, y_points,  zorder=3)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
