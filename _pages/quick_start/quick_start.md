---
title: Quick Start
layout: posts
permalink: /quick_start/

collection: posts

classes: wide

sidebar:
  nav: "docs"

gallery1:
  - url: /assets/images/behavior_suite_diagram.png
    image_path: /assets/images/behavior_suite_diagram.png
    alt: ""
gallery2:
  - url: /assets/images/matrix_schema.png
    image_path: /assets/images/matrix_schema.png
    alt: ""
gallery3:
  - url: /assets/images/config1.png
    image_path: /assets/images/config1.png
    alt: ""
gallery4:
  - url: /assets/images/default_config.png
    image_path: /assets/images/default_config.png
    alt: ""
gallery5:
  - url: /assets/images/main_window.png
    image_path: /assets/images/main_window.png
    alt: ""
gallery6:
  - url: /assets/images/toolbar.png
    image_path: /assets/images/toolbar.png
    alt: ""
gallery6.1:
  - url: /assets/images/stats.png
    image_path: /assets/images/stats.png
    alt: ""
gallery7:
  - url: /assets/images/dataset.png
    image_path: /assets/images/dataset.png
    alt: ""
gallery8:
  - url: /assets/images/brain.png
    image_path: /assets/images/brain.png
    alt: ""
gallery9:
  - url: /assets/images/change_brain.gif
    image_path: /assets/images/change_brain.gif
    alt: ""
gallery10:
  - url: /assets/images/simulation.png
    image_path: /assets/images/simulation.png
    alt: ""
gallery11:
  - url: /assets/images/gzclient.gif
    image_path: /assets/images/gzclient.gif
    alt: ""
gallery12:
  - url: /assets/images/brain_sim.gif
    image_path: /assets/images/brain_sim.gif
    alt: ""
gallery13:
  - url: /assets/images/reload_sim.gif
    image_path: /assets/images/reload_sim.gif
    alt: ""
gallery14:
  - url: /assets/images/layout.png
    image_path: /assets/images/layout.png
    alt: ""
gallery15:
  - url: /assets/images/frame.png
    image_path: /assets/images/frame.png
    alt: ""
gallery16:
  - url: /assets/images/rename.gif
    image_path: /assets/images/rename.gif
    alt: ""
gallery17:
  - url: /assets/images/frame_config.gif
    image_path: /assets/images/frame_config.gif
    alt: ""
---

First, you need to install Behavior Metrics. If you haven't completed that step, please go to the [installation section](/install/).

Follow the [tutorial](tutorial/) for training your first brain using deep learning and running it on Behavior Studio. 

We additionally have some pretrained brains that you can use in Behavior Metrics to illustrate how it works. Find them in the [brains zoo](brains_zoo/).

If you'd like to train your own brain, we provide you with the [datasets](datasets/).

To run the application with GUI (Graphic User Interface) just run:

```bash
python driver.py -c ./configs/default.yml -g
```

To run the application with TUI (Terminal User Interface) just run:

```bash
python driver.py -c ./configs/default.yml -t
```

To run the application as a script to run several worlds/brains at the same time  run:

```bash
python driver.py -c ./configs/default-multiple.yml -s
```

## Reference times for each circuit

For each circuit, the explicit brain can complete it in these times.

| Circuit      | Time |
| ----------- | ----------- |
| Simple Circuit      | ~ 145 secs (2:25)       |
| Many curves   | ~ ... secs (:)        |
| Montmeló   | ~ ... secs (:)        |
| Montreal   | ~ ... secs (:)        |
| Nurburgring   | ~ ... secs (:)        |


## How to use

First **if you are going to use the GUI** you need to create the resources file for the application. **YOU JUST HAVE TO DO THIS STEP THE FIRST TIME YOU RUN THE APPLICATION**


```bash
pyrcc5 -o ui/gui/resources/resources.py ui/gui/resources/resources.qrc
```


To launch the application just run the python script as follows:

```bash
python driver.py -c ./configs/default.yml -g
```

This command will launch the application with the default configuration:

* Simulated world of a F1 car inside a circuit
* Sensors: RGB camera and odometry
* Actuators: Motors
* Explicit brain based on OpenCV image processing.

The program allows the following arguments:

* `-c <profile>` or `--config <profile>`: this argument is mandatory and specifies the path of the configuration file for the application.
* `-g` or `--gui`: this argument is optional and enables the GUI launching. If not specified, it will show the TUI (terminal user interface).
* `-t` or `--tui`: this argument is optional and enables the TUI.
* `-s` or `--script`: this argument is optional and enables the scriptable application.
* `-r` or `--random`: this argument is optional and enables initialization of the Formula-1 car at random positions on the circuit with random orientation.

For more information run `help driver.py` in a terminal.

Furthermore, the iris drone counterparts are based on `JdeRobot/drones` and the corresponding DroneWrapper. All low level drone control and reading sensor data functionalities can be directly used from the wrapper. To launch a similar case of simulating the line following task based on OpenCV for drones, execute:

```bash
python driver.py -c ./configs/default-drone.yml -g
```


### Building your configuration file

If you want to create your own **configuration file** for the application (changing the robot, brain, layout, etc) you can either use the desktop GUI or creating a yml file with your own configuration. The default profile looks like this (**Make sure you respect the indentation**):

```yaml
Behaviors:
    Robot:
        Sensors:
            Cameras:
                Camera_0:
                    Name: 'camera_0'
                    Topic: '/F1ROS/cameraL/image_raw'
            Pose3D:
                Pose3D_0:
                    Name: 'pose3d_0'
                    Topic: '/F1ROS/odom'
        Actuators:
            Motors:
                Motors_0:
                    Name: 'motors_0'
                    Topic: '/F1ROS/cmd_vel'
                    MaxV: 3
                    MaxW: 0.3
        BrainPath: 'brains/f1/brain_f1_opencv2.py'
        Type: 'f1'
    Simulation:
        World: /opt/jderobot/share/jderobot/gazebo/launch/f1_1_simplecircuit.launch
    Dataset:
        In: '/tmp/my_bag.bag'
        Out: ''
    Layout:
        Frame_0:
            Name: frame_0
            Geometry: [1, 1, 2, 2]
            Data: rgbimage
        Frame_1:
            Name: frame_1
            Geometry: [0, 1, 1, 1]
            Data: rgbimage
        Frame_2:
            Name: frame_2
            Geometry: [0, 2, 1, 1]
            Data: rgbimage
        Frame_3:
            Name: frame_3
            Geometry: [0, 3, 3, 1]
            Data: rgbimage
```

The keys of this file are as follows:

**Robot**

This key defines the robot configuration:

* **Sensors**: defines the sensors configuration. Every sensor **must** have a name and a ROS topic. 
* **Actuators**: defines the actuators configuration. Every actuator **must** have a name and a ROS topic.
* **BrainPath**: defines the path where the control algorithm is located in your system.
* **Type**: defines the type of robot. Possible values: f1, drone, turtlebot, car.

**Simulation**

This key defines the launch file of the environment. It can be used to launch a gazebo simulation or a real robot.

**Dataset**

This key define the dataset output path.

**Layout**

This key defines how the GUI will be shown. This is the trickiest part of the configuration file. The grid of the GUI is a 3x3 matrix where you can configure the layout positions to show your sensors data. Each *Frame_X* key corresponds to a view of a sensor in the GUI and includes the following keys:

* Name: this is mandatory in order to send data to that frame in the GUI to be shown.
* Geometry: is the position and size configuration of that frame following this criteria: ` [x, y, height, width]`
* Data: tells the GUI which kind of GUI should be create in order to show the information. Possible values: rgbimage, laser, pose3d.

The geometry is defined as follows.

{% include gallery id="gallery2" caption="" %}

So if you want to create a view for one of the camera sensors of the robot located in the top-left corner of size 1x1 followed by another view of another camera in the bottom-right corner of size 2x2, you should configure the geometry array as:

```yaml
Frame_0:
    Name: 'Camera1'
    Geometry: [0, 1, 1, 1]
    Data: rgbimage
Frame_1:
    Name: 'Camera2'
    Geometry: [1, 2, 2, 2]
```

So it will look like this in the GUI:

{% include gallery id="gallery3" caption="" %}

Following this logic, you will see that the **default configuration** file will show something like this:

{% include gallery id="gallery4" caption="" %}

**Custom Brain parameters**

Some parameters for custom brains can be set up directly from the *yml* file below the type of the robot which in this example is *f1*. This format is used to setup paths for trained models and any brain kwargs required by the user.

```yml
BrainPath: 'brains/f1/brain_f1_torch.py'
Type: 'f1'
Parameters:
    Model: 'trained_model_name.checkpoint'
    ImageCrop: True 
    ParamExtra: {Specify value}
```     

**Reinforcement Learning parameters**

Optionally some parameters for reinforcement learning can be set up directly from the *yml* file below the type of the robot which in this example is *f1rl*.

```yml
BrainPath: 'brains/f1rl/train.py'
Type: 'f1rl'
Parameters:
    action_set: 'simple'
    gazebo_positions_set: 'pista_simple'
    alpha: 0.2 
    gamma: 0.9
    epsilon: 0.99
    total_episodes: 20000
    epsilon_discount: 0.9986 
    env: 'camera'
```           

### Using the application

Once the configuration file is created and the application has launched, you will see something like this (depending on your layout configuration. We assume you launched the default profile):

{% include gallery id="gallery5" caption="" %}

You will see 2 different sections, one on the left: **the toolbar**, and another one in the right: **the layout**. 

#### The toolbar

You have all the tools needed for controlling the whole application, simulation and control of the application. For usability sake, this section is subdivided in 4 different subsections: **stats, dataset, brains** and **simulation**.

{% include gallery id="gallery6" caption=""%}

**Stats**

{% include gallery id="gallery6.1" caption=""%}


You can save metrics from the brain with the stats functionality. For saving, press play while the brain is running and 
press again to finish. After that, a general view of the stats should appear. For further detail, run the `show_plots.py` script:

```
    python3 show_plots.py -b [bag_name].bag -b [bag_name].bag
``` 

This script will load further information related with the execution.

**Visualizing Brain Performances**

The `behavior_metrics/show_plots.py` file uses the QtWindow to generate it runtime on the DISPLAY. So, the new script setup as found in `behavior_metrics/scripts/analyze_brain.bash` eases the overall process by first collecting the ROS bags with the confg file provided and then generates all the analysis plots. The argument for the analysis is the config file suitable for using with the `script` mode.

```
    bash scripts/analyze_brain.bash ./configs/default-multiple.yml
``` 

Finally, this saves everything at `behavior_metrics/bag_analysis/bag_analysis_plots/` directory sorted according to the different circuits. The current formulation of the saving plots analysis creates the following directory structure:

```
behavior_metrics/bag_analysis
	+-- bag_analysis_plots/
	|	+-- circuit_name/ 						
	|   	+-- performances   
	|   	|   +-- completed_distance.png
	|   	|   +-- percentage_completed.png
	|   	|   +-- lap_seconds.png	
	|   	|   +-- circuit_diameter.png 		
	|			|   +-- average_speed.png 
	|   	+-- first_images/			
	|		  +-- path_followed/ 					
	+-- bags/ 
```

**Dataset**

{% include gallery id="gallery7" caption="" %}

This subsection will allow you to specify where the datasets will be saved by setting up the path and the name of the bag file.

To specify the output ROS bag file, just click on the three dots and a dialog window will open with the file system.

The button **Select topics** is used to select which active topics the user wants to record in the ROS bag file.

Use the play button to start/stop recording the rosbag.

**Note: if you don't change your ROS bag name between recordings, the file will be overwritten with the new recording data.**

**Brain**

{% include gallery id="gallery8" caption="" %}

This subsection will allow you to control the logic of the robot: its behavior. 

You have a drop-down menu that will detect the available brains for the current simulation, so you can select whatever brain you want in each moment. **This can be done on the go, if the simulation is paused**

The **Load** button will load a new brain in execution time **if the simulation is paused**

{% include gallery id="gallery9" caption="" %}

All of this tools will be disabled while the simulation is running, so in order to interact with it, you should pause the simulation first.

**Simulation**

{% include gallery id="gallery10" caption="" %}

This subsection will allow you to control the simulation.

You have a drop-down menu to change the world of the simulation on the go **if the simulation is paused**

The **Load** button will load the specified world in the drop-down menu **if the simulation is paused**

You have 3 additional buttons which:

* Will load Gazebo GUI if it wasn't launched, or close it otherwise

{% include gallery id="gallery11" caption="" %}

* Play/pause button for **pausing/resuming the simulation**

{% include gallery id="gallery12" caption="" %}

* Reload button will reload the simulation by resetting the robot position and both real and simulation time.

{% include gallery id="gallery13" caption="" %}

respectively.

#### The layout

This section is meant to show the data coming from the sensors of the robot (cameras, laser, odometry, etc.). For that purpose, the GUI is divided in sections conforming a **layout**. This disposition will come specified in the configuration file (see *Building your configuration file*) section.

{% include gallery id="gallery14" caption="" %}

As you can see, there are several boxes or **frames** that will host data from different sensors. The view above shows the GUI before specifying what kind of sensor and data the frame will show. You only have to give the frame a **name** (if you leave it blank it will take the default name *frame_X* where X is the frame number), and pick which kind of data that frame will contain by clicking on one of the radio buttons.

{% include gallery id="gallery15" caption="" %}

As you type down the name of the frame, you will see how the name in the top-left corner of the frame changes dynamically.

{% include gallery id="gallery16" caption="" %}

Once you have chosen the frame name (this is important for later), you have to chose the data type the frame will show, from one of the checkboxes below the name textbox. After that, you will only have to click the **Confirm** button and the sensor will show its data.

{% include gallery id="gallery17" caption="" %}
