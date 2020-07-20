# Gym Gazebo

**An OpenAI gym extension for using Gazebo known as `gym-gazebo`**.

The are 3 repositories:

- [Erlerobot](https://github.com/erlerobot/gym-gazebo). Current version.
- [Acutronic](https://github.com/AcutronicRobotics/gym-gazebo2) using ROS2 (newer version).
- [The construct](https://www.theconstructsim.com/machine-learning-openai-gym-ros-development-studio-2/) + ROS for MachineLearning.

This work presents an extension of the initial OpenAI gym for robotics using ROS and Gazebo. A whitepaper about this work is available at https://arxiv.org/abs/1608.05742.

`gym-gazebo` is a complex piece of software for robotics that puts together simulation tools, robot middlewares (ROS, ROS 2), machine learning and reinforcement learning techniques. All together to create an environment where to benchmark and develop behaviors with robots. Setting up `gym-gazebo` appropriately requires relevant familiarity with these tools.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Environments](#community-maintained-environments)

## Installation

Installation over Ubuntu 18.04:

- ROS Melodic: Desktop-Full Install recommended, includes Gazebo 9.0.0 (http://wiki.ros.org/melodic/Installation/Ubuntu).
- Gazebo 9.0.0

### ROS Melodic related dependencies

Full ROS package:

```bash
 sudo apt install ros-melodic-desktop-full
```

More packages:

```
sudo apt-get install                     \
python-pip python3-vcstool python3-pyqt4 \
pyqt5-dev-tools                          \
libbluetooth-dev libspnav-dev            \
pyqt4-dev-tools libcwiid-dev             \
cmake gcc g++ qt4-qmake libqt4-dev       \
libusb-dev libftdi-dev                   \
python3-defusedxml python3-vcstool       \
ros-melodic-octomap-msgs                 \
ros-melodic-joy                          \
ros-melodic-geodesy                      \
ros-melodic-octomap-ros                  \
ros-melodic-control-toolbox              \
ros-melodic-pluginlib	                 \
ros-melodic-trajectory-msgs              \
ros-melodic-control-msgs                 \
ros-melodic-std-srvs 	                 \
ros-melodic-nodelet                      \
ros-melodic-urdf                         \
ros-melodic-rviz                         \
ros-melodic-kdl-conversions              \
ros-melodic-eigen-conversions            \
ros-melodic-tf2-sensor-msgs              \
ros-melodic-pcl-ros                      \
ros-melodic-rosbash                      \
ros-melodic-navigation                   \
ros-melodic-sophus                       \
python-rviz
```

Add the `setup.bash` file to your `.bashrc`:

```bash
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
```

### Create a virtualenv

```bash
virtualenv -p python3 gym-gazebo-env
```

Activate the virtual environment:

```bash
source gym-gazebo-env/bin/activate
```

### Install Python Packages:

```bash
pip install -r requirements
```

#### Install gym-gazebo

```bash
cd ~
git clone https://github.com/RoboticsLabURJC/2019-tfm-ignacio-arranz.git
cd gym-gazebo
```

#### Run bash files, build the ROS workspace:

```bash
cd gym-gazebo/gym_gazebo/envs/installation
bash setup_melodic.bash
```

## Usage

### Build and install gym-gazebo

In the root directory of the repository:

```bash
pip install -e .
```

Configure ROS environment executing the following script (actually using ROS melodic):

```bash
bash setup_melodic.bash
```

### Running an environment

- Load the environment variables corresponding to the robot you want to launch. E.g. to load the Formula 1:

```bash
cd gym_gazebo/envs/installation
bash formula1_setup.bash
```

Note: all the setup scripts are available in `gym_gazebo/envs/installation`

- Run any of the examples available in `examples/`. E.g.:

```bash
cd examples/f1
python f1_follow_line_camera.py
```

### Display the simulation

To see what's going on in Gazebo during a simulation, run gazebo client. In order to launch the `gzclient` and be able to connect it to the running `gzserver`:

1. Open a new terminal.
2. Source the corresponding setup script, which will update the _GAZEBO_MODEL_PATH_ variable: e.g. `source setup_turtlebot.bash`
3. Export the _GAZEBO_MASTER_URI_, provided by the [gazebo_env](https://github.com/erlerobot/gym-gazebo/blob/7c63c16532f0d8b9acf73663ba7a53f248021453/gym_gazebo/envs/gazebo_env.py#L33). You will see that variable printed at the beginning of every script execution. e.g. `export GAZEBO_MASTER_URI=http://localhost:11311`

Finally, launch `gzclient`.

```bash
gzclient
```

Also, you can see the F1 camera using `rviz` + `ros_topic` like this:

```bash
rosrun image_view image_view image:=/F1ROS/cameraL/image_raw
```

### Display reward plot (under review)

Display a graph showing the current reward history by running the following script:

```bash
cd examples/utilities
python display_plot.py
```

HINT: use `--help` flag for more options.

### Killing background processes

Sometimes, after ending or killing the simulation `gzserver` and `rosmaster` stay on the background, make sure you end them before starting new tests.

We recommend creating an alias to kill those processes.

```bash
echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> ~/.bashrc
```

You can also run a script that stops the processes, in the `scripts` folder:

```bash
gym_gazebo/scripts/stop.sh
```

## Community-maintained environments

The following are some of the gazebo environments maintained by the community using `gym-gazebo`.

| Name                                                         | Middleware                                         | Description                                                  | Reward range |
| ------------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------------ | ------------ |
| ![GazeboCircuit2TurtlebotLidar-v0](docs/GazeboCircuit2TurtlebotLidar-v0.png)`GazeboCircuit2TurtlebotLidar-v0` | ROS                                                | A simple circuit with straight tracks and 90 degree turns. Highly discretized LIDAR readings are used to train the Turtlebot. Scripts implementing **Q-learning** and **Sarsa** can be found in the _examples_ folder. |              |
| ![GazeboCircuitTurtlebotLidar-v0](docs/GazeboCircuitTurtlebotLidar-v0.png)`GazeboCircuitTurtlebotLidar-v0.png` | ROS                                                | A more complex maze  with high contrast colors between the floor and the walls. Lidar is used as an input to train the robot for its navigation in the environment. | TBD          |
| `GazeboMazeErleRoverLidar-v0`                                | ROS, [APM](https://github.com/erlerobot/ardupilot) | **Deprecated**                                               |              |
| `GazeboErleCopterHover-v0`                                   | ROS, [APM](https://github.com/erlerobot/ardupilot) | **Deprecated**                                               |              |

## Other environments (no support provided for these environments)

The following table compiles a number of other environments that **do not have
community support:**

- `GazeboCartPole-v0`
- `GazeboModularArticulatedArm4DOF-v1`
- `GazeboModularScara4DOF-v3`
- `GazeboModularScara3DOF-v3`
- `GazeboModularScara3DOF-v2`
- `GazeboModularScara3DOF-v1`
- `GazeboModularScara3DOF-v0`
- `ARIACPick-v0`

More information in original [gym-gazebo](https://github.com/erlerobot/gym-gazebo) repository.


## Troubleshooting

Other possible libraries. Maybe these packages are necessary.

```bash
sudo apt install libosmesa6-dev
sudo apt install meshlab
sudo apt install libsdl1.2-dev
sudo apt-get install python3-empy
```

- Problem: `[Err] [REST.cc:205] Error in REST request python`. Solution: Change `~/.ignition/fuel/config.yaml` as following. From: `url: https://api.ignitionfuel.org` to `url: https://api.ignitionrobotics.org`.
- Problem: `No module named 'rospy'`. Solution: `export PATH="${PATH}:${HOME}/.local/bin/"`

- `Error. No module named rospy` in PyCharm-community edition. [Follow this thread in ROS forum](https://answers.ros.org/question/304101/configure-pycharm-to-make-the-linting-work-properly/).