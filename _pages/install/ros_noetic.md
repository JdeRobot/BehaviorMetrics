---
permalink: /install/ros_noetic

title: "Installation with the ROS Noetic version"

sidebar:
  nav: "docs"


gallery:
  - url: /install/ros_noetic
    image_path: /assets/images/install/ros_noetic.jpg
    alt: "ROS noetic"
    title: "ROS noetic"
gallery1:
  - url: /install/ros_noetic
    image_path: /assets/images/install/noetic/behavior-metrics.png
    alt: ""
gallery2:
  - url: /install/ros_noetic
    image_path: /assets/images/install/noetic/jupyter.png
    alt: ""
gallery3:
  - url: /install/ros_noetic
    image_path: /assets/images/install/noetic/terminal.png
    alt: ""
gallery4:
  - url: /install/ros_noetic
    image_path: /assets/images/install/noetic/vnc.png
    alt: ""
gallery5:
  - url: /install/ros_noetic
    image_path: /assets/images/install/noetic/vnc-viewer.png
    alt: ""
---

{% include gallery id="gallery" %}

Behavior Metrics with ROS Noetic can be installed as usual in the machine or using Docker. 
Since ROS Noetic needs Ubuntu 20 and the dependencies are quite new, that workflow is also provided.


## Table of Contents

1. [Ordinary Installation](#installation)
    1. [Requirements](#requisites)
    2. [Installing ROS Noetic](#noetic)
    3. [Installing Jderobot's dependencies](#dependencies)
    4. [Installing Behavior Metrics](#behavior-metrics)
    5. [Installing CARLA simulator and support](#carla-simulator)
2. [Installation using Docker](#docker-installation)
    1. [Starting Docker Container](#starting-docker)
        1. [VNC container viewer](#vnc)
        2. [Terminal in container](#term)
        3. [Stopping container](#stop)
        4. [Resuming container](#resume)
    2. [Building the container](#building)

## Installation

### Requirements <a name="requisites"></a>

- Ubuntu 20.04

### Installing ROS Noetic <a name="noetic"></a>

A detailed ROS Noetic installation guide can be found in the [ROS Wiki](http://wiki.ros.org/noetic/Installation/Ubuntu)

#### Setup your sources

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

#### Set up your keys

```bash
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

#### Installing ROS Noetic

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full
```

#### Environment setup

```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Installing Python 3.7 and creating a virtualenv

It is recommended to use virtual environment for Behavior Metrics.

```bash
sudo apt install software-properties-common python3-pip python3-virtualenv
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7

# Create virtualenv
virtualenv -p python3.7 .behavior-metrics
source .behavior-metrics/bin/activate
pip install empy
sudo apt-get install python3.7-dev
```

### Installing dependencies <a name="dependencies"></a>

#### JdeRobot's CustomRobots

```bash
git clone -b noetic-devel https://github.com/JdeRobot/CustomRobots
cd CustomRobots/f1 && mkdir build && cd build
/bin/bash -c "source /opt/ros/noetic/setup.bash;
cmake .. && make && make install;"
echo "source /opt/jderobot/share/jderobot/gazebo/assets-setup.sh" >> ~/.bashrc
```

#### ROS additional package

```bash
git clone https://github.com/strasdat/Sophus
cd Sophus && mkdir build && cd build
cmake ../ && make && make install
```

#### Gym-gazebo

```bash
sudo apt-get install libbluetooth-dev libcwiid-dev libftdi-dev libspnav-dev libsdl-dev libsdl-image1.2-dev libusb-dev ros-noetic-octomap-msgs ros-noetic-geodesy ros-noetic-octomap-ros ros-noetic-control-toolbox ros-noetic-pluginlib	ros-noetic-trajectory-msgs ros-noetic-control-msgs ros-noetic-std-srvs ros-noetic-nodelet ros-noetic-urdf ros-noetic-rviz ros-noetic-kdl-conversions ros-noetic-eigen-conversions ros-noetic-tf2-sensor-msgs ros-noetic-pcl-ros ros-noetic-navigation
```

### Installing Behavior Metrics <a name="behavior-metrics"></a>

This application depends on some third party libraries, most of them are included in the requirements file. To install them just type the following:

```bash
git clone -b noetic-devel https://github.com/JdeRobot/BehaviorMetrics
cd BehaviorMetrics
pip3 install -r requirements.txt
cd gym-gazebo
pip3 install -e .
```

If you are going to use the GUI you need to create the resources file for the application. 

```bash
pyrcc5 -o behavior_metrics/ui/gui/resources/resources.py \
	behavior_metrics/ui/gui/resources/resources.qrc
```

### Installing CARLA simulator and support <a name="carla-simulator"></a>

For installing CARLA and supporting this new simulator:

1. Install [CARLA 0.9.13](https://carla.readthedocs.io/en/0.9.13/start_quickstart/)
2. Install [CARLA ROS Bridge](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros1/)
3. Install [CARLA bird-eye-view](https://github.com/deepsense-ai/carla-birdeye-view)
```
    pip3 install carla_birdeye_view==1.1.1 # already in requirements.txt
```
5. Add variables to PYTHONPATH following the simulator installation instructions and ROS bridge installation instructions.
```
    source ~/carla-ros-bridge/catkin_ws/devel/setup.bash
    export CARLA_ROOT=<PATH-TO-CARLA>/carla_simulator_0_9_13/
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg   
```

## Installation using Docker <a name="docker-installation"></a>

The docker installation guide is very clear and can be found in this [link](https://docs.docker.com/get-docker/) which is well documented.

### Download Docker in Ubuntu

First remove older versions.

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

Then setup the stable repository

```bash
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"    
```

Install the docker engine

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

Add your user to the docker's group to avoid using `sudo` for docker, you have to log out and log in and restart to for this change to take effect. 

```
sudo usermod -aG docker your-user
```

Test your installation

```bash
docker run hello-world
```

## Running Behavior Metrics Containers <a name="starting-docker"></a>

Open up a terminal a paste the following command. Creating a volume is recommended so you can add models or datasets easily.
To create the volume update `local_directory` to yur local directory where your datasets and models are located and `docker_directory` to the
directory you want them to be stored inside the container.

### For CPU only

```bash
docker run -dit --name behavior-metrics-noetic \
	-p 5900:5900 \
	-p 8888:8888 \
        -v [local_directory]:[docker_directory] \
	jderobot/behavior-metrics:noetic
```

### For GPU support (CUDA 10.1 Cudnn 7)

Some extra packages are needed for Ubuntu 16.04/18.04/20.04, more about installation in [nvidia-docker docs](https://github.com/NVIDIA/nvidia-docker).

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

The flag `--gpus` is added along with the correct image that contains cuda drivers.

```bash
docker run --gpus all -dit --name behavior-metrics-noetic \
        -p 5900:5900 \
        -p 8888:8888 \
        jderobot/behavior-metrics:noetic-10.1-cudnn7
```
For GPU support on WSL refer [Getting started with CUDA on Ubuntu on WSL 2](https://ubuntu.com/blog/getting-started-with-cuda-on-ubuntu-on-wsl-2).

### Using VNC to visualize container <a name="vnc"></a>

To connect to our container [RealVNC](https://www.realvnc.com/en/) can be installed to access the GUI through the port 5900.

{% include gallery id="gallery4" %}

Once vnc-viewer is open fill in `localhost:5900` in the address and then press connect.

{% include gallery id="gallery5" %}

You will need to authenticate, the current password is **jderobot**, although it can be changed in the script `vnc_startup.sh`.

### Using terminal in container <a name="term"></a>

The recommended way to work, is by writing down `docker logs container-name` and you will get an URL, which will take you to notebook, double click on the last URL to open Jupyter.

```bash
docker logs behavior-metrics-noetic
```

{% include gallery id="gallery2" %}

Go to that URL in the browser (outside VNC) and once you are in the notebook you can open up a terminal by clicking in Terminal.

{% include gallery id="gallery3" %}

A terminal window will open. Type 

```bash
bash
``` 

and this window will behave as any other Ubuntu terminal, so you are ready to run Behavior Metrics, once the GUI is opened it will be displayed in the VNC window.

```bash
cd BehaviorMetrics/behavior_metrics/
python3 driver.py -c configs/default.yml -g
```

This command will open the Gazebo Simulation in the VNC window. You can also directly run the previous command inside VNC window in a terminal.

**IF THE PREVIOUS COMMAND FAILS** try the following and try again:

```bash
sudo apt-get update && sudo apt-get upgrade
```

{% include gallery id="gallery1" %}

### Stopping container <a name="stop"></a>

`behavior-metrics-noetic` should be replaced with the name of your container.

```bash
docker stop behavior-metrics-noetic
```

### Resuming container <a name="resume"></a>

`behavior-metrics-noetic` should be replace with the name of your container, this command is similar to `docker run` so now you can run `docker logs container_name` to get a new link for jupyter, and then connect as usual to your VNC viewer.

```bash
docker restart behavior-metrics-noetic
```

## Building the latest container <a name="building"></a>

First go to the folder where the `Dockerfile` is, then use docker use docker built command with the desired name tag.

```bash
cd BehaviorMetrics/.docker/noetic/
docker build -t any-tag-you-want .
```

From here you are to good to go to the [Quick Start guide!](../quick_start/)