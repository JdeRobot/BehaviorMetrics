# Behavior Metrics Docker

## Table of Contents

1. [Installing Docker](#docker-installation)
2. [Starting Docker Container](#starting-docker)
    1. [VNC container viewer](#vnc)
    2. [Terminal in container](#term)
    3. [Stopping container](#stop)
    4. [Resuming container](#resume)
3. [Building the container](#building)

## Docker installation <a name="docker-installation"></a>

The docker installation guide is very clear and can be found in this [link](https://docs.docker.com/get-docker/) which is well documented.

### Ubuntu

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

Add your user to the docker's group to avoid using `sudo` for docker, you have to log out and log in to for this change to take effect.

```
sudo usermod -aG docker your-user
```

Test your installation

```bash
docker run hello-world
```

## Running Behavior Metrics Containers <a name="starting-docker"></a>

Open up a terminal a paste the following command

### For CPU only

```bash
docker run -dit --name behavior-metrics-noetic \
	-p 5900:5900 \
	-p 8888:8888 \
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

### Using VNC to visualize container <a name="vnc"></a>

To connect to our container [VNC viewer for chrome](https://chrome.google.com/webstore/detail/vnc%C2%AE-viewer-for-google-ch/iabmpiboiopbgfabjmgeedhcmjenhbla?hl=en) (recommended) or [RealVNC](https://www.realvnc.com/en/) can be installed to access the GUI through the port 5900.

![vnc](imgs/vnc.png?raw=true "Searching VNC")

Once vnc-viewer is open fill in `localhost:5900` in the address and then press connect.

![vnc-viewer](imgs/vnc-viewer.png?raw=true "vnc-viewer")

You will need to authenticate, the current password is **jderobot**, although it can be changed in the script `vnc_startup.sh`.

### Using terminal in container <a name="term"></a>

The recommended way to work, is by writing down `docker logs container-name` and you will get an URL, which will take you to notebook, double clock on the last URL to open Jupyter.

```bash
docker logs behavior-metrics-noetic
```

![jupyter](imgs/jupyter.png?raw=true "Jupyter")

Once you are in the notebook you can open up a terminal by clicking in Terminal.

![terminal](imgs/terminal.png?raw=true "Terminal")

A terminal window will open and type `bash` and this window will behave as any other Ubuntu terminal, so you are ready to run Behavior Metrics, one the GUI is opened it will be displayed in the VNC window.

```bash
cd BehaviorMetrics/behavior_metrics
python3 driver.py -c -default -g
```

![behavior-metrics](imgs/behavior-metrics.png?raw=true "Behavior Metrics")

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
