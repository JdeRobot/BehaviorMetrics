[![Publish Docker image](https://github.com/JdeRobot/BehaviorMetrics/actions/workflows/main.yml/badge.svg)](https://github.com/JdeRobot/BehaviorMetrics/actions/workflows/main.yml)
[![Publish 10.1-cudnn7 Docker image](https://github.com/JdeRobot/BehaviorMetrics/actions/workflows/generate_docker_10_1_cudnn7.yml/badge.svg)](https://github.com/JdeRobot/BehaviorMetrics/actions/workflows/generate_docker_10_1_cudnn7.yml)
[![Publish cuda 11 Docker image](https://github.com/JdeRobot/BehaviorMetrics/actions/workflows/generate_docker_cuda_11.yml/badge.svg)](https://github.com/JdeRobot/BehaviorMetrics/actions/workflows/generate_docker_cuda_11.yml)
# Behavior Metrics

This software tool provides evaluation capabilities for autonomous driving solutions using simulation. 
We provide a series of quantitative metrics for the evaluation of autonomous driving solutions with support for two simulators, [CARLA](https://carla.org/) (main supported simulator) and [gazebo](https://gazebosim.org/home) (partial support).
Currently supported tasks include:

* Follow-lane
* Driving in traffic
* Navigation

Each task comes with its own custom evaluation metrics that can help compare autonomous driving solutions.
The main component of the ego vehicle is the brain, which receives sensor data, manipulates it, and generates robot control commands based on it. 
The inner part of the brain can be controlled by an end-to-end model, written in Tensorflow or PyTorch, a reinforcement learning policy, or even an explicitly programmed policy.

The software provides two main pipelines, a graphical user interface (GUI) and a headless mode (scripted). 
The first one is intended for testing one brain+model at a time and debugging it visually while the headless mode is intended for running lots of experiments at the same time for comparison of a batch of brain+models in different scenarios.


![alt text](./assets/behavior_metrics_paper_behavior_metrics_full_architecture-1.png)


### Installation

For more information about the project and how to install it, you can consult the [website of Behavior Metrics](https://jderobot.github.io/BehaviorMetrics/). 

### Examples

We provide examples for the follow-lane task using CARLA:

* For an example of a robot brain using a Tensorflow model for control with GUI pipeline, run:

```
python3 driver_carla.py -c configs/CARLA/default_carla_tensorflow.yml -g
```

* For an example of a robot brain using a PyTorch model for control with GUI pipeline, run:

```
python3 driver_carla.py -c configs/CARLA/default_carla_torch.yml -g
```

* For an example of an explicitly programmed robot brain with GUI pipeline, run:

```
python3 driver_carla.py -c configs/CARLA/default_carla.yml -g
```

* For an example of the headless pipeline, run:

```
python3 driver_carla.py -c configs/CARLA/default_carla_multiple.yml -s
```

### Contributing to the project

If you want to contribute, please first check out [CONTRIBUTING.md](CONTRIBUTING.md) section.

<img src="https://jderobot.github.io/assets/images/projects/neural_behavior/autonomous.jpeg" alt="config" style="zoom:20%;" />
