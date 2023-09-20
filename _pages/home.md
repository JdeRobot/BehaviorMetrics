---
layout: splash
permalink: /
header:
  overlay_color: "#2F5565"
  overlay_image: https://jderobot.github.io/assets/images/projects/neural_behavior/autonomous.jpeg
  actions:
    #- label: "<i class='fas fa-download'></i> Install now"
    #  url: "/installation/"
excerpt: 
  Autonomous driving solutions comparison software tool
feature_row:
  - image_path: /assets/images/cover/test_your_network.jpeg
    alt: "Quick start CARLA examples"
    title: "Quick start CARLA examples"
    excerpt: "Load, run, get results and compare an example robot brain in CARLA"
    url: "/quick_start/"
    btn_class: "btn--primary"
    btn_label: "Quick start CARLA examples"

  - image_path: /assets/images/cover/install.png
    alt: "Installation"
    title: "Installation"
    excerpt: "Use the software tool for comparing autonomous driving solutions. Install Behavior Metrics"
    url: "/install/"
    btn_class: "btn--primary"
    btn_label: "Installation"

  - image_path: /assets/images/cover/logbook.jpg
    alt: "Documentation"
    title: "Documentation"
    excerpt: "More information about the project architecture. References used, guides, articles, etc."
    url: "/documentation/"
    btn_class: "btn--primary"
    btn_label: "Documentation"   
youTube_id: ID7qaEcIu4k

gallery1:
  - url: /assets/images/behavior_metrics_full_architecture.png
    image_path: /assets/images/behavior_metrics_full_architecture.png
    alt: ""
---

{% include feature_row %}

**We are always open for new contributions from outside developers. If you want to contribute to this project, please visit the [CONTRIBUTING guide](/documentation/contributing/)**

This software tool provides evaluation capabilities for autonomous driving solutions using simulation. 
We provide a series of quantitative metrics for the evaluation of autonomous driving solutions with support for two simulators, [CARLA](https://carla.org/) (main supported simulator) and [gazebo](https://gazebosim.org/home) (partial support).
Currently supported tasks include:

* **Follow-lane**
* **Driving in traffic**
* **Navigation**

Each task comes with its own custom evaluation metrics that can help compare autonomous driving solutions.
The main component of the ego vehicle is the brain, which receives sensor data, manipulates it, and generates robot control commands based on it. 
The inner part of the brain can be controlled by an end-to-end model, written in Tensorflow or PyTorch, a reinforcement learning policy, or even an explicitly programmed policy.

The software provides two main pipelines, a graphical user interface (GUI) and a headless mode (scripted). 
The first one is intended for testing one brain+model at a time and debugging it visually while the headless mode is intended for running lots of experiments at the same time for comparison of a batch of brain+models in different scenarios.

{% include gallery id="gallery1" caption=""%}

### Installation

For more information about the installation, go to this [link](/install/). 

### Examples

* [CARLA example](/carla/quick_start/)
* [Gazebo example](/gazebo/quick_start/)

<img src="https://jderobot.github.io/assets/images/projects/neural_behavior/autonomous.jpeg" alt="config" />


You can see the project status in the [GitHub repository here](https://github.com/JdeRobot/BehaviorMetrics).
