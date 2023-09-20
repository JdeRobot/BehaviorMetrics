---
title: Behavior Metrics tutorial using CARLA
layout: posts
permalink: /carla/tutorial

collection: posts

classes: wide

sidebar:
  nav: "docs"
  
---

In this tutorial, you will use Behavior Metrics to extract evaluation metrics using CARLA

## Table of Contents

- [Prerequisites](#prerequisites)
- [Brain Class](#brain-class)
- [Evaluation metrics](#evaluation-metrics)
- [Running evaluation with GUI](#running-evalaution-gui)


## Prerequisites

First of all, make sure you have Behavior Metrics installed, following the [installation section](/install/). You can try running the *brain_f1_explicit* that is already included on Behavior Metrics brains folder. This brain is capable of finishing every circuit available for the project.

## Brain Class

There are three main functions that can be used to for performing I/O operations on the robot. You can also find a [dummy file](https://github.com/JdeRobot/BehaviorMetrics/blob/noetic-devel/behavior_metrics/brains/f1/brain_f1_dummy.py), with all the instructions, that you can use as a template for deployment.

**Update Frame**

Update the information to be shown in one of the GUI's frames.

Arguments:<br>
- frame_id {str} --  Id of the frame that will represent the data
- data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)

```python
def update_frame(self, frame_id, data):
```

**Update Pose**

Update the pose 3D information obtained from the robot.

Arguments: <br>
- data {*} -- Data to be updated, will be retrieved later by the UI.

```python
def update_pose(self, pose_data):
```

**Execute**

Main loop of the brain. This will be called iteratively each TIME_CYCLE

```python
def execute(self):
```

## Evaluation metrics

## Running evaluation with GUI



