# Neural Behaviors

This project presents different approaches to the follow-the-line exercise but using artificial intelligence to complete the circuits. The solutions presented are:
- Using classification networks.
- Using regression networks.
- Using reinforcement learning.
- Solution for real robots.

<img src="https://jderobot.github.io/assets/images/projects/neural_behavior/autonomous.jpeg" alt="config" style="zoom:20%;" />

Index of the project:

- [Vision-based end-to-end using Deep Learning](https://github.com/JdeRobot/NeuralBehaviors/tree/master/vision-based-end2end-learning).

- [Deep Reinforcement Learning for autonomous car](https://github.com/RoboticsLabURJC/2019-tfm-ignacio-arranz) (WIP).

- [Deep Learning in autonomous vision based navigation of real robots](https://github.com/RoboticsLabURJC/2017-tfm-francisco-perez) (WIP).

  

## Design

Neural Behaviors project has the following structure (WIP):

![neural_behavior_diagram](docs/assets/images/neural_behavior_diagram.png)

## Current Status

We are currently redesigning the project. The following **functional requirements** have been specified:

| Number | Description                                                  | Status |
| ------ | ------------------------------------------------------------ | ------ |
| RF01   | Changing run-time intelligence                               | WIP    |
| RF02   | Save tagged dataset (IMG + ROSbags, cmd-vel)                 | WIP    |
| RF03   | 'Manual' Autopilot. User solution (OpenCV)                   | DONE   |
| RF04   | Teleoperation                                                | WIP    |
| RF05   | Benchmarking (neuronal network vs groundthruth, checkpoints, center desviation, ...) | -      |
| RF06   | Support for different environments (TensorFlow, Keras, Pytorch, OpenCV, ...) | WIP    |
| RF07   | User profiles (configuration file)                           | -      |

The following table contains **non-functional requirements**:

| Number | Description | Status |
| ------ | ----------- | ------ |
| RN01   | Real time   | -      |
| RN02   | Memory      | -      |
| RN03   | GPU-Ready   | -      |

## References

Zhicheng Li, Zhihao Gu, Xuan Di, Rongye Shi. An LSTM-Based Autonomous Driving Model Using Waymo Open Dataset.
*arXiv e-prints, art.arXiv:2002.05878*, Feb 2020. https://arxiv.org/abs/2002.05878
 
Pei Sun et at. Scalability in Perception for Autonomous Driving: Waymo Open Dataset. 
*arXiv e-prints, art.arXiv:1912.04838*, Dec 2019. https://arxiv.org/abs/1912.04838 
(Waymo Open Dataset)[https://waymo.com/open/]


