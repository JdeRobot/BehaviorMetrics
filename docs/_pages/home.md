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
  Autonomous driving network comparison tool
feature_row:
  - image_path: /assets/images/cover/test_your_network.jpeg
    alt: "Test your Network"
    title: "Test your Network"
    excerpt: "Load, run, get results and compare your network in different environments."
    url: "/test_your_network/"
    btn_class: "btn--primary"
    btn_label: "Learn more"

  - image_path: /assets/images/cover/install.png
    alt: "Install"
    title: "Install"
    excerpt: "Use of the software. Instructions for replicating project content."
    url: "/install/"
    btn_class: "btn--primary"
    btn_label: "Learn more"

  - image_path: /assets/images/cover/logbook.jpg
    alt: "Documentation"
    title: "Documentation"
    excerpt: "More information about the project. References used, guides, articles, etc."
    url: "/documentation/"
    btn_class: "btn--primary"
    btn_label: "Learn more"   
youTube_id: ID7qaEcIu4k
---

{% include feature_row %}

Behavior Suite is a tool written in Python that, using the JdeRobot environment, allows to compare different autonomous driving networks as well as programs made manually.

This project aims to have a platform for evaluating and testing complex behaviors for different robots using machine learning and deep learning algorithms. This application provides different functionalities such as:

* Loading a simulated environment for different scenarios where you can evaluate complex robots behaviors.
* Generating datasets to train your models to be tested later.
* Evaluate the performance of your models comparing them against a other models.
* Change the scenarios/models (called brains) on the go.
* Live view of sensor readings.

The algorithms that command the robots are called **brains**, and there is where the neural logic (behaviors) is at.

<img src="https://jderobot.github.io/assets/images/projects/neural_behavior/autonomous.jpeg" alt="config" style="zoom:50%;" />

This project presents different approaches to the follow-the-line exercise but using artificial intelligence to complete the circuits. The solutions presented are:

* Using classification networks.
* Using regression networks.
* Using reinforcement learning.
* Solution for real robots.


You can see the project status in the [GitHub repository here](https://github.com/JdeRobot/BehaviorSuite).
