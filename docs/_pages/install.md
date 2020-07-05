---
permalink: /install/

title: "Installation and use"

sidebar:
  nav: "docs"
---

This application depends on dome third party libraries, most of them are included in the requirements file. To install them just type the following:

```bash
pip install -r requirements.txt
```

However since this project is built under python 2.7 the Qt5 dependencies cannot be installed via pip (it's included in the pip3 repositories). So it's needed to install the Qt5 dependencies through apt as follows:

```bash
sudo apt install python-pyqt5 python-pyqt5.qtsvg python-pyqt5.qsci pyqt5-dev-tools
```

Aside from PyQt5, you will need to install also: **jderobot-base**, **jderobot-assets** and **ROS melodic**. To do that, you have a complete guide [**here**](https://jderobot.github.io/RoboticsAcademy/).


[Pretrained network models](http://jderobot.org/store/deeplearning-networks/).
{: .notice--info}
