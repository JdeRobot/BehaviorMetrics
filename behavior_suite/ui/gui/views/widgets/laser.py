#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (C) 1997-2016 JDE Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/.
#  Authors :
#       Carlos Awadallah Estévez<carlosawadallah@gmail.com>

from PyQt5.QtWidgets import QWidget, QGridLayout, QGraphicsView, QGraphicsScene, QFrame
from PyQt5.QtGui import QPen, QPainter, QPalette
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5 import QtGui, QtCore
import numpy as np
import math



class LaserWidgetPro(QFrame):

    def __init__(self, id, w, h, parent=None):
        QFrame.__init__(self)
        self.laser_data = None
        self.id = id
        self._width = w
        self._height = h
        self.parent = parent
        # self.mutex = Lock()
        # self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: white')
        # palette = QPalette()
        # self.setPalette(palette)
        # self.setMaximumSize(500, 500)
        # self.setMinimumSize(500, 500)
        print(self._width, self._height)
        self.resize(self._width, self._height)

    def paintEvent(self, event):
        self.laser_data = self.parent.controller.get_data(self.id)
        _width = self.width()
        _height = self.height()

        cx = _width / 2
        cy = _height / 2

        x1 = y1 = d = ang = 0

        width = 6
        pen = QPen(Qt.blue, width)

        painter = QPainter(self)
        painter.setPen(pen)
        # painter.drawPoint(QPoint(100, 100))
        # painter.drawLine(QPointF(0,0), QPoint(100,100))
        if self.laser_data:
            if len(self.laser_data.values) > 0:
                step = (self.laser_data.maxAngle - self.laser_data.minAngle) / len(self.laser_data.values)
                d = self.laser_data.maxRange / (_width / 2)
                ang = self.laser_data.minAngle
                for i in range(len(self.laser_data.values)):
                    ang = self.laser_data.minAngle + i * step
                    x1 = cx + (self.laser_data.values[i] / d) * math.cos(ang)
                    y1 = cy - (self.laser_data.values[i] / d) * math.sin(ang)

                    painter.drawPoint(QPointF(x1, y1))
                    # print('painting point at ', x1, ',', y1)

