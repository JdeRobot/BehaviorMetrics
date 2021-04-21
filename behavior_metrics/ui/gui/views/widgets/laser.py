#!/usr/bin/env python
""" This module contains the code to show laser data in the GUI as a widget

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import math

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QFrame

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class LaserWidgetPro(QFrame):
    """Class that defines a Qt widget to show laser data in a frame."""

    def __init__(self, frame_id, parent_width, parent_height, parent=None):
        """Constructor of the class

        Arguments:
            frame_id {str} -- Id of the frame that will show the image
            parent_width {int} -- Parent container width
            parent_height {int} -- Parent container height

        Keyword Arguments:
            parent {ui.gui.views.main_view.MainView} -- Parent of this widget (default: {None})
        """
        QFrame.__init__(self)
        self.laser_data = None
        self.id = frame_id
        self._width = parent_width
        self._height = parent_height
        self.parent = parent
        self.setStyleSheet('background-color: rgb(51,51,51)')
        self.resize(self._width, self._height)

    def paintEvent(self, event):
        """Update the frame with all the new laser information. Updated with the GUI loop"""

        self.laser_data = self.parent.controller.get_data(self.id)
        _width = self.width()
        _height = self.height()

        cx = _width / 2
        cy = _height / 2

        x1 = y1 = d = ang = 0

        width = 2
        pen = QPen(Qt.white, width)

        painter = QPainter(self)
        painter.setPen(pen)

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
