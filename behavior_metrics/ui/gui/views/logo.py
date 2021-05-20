#!/usr/bin/env python
""" This module is responsible for handling the logic of the robot and its current brain.

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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class Logo(QWidget):
    """Class that contains the logo of the application"""

    def __init__(self):
        """Constructor of the class"""
        QWidget.__init__(self)

        self.setFixedHeight(150)
        self.initUI()

    def initUI(self):
        """Setup the GUI elements"""

        main_layout = QVBoxLayout()
        img = QImage(':/assets/logo_100.svg')
        pmap = QPixmap.fromImage(img).scaled(50, 50, Qt.KeepAspectRatio, )
        logo_label = QLabel()
        logo_label.setPixmap(pmap)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedHeight(50)

        title_label = QLabel('Behavior Suite')
        title_label.setFont(QFont('Techno Capture', 30))
        title_label.setStyleSheet('color: white')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(50)

        main_layout.addWidget(logo_label)
        main_layout.addWidget(title_label)

        self.setLayout(main_layout)

        self.show()
