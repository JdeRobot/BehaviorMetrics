#!/usr/bin/env python
""" This module contains the code to show an image in the GUI as a widget

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

from threading import Lock

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class CameraWidget(QWidget):
    """Class that defines a Qt widget to show an image in a frame."""

    signal_update = pyqtSignal()

    def __init__(self, frame_id, parent_width, parent_height, keep_ratio, parent=None):
        """Constructor of the class

        Arguments:
            frame_id {str} -- Id of the frame that will show the image
            parent_width {int} -- Parent container width
            parent_height {int} -- Parent container height
            keep_ratio {bool} -- Flag to determine if the image should maintain aspect ratio or not

        Keyword Arguments:
            parent {ui.gui.views.main_view.MainView} -- Parent of this widget (default: {None})
        """
        QWidget.__init__(self, parent)
        self.parent = parent
        self.id = frame_id
        self.signal_update.connect(self.update)
        self.setMaximumSize(1000, 1000)
        self.keep_ratio = keep_ratio
        self.parent_width = parent_width
        self.parent_height = parent_height
        self.lock_update = Lock()
        self.initUI()

    def initUI(self):
        """Inintialize the widget elements"""

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 5, 0, 0)
        self.image_label = QLabel()
        self.image_label.setMouseTracking(True)

        self.image_label.setPixmap(QPixmap(':/assets/logo_200.svg'))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        self.main_layout.addWidget(self.image_label)
        self.setLayout(self.main_layout)

    def update(self):
        """Update the widget with the GUI loop"""

        image = self.parent.controller.get_data(self.id)
        if image is not None:
            with self.lock_update:
                im = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(im)
                scale_width = self.parent_width
                scale_height = self.parent_height
                if self.keep_ratio:
                    pixmap = pixmap.scaled(scale_width, scale_height, Qt.KeepAspectRatio)
                else:
                    pixmap = pixmap.scaled(scale_width, scale_height)
                self.image_label.setPixmap(pixmap)
