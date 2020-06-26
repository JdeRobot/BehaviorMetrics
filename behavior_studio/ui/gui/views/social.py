#!/usr/bin/env python
""" This module shows the social media links and icons.

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

import webbrowser

import ui.gui.resources.resources

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QSizePolicy, QSpacerItem,
                             QWidget)

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class ClickableLabel(QLabel):
    """Class that extends the functionality of the default QLabel adding mouse events.

    Attributes:
        url {str} -- URL of the social network
    """

    def __init__(self, url):
        """Constructor of the class

        Arguments:
            url {str} -- URL of the social network
        """
        QLabel.__init__(self)
        self.url = url

    def mousePressEvent(self, event):
        """Function that opens the browser and goes to the class url."""
        webbrowser.open(self.url)


class SocialMedia(QWidget):
    """Social media widget. Handles the creation and painting of the social media information"""

    def __init__(self):
        """Constructor of the class"""
        QWidget.__init__(self)

        main_layout = QHBoxLayout()
        self.setMaximumHeight(30)
        self.setContentsMargins(0, 0, 0, 0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        github_label = self.create_button('https://github.com/jderobot/behaviorsuite', ':/assets/github.png')
        twitter_label = self.create_button('https://twitter.com/jderobot', ':/assets/twitter.png')
        youtube_label = self.create_button('https://www.youtube.com/channel/UCgmUgpircYAv_QhLQziHJOQ',
                                           ':/assets/youtube.png')

        horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        main_layout.addItem(horizontalSpacer)
        main_layout.addWidget(github_label, alignment=Qt.AlignRight)
        main_layout.addWidget(youtube_label, alignment=Qt.AlignRight)
        main_layout.addWidget(twitter_label, alignment=Qt.AlignRight)

        self.setLayout(main_layout)

        self.show()

    def create_button(self, url, pixmap):
        """Function that creates a button for an specific social network

        Arguments:
            url {str} -- URL for the social network
            pixmap {QPixmap} -- Icon of the social network

        Returns:
            ui.gui.views.social.ClickableLabel -- Label with the social network logo in it.
        """
        pmap2 = QPixmap(pixmap).scaled(50, 50, Qt.KeepAspectRatio)
        logo = ClickableLabel(url)
        logo.setPixmap(pmap2)
        logo.setFixedSize(30, 30)
        logo.setScaledContents(True)

        return logo
