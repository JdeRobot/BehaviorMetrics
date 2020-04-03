import webbrowser

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QSizePolicy, QSpacerItem, QWidget)


class ClickableLabel(QLabel):

    def __init__(self, url):
        QLabel.__init__(self)
        self.url = url

    def mousePressEvent(self, event):
        webbrowser.open(self.url)


class SocialMedia(QWidget):

    def __init__(self):
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
        pmap2 = QPixmap(pixmap).scaled(50, 50, Qt.KeepAspectRatio)
        logo = ClickableLabel(url)
        logo.setPixmap(pmap2)
        logo.setFixedSize(30, 30)
        logo.setScaledContents(True)

        return logo
