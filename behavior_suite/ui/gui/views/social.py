import webbrowser

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QSizePolicy, QSpacerItem,
                             QWidget)


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
        self.setContentsMargins(0,0,0,0)
        main_layout.setContentsMargins(0,0,0,0)
        # self.setFixedHeight(60)

        pmap = QPixmap(':/assets/github.png').scaled(50, 50, Qt.KeepAspectRatio)
        logo_label = ClickableLabel('https://github.com/jderobot/behaviorsuite')
        logo_label.setPixmap(pmap)
        logo_label.setFixedSize(30, 30)
        logo_label.setScaledContents(True)
        logo_label.setOpenExternalLinks(True)

        pmap1 = QPixmap(':/assets/youtube.png').scaled(50, 50, Qt.KeepAspectRatio)
        logo_label1 = ClickableLabel('https://www.youtube.com/channel/UCgmUgpircYAv_QhLQziHJOQ')
        logo_label1.setPixmap(pmap1)
        logo_label1.setFixedSize(30, 30)
        logo_label1.setScaledContents(True)

        pmap2 = QPixmap(':/assets/twitter.png').scaled(50, 50, Qt.KeepAspectRatio)
        logo_label2 = ClickableLabel('https://twitter.com/jderobot')
        logo_label2.setPixmap(pmap2)
        logo_label2.setFixedSize(30, 30)
        logo_label2.setScaledContents(True)

        horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        main_layout.addItem(horizontalSpacer)
        main_layout.addWidget(logo_label, alignment=Qt.AlignRight)
        main_layout.addWidget(logo_label1, alignment=Qt.AlignRight)
        main_layout.addWidget(logo_label2, alignment=Qt.AlignRight)

        self.setLayout(main_layout)

        self.show()
