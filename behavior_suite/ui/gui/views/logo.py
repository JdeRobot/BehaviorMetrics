from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QFont, QPixmap, QImage
import resources


class Logo(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        main_layout = QVBoxLayout()
        self.setFixedHeight(150)

        img = QImage(':/assets/logo_100.svg')
        pmap = QPixmap.fromImage(img).scaled(50, 50, Qt.KeepAspectRatio, )
        logo_label = QLabel()
        logo_label.setPixmap(pmap)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedHeight(50)

        title_label = QLabel('Behavior Suite')
        title_label.setFont(QFont('Techno Capture', 36))
        title_label.setStyleSheet('color: white')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(50)

        main_layout.addWidget(logo_label)
        main_layout.addWidget(title_label)

        self.setLayout(main_layout)

        self.show()
