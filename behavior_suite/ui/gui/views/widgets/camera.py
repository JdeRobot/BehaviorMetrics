from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel,  QVBoxLayout, QWidget
from threading import Lock


class CameraWidget(QWidget):

    signal_update = pyqtSignal()

    def __init__(self, id, w, h, keep_ratio, parent=None):
        QWidget.__init__(self, parent)
        self.parent = parent
        self.id = id
        self.signal_update.connect(self.update)
        self.setMaximumSize(1000, 1000)
        self.keep_ratio = keep_ratio
        self.parent_width = w
        self.parent_height = h
        self.lock_update = Lock()
        self.initUI()

    def initUI(self):
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
