from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys


class ExampleWindow(QMainWindow):
    updGUI = pyqtSignal()

    def __init__(self, sensors):
        super(ExampleWindow, self).__init__()
        self.updGUI.connect(self.update_gui)
        self.windowsize = QSize(1920,1080)
        self.initUI()
        
        self.camera = sensors.get_camera('camera_0')

    def initUI(self):
        self.setFixedSize(self.windowsize)
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.FramelessWindowHint)

        widget = QWidget()
        self.setCentralWidget(widget)
        
        self.image = QLabel()

        layout_box = QHBoxLayout(widget)
        layout_box.setContentsMargins(0, 0, 0, 0)
        layout_box.addWidget(self.image)

        self.image2 = QLabel(widget)
        self.image2.setFixedSize(QSize(500,500))

        p = self.geometry().bottomRight() - self.image2.geometry().bottomRight() - QPoint(100, 100)
        self.image2.move(p)
    
    def update_gui(self):
        self.im_prev = self.camera.getImage()
        im = QImage(self.im_prev.data, self.im_prev.data.shape[1], self.im_prev.data.shape[0],
                          QImage.Format_RGB888)
        pixmap1 = QPixmap.fromImage(im)
        pixmap1 = pixmap1.scaledToWidth(1920)
        pixmap2 = QPixmap.fromImage(im)
        pixmap2 = pixmap2.scaledToWidth(500)
        
        self.image.setPixmap( pixmap1)
        self.image2.setPixmap(pixmap2)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     screensize = QSize(800,600)

#     ex = ExampleWindow(screensize)
#     ex.show()

# sys.exit(app.exec_())