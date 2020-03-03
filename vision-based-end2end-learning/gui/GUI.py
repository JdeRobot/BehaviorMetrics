
# Based on @frivas
__author__ = 'vmartinezf'

import shutil

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from QLed import QLed
from net.utils.generator import *
from gui.conf_widget import ConfWidget

class MainWindow(QtWidgets.QWidget):

    updGUI = QtCore.pyqtSignal()
    stopSIG = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the real-time
        driving.
        '''

        QtWidgets.QWidget.__init__(self, parent)
        self.resize(800, 1000)
        self.move(150, 50)
        self.setWindowIcon(QtGui.QIcon('gui/resources/jderobot.png'))

        self.updGUI.connect(self.updateGUI)
        self.train_class_params = None
        self.train_reg_params = None
        self.autopilot_enabled = False

        # Menu Bar
        self.menubar = QtWidgets.QMenuBar(self)
        self.menu_config = self.menubar.addMenu('Config')
        self.menu_about = self.menubar.addAction('About')
        self.action_config = self.menu_config.addAction('Network')

        self.confWidget = ConfWidget(self)
        self.action_config.triggered.connect(lambda: self.confWidget.show())
        self.menu_about.triggered.connect(self.aboutWindow)
        
        # Original image label.
        self.camera1 = QtWidgets.QLabel(self)
        self.camera1.resize(450, 350)
        self.camera1.move(50, 50)
        self.camera1.show()

        # Prediction speeds label
        self.predict_v_label = QtWidgets.QLabel(self)
        self.predict_v_label.move(600, 100)
        self.predict_v_label.resize(100, 90)
        newfont = QtGui.QFont("Times", 18, QtGui.QFont.Bold)
        self.predict_v_label.setFont(newfont)
        self.predict_v_label.setText("%d v" % (0))
        self.predict_v_label.show()

        self.predict_w_label = QtWidgets.QLabel(self)
        self.predict_w_label.move(600, 150)
        self.predict_w_label.resize(100, 90)
        self.predict_w_label.setFont(newfont)
        self.predict_w_label.setText("%d w" % (0.0))
        self.predict_w_label.show()

        # Leds for w
        self.w_label = QtWidgets.QLabel(self)
        self.w_label.move(50, 410)
        newfont = QtGui.QFont("Times", 16, QtGui.QFont.Bold)
        self.w_label.setFont(newfont)
        self.w_label.setText("w")
        self.w_label.show()

        self.led_w_1 = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        self.led_w_1.move(70, 410)
        self.led_w_1.value = False
        self.led_w_2 = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        self.led_w_2.move(130, 410)
        self.led_w_2.value = False
        self.led_w_3 = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        self.led_w_3.move(190, 410)
        self.led_w_3.value = False
        self.led_w_4 = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        self.led_w_4.move(250, 410)
        self.led_w_4.value = False
        self.led_w_5 = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        self.led_w_5.move(310, 410)
        self.led_w_5.value = False
        self.led_w_6 = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        self.led_w_6.move(370, 410)
        self.led_w_6.value = False
        self.led_w_7 = QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        self.led_w_7.move(430, 410)
        self.led_w_7.value = False

        # Leds for v
        self.v_label = QtWidgets.QLabel(self)
        self.v_label.move(525, 50)
        self.v_label.setFont(newfont)
        self.v_label.setText("v")
        self.v_label.show()

        self.led_v_1 = QLed(self, onColour=QLed.Blue, shape=QLed.Circle)
        self.led_v_1.move(510, 90)
        self.led_v_1.value = False
        self.led_v_2 = QLed(self, onColour=QLed.Blue, shape=QLed.Circle)
        self.led_v_2.move(510, 150)
        self.led_v_2.value = False
        self.led_v_3 = QLed(self, onColour=QLed.Blue, shape=QLed.Circle)
        self.led_v_3.move(510, 210)
        self.led_v_3.value = False
        self.led_v_4 = QLed(self, onColour=QLed.Blue, shape=QLed.Circle)
        self.led_v_4.move(510, 270)
        self.led_v_4.value = False

        # Play button
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setStyleSheet("QPushButton { background-color: lightgreen}")
        self.pushButton.move(50, 470)
        self.pushButton.resize(450,50)
        self.pushButton.setText('Play Code')
        self.pushButton.clicked.connect(self.playClicked)
        self.pushButton.setCheckable(True)

        # Teleoperator
        self.line = QtCore.QPointF(0, 0)
        self.qimage = QtGui.QImage()
        self.qimage.load('gui/resources/ball.png')
        self.stopSIG.connect(self.stop)
        self.layout = QtWidgets.QLabel()
        self.widthTeleop = 350
        self.heightTeleop = 350
        self.pointxTeleop = 100
        self.pointyTeleop = 550
        self.layout.move(self.pointxTeleop, self.pointyTeleop)
        self.layout.resize(self.widthTeleop, self.heightTeleop)

        # Stop button
        icon = QtGui.QIcon('gui/resources/stop.png')
        self.stopButton = QtWidgets.QPushButton(self)
        self.stopButton.setStyleSheet("QPushButton { background-color: tomato}")
        self.stopButton.move(100, 925)
        self.stopButton.resize(175, 50)
        self.stopButton.setIcon(icon)
        self.stopButton.setText('Stop')
        self.stopButton.clicked.connect(self.stopClicked)

        # Autopilot button
        icon = QtGui.QIcon('gui/resources/stop.png')
        self.autoButton = QtWidgets.QPushButton(self)
        self.autoButton.setStyleSheet("QPushButton { background-color: lightblue}")
        self.autoButton.move(280, 925)
        self.autoButton.resize(175, 50)
        self.autoButton.setIcon(icon)
        self.autoButton.setText('AutoPilot')
        self.autoButton.clicked.connect(self.autopilotClicked)

        # Train/Test group
        self.group = QtWidgets.QGroupBox(self)
        self.group.setStyleSheet(
            """QGroupBox {
                font: bold;
                border: 1px solid silver;
                border-radius: 6px;
                margin-top: 6px;
                padding: 10px 5px 5px 5px
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0px 5px 0px 5px;
            }"""
        )
        self.group.setTitle("Train/Test")
        self.group.move(550,600)
        self.group.setFixedSize(200,300)
        layout = QtWidgets.QVBoxLayout(self)

        # Save button
        self.save = False
        self.saveButton = QtWidgets.QPushButton(self)
        self.saveButton.setText('Save Dataset')
        self.saveButton.clicked.connect(self.saveDataset)

        # Remove button
        self.removeButton = QtWidgets.QPushButton(self)
        self.removeButton.setText('Remove Dataset')
        self.removeButton.clicked.connect(self.removeDataset)

        # See config button
        self.config_set = False
        self.seeConfigButton = QtWidgets.QPushButton(self)
        self.seeConfigButton.setEnabled(False)
        self.seeConfigButton.setText('See configuration')
        self.seeConfigButton.clicked.connect(self.seeConfigClicked)

        # Train button
        self.trained = False
        self.trainButton = QtWidgets.QPushButton(self)
        self.trainButton.setEnabled(False)
        self.trainButton.setText('Train Network')
        self.trainButton.clicked.connect(self.trainClicked)

        # Test button
        self.testButton = QtWidgets.QPushButton(self)
        self.testButton.setEnabled(False)
        self.testButton.setText('Test Network')
        self.testButton.clicked.connect(self.testClicked)

        layout.addWidget(self.saveButton)
        layout.addWidget(self.removeButton)
        layout.addWidget(self.seeConfigButton)
        layout.addWidget(self.trainButton)
        layout.addWidget(self.testButton)

        self.group.setLayout(layout)

        # Logo
        self.logo_label = QtWidgets.QLabel(self)
        self.logo_label.resize(100, 100)
        self.logo_label.move(600, 700)
        self.logo_label.setScaledContents(True)

        logo_img = QtGui.QImage()
        logo_img.load('gui/resources/jderobot.png')
        self.logo_label.setPixmap(QtGui.QPixmap.fromImage(logo_img))
        self.logo_label.show()
        
    def updateGUI(self):
        ''' Updates the GUI for every time the thread change '''
        # We get the original image and display it.

        self.im_prev = self.camera.getImage()
        im = QtGui.QImage(self.im_prev.data, self.im_prev.data.shape[1], self.im_prev.data.shape[0],
                          QtGui.QImage.Format_RGB888)
        self.im_scaled = im.scaled(self.camera1.size())

        self.camera1.setPixmap(QtGui.QPixmap.fromImage(self.im_scaled))

        # We get the v and w
        self.predict_v_label.setText('{0:0.2f}'.format(self.motors.v) + " v")
        self.predict_w_label.setText('{0:0.2f}'.format(self.motors.w) + " w")

        self.turn_on_off_leds()

        if self.autopilot_enabled:
            im = self.autopilot.get_threshold_image()
            im1 = QtGui.QImage(im, im.shape[1], im.shape[0],
                          QtGui.QImage.Format_RGB888)

            self.autopilotdialog_camera.setPixmap(QtGui.QPixmap.fromImage(im1))

        if self.save:
            img = cv2.cvtColor(self.im_prev.data, cv2.COLOR_RGB2BGR)
            save_image(img)
            save_json(self.motors.v, self.motors.w)

    def getCamera(self):
        return self.camera

    def setCamera(self,camera):
        self.camera=camera

    def getMotors(self):
        return self.motors

    def setMotors(self,motors):
        self.motors=motors
    
    def setThreadConnector(self, t_network):
        self.network_connector = t_network

    def stop(self):
        self.line = QtCore.QPointF(0, 0)
        self.repaint()

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:
            x = e.x() - self.pointxTeleop - self.widthTeleop / 2
            y = e.y() - self.pointyTeleop - self.heightTeleop / 2
            self.line = QtCore.QPointF(x, y)
            v_normalized = (1.0 / (self.heightTeleop / 2)) * self.line.y()
            v_normalized = float("{0:.2f}".format(v_normalized))
            w_normalized = (1.0 / (self.widthTeleop / 2)) * self.line.x()
            w_normalized = float("{0:.2f}".format(w_normalized))
            self.setXYValues(w_normalized, v_normalized)
            self.repaint()

    def returnToOrigin(self):
        x = 0
        y = 0
        self.line = QtCore.QPointF(x, y)
        self.repaint()

    def paintEvent(self, e):
        _width = self.widthTeleop
        _height = self.heightTeleop

        width = 2

        painter = QtGui.QPainter(self)

        # Background
        painter.fillRect(self.pointxTeleop, self.pointyTeleop, self.widthTeleop, self.heightTeleop, QtCore.Qt.black)

        # Lines

        pen = QtGui.QPen(QtCore.Qt.blue, width)
        painter.setPen(pen)
        painter.translate(QtCore.QPoint(self.pointxTeleop + _width/2, self.pointyTeleop + _height/2))

        # Axis
        painter.drawLine(QtCore.QPointF(-_width/2, 0),
                         QtCore.QPointF(_width/2, 0))

        painter.drawLine(QtCore.QPointF(0, -_height/2),
                         QtCore.QPointF(0, _height/2))

        # With mouse
        pen = QtGui.QPen(QtCore.Qt.red, width)
        painter.setPen(pen)

        # We check if mouse is in the limits
        if abs(self.line.x() * 2) >= self.widthTeleop:
            if self.line.x() >= 0:
                self.line.setX(self.widthTeleop / 2)
            elif self.line.x() < 0:
                self.line.setX((-self.widthTeleop / 2) + 1)

        if abs(self.line.y() * 2) >= self.heightTeleop:
            if self.line.y() >= 0:
                self.line.setY(self.heightTeleop / 2)
            elif self.line.y() < 0:
                self.line.setY((-self.heightTeleop / 2) + 1)

        painter.drawLine(QtCore.QPointF(self.line.x(), -_width/2),
                         QtCore.QPointF(self.line.x(), _width/2))

        painter.drawLine(QtCore.QPointF(-_height/2, self.line.y()),
                         QtCore.QPointF(_height/2, self.line.y()))

        painter.drawImage(self.line.x() - self.qimage.width() / 2, self.line.y() - self.qimage.height() / 2, self.qimage)

    def playClicked(self):
        if self.pushButton.isChecked():
            self.pushButton.setText('Stop Code')
            self.pushButton.setStyleSheet("background-color: #7dcea0")
            self.network_connector.setPlaying(True)
            self.algorithm.play()
        else:
            self.pushButton.setText('Play Code')
            self.pushButton.setStyleSheet("background-color: #ec7063")
            self.motors.sendV(0)
            self.motors.sendW(0)
            self.network_connector.setPlaying(False)
            self.algorithm.stop()

    def turn_on_off_leds(self):
        self.led_w_1.value = False
        self.led_w_2.value = False
        self.led_w_3.value = False
        self.led_w_4.value = False
        self.led_w_5.value = False
        self.led_w_6.value = False
        self.led_w_7.value = False
        self.led_v_1.value = False
        self.led_v_2.value = False
        self.led_v_3.value = False
        self.led_v_4.value = False

        if self.motors.w <= -1.0:
            self.led_w_7.value = True
        elif -1 < self.motors.w and self.motors.w <= -0.5:
            self.led_w_6.value =  True
        elif -0.5 < self.motors.w and self.motors.w <= -0.1:
            self.led_w_5.value = True
        elif -0.1 < self.motors.w and self.motors.w < 0.1:
            self.led_w_4.value = True
        elif 0.1 <= self.motors.w and self.motors.w < 0.5:
            self.led_w_3.value = True
        elif 0.5 <= self.motors.w and self.motors.w < 1:
            self.led_w_2.value = True
        elif self.motors.w >= 1:
            self.led_w_1.value = True

        if self.motors.v <= 7:
            self.led_v_4.value = True
        elif self.motors.v > 7 and self.motors.v <= 9:
            self.led_v_3.value = True
        elif self.motors.v > 9 and self.motors.v <= 11:
            self.led_v_2.value = True
        elif self.motors.v > 11:
            self.led_v_1.value = True

    def saveDataset(self):
        create_dataset()
        self.save = True
        self.saveButton.setStyleSheet("QPushButton { background-color: green }")

    def removeDataset(self):
        if os.path.exists('Net/Dataset'):
            shutil.rmtree('Net/Dataset')
        self.save = False
        self.saveButton.setStyleSheet("QPushButton { }")

    def setAlgorithm(self, algorithm):
        self.algorithm=algorithm

    def setAutopilot(self, autopilot):
        self.autopilot=autopilot

    def getAlgorithm(self):
        return self.algorithm

    def setXYValues(self,newX,newY):
        myW=-newX*self.motors.getMaxW()
        myV=-newY*self.motors.getMaxV()
        self.motors.sendV(myV)
        self.motors.sendW(myW)
        None

    def stopClicked(self):
        self.motors.sendV(0)
        self.motors.sendW(0)
        self.autopilot_enabled =False
        self.autopilot.stop()
        self.returnToOrigin()

    def closeEvent(self, event):
        self.algorithm.kill()
        self.autopilot.kill()
        self.camera.stop()
        event.accept()
        self.exit()

    def aboutWindow(self):
        about = QtWidgets.QDialog()
        about.setFixedSize(550,350)
        about.setWindowTitle("About JdeRobot")
        logoLayout = QtWidgets.QHBoxLayout()
        text = QtWidgets.QLabel(about)
        str = "<span style='font-size:15pt; font-weight:600;'>Jderobot 5.5.2</span> <br><br>Software suite for robotics and computer vision. <br><br>You can find more info <a href='http://jderobot.org'>here</a><br><br>Github <a href='https://github.com/jderobot/jderobot.git'>repository</a>"
        text.setFixedSize(200, 350)
        text.setWordWrap(True);
        text.setTextFormat(QtCore.Qt.RichText)
        text.setOpenExternalLinks(True)
        text.setText(str)
        logoLayout.addWidget(text, 0, QtCore.Qt.AlignTop)
        about.setLayout(logoLayout)
        about.exec_()
    
    def setClassTrainParams(self,
            variable,
            classes,
            net_model,
            dataset_mode,
            train_cropped
        ):
        self.train_reg_params = None
        self.train_class_params = [variable, classes, net_model, dataset_mode, train_cropped]
        print(self.train_class_params)

    def setRegTrainParams(self, type_image, type_net):
        self.train_class_params = None
        self.train_reg_params = [type_image, type_net]


    def trainClicked(self):
        if self.train_class_params:
            import net.keras.classification.classification_train as classification_train
            classification_train.train(self.train_class_params)
        else:
            import net.keras.regression.regression_train as regression_train
            regression_train.train(self.train_reg_params)
        self.trained = True
        self.testButton.setEnabled(True)


    def testClicked(self):
        if self.train_class_params:
            import net.keras.classification.classification_test as classification_test
            classification_test.test(self.train_class_params)
        else:
            import net.keras.regression.regression_test as regression_test
            regression_test.test(self.train_reg_params)
        
    def autopilotClicked(self):
        self.autopilot_enabled = True
        self.autopilotdialog = QtWidgets.QDialog()
        self.autopilotdialog.setWindowTitle("About JdeRobot")
        layout = QtWidgets.QHBoxLayout()
        self.autopilotdialog_camera = QtWidgets.QLabel(self)
        self.autopilotdialog_camera.resize(450, 350)
        self.autopilotdialog_camera.move(50, 50)
        self.autopilotdialog_camera.show()
        layout.addWidget(self.autopilotdialog_camera, 0, QtCore.Qt.AlignTop)
        self.autopilotdialog.setLayout(layout)

        self.autopilot.play()
        self.autopilotdialog.exec_()
    
    def seeConfigClicked(self):
        about = QtWidgets.QDialog()
        about.setFixedSize(550,350)
        about.setWindowTitle("About JdeRobot")
        logoLayout = QtWidgets.QHBoxLayout()
        text = QtWidgets.QLabel(about)

        if self.train_class_params:
            content = """
                    <table style="width: 49%; margin-right: calc(51%);">
                    <tbody>
                        <tr>
                        <td style="width: 62.5%;"><span style="font-size: 18px;"><strong>Parameters</strong></span></td>
                        <td style="width: 37.2549%;">
                            <div style="text-align: center;"><span style="font-size: 18px;"><strong>Values</strong></span></div>
                        </td>
                        </tr>
                        <tr>
                        <td style="width: 62.5%;">Variable to train</td>
                        <td style="width: 37.2549%;">
                            <div style="text-align: center;">{0}</div>
                        </td>
                        </tr>
                        <tr>
                        <td style="width: 62.5%;">Number of classes</td>
                        <td style="width: 37.2549%;">
                            <div style="text-align: center;">{1}</div>
                        </td>
                        </tr>
                        <tr>
                        <td style="width: 62.5%;">Selected network</td>
                        <td style="width: 37.2549%;">
                            <div style="text-align: center;">{2}</div>
                        </td>
                        </tr>
                        <tr>
                        <td style="width: 62.5%;">Dataset mode</td>
                        <td style="width: 37.2549%;">
                            <div style="text-align: center;">{3}</div>
                        </td>
                        </tr>
                        <tr>
                        <td style="width: 62.5%;">Cropped images</td>
                        <td style="width: 37.2549%;">
                            <div style="text-align: center;">{4}</div>
                        </td>
                        </tr>
                    </tbody>
                    </table>
                    """.format(self.train_class_params[0],
                                    self.train_class_params[1],
                                    self.train_class_params[2],
                                    self.train_class_params[3],
                                    self.train_class_params[4])
        else:
            content = """
            <table style="width: 100%; margin-right: calc(100%);">
            <tbody>
                <tr>
                <td style="width: 50%;"><span style="font-size: 18px;"><strong>Parameters</strong></span></td>
                <td style="width: 100%;">
                    <div style="text-align: center;"><span style="font-size: 18px;"><strong>Values</strong></span></div>
                </td>
                </tr>
                <tr>
                <td style="width: 100%;">Cropped images</td>
                <td style="width: 100%;">
                    <div style="text-align: center;">{0}</div>
                </td>
                </tr>
                <tr>
                <td style="width: 100%;">Selected network</td>
                <td style="width: 100%;">
                    <div style="text-align: center;">{1}</div>
                </td>
                </tr>
            </tbody>
            </table>

            """.format(self.train_reg_params[0], self.train_reg_params[1])
        text.setFixedSize(200, 350)
        text.setWordWrap(True)
        text.setTextFormat(QtCore.Qt.RichText)
        text.setText(content)
        logoLayout.addWidget(text)
        about.setLayout(logoLayout)
        about.exec_()
