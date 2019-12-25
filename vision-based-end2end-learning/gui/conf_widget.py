from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from network_configurator import NetworkConfiurator

class ConfWidget(QtWidgets.QWidget):

    
    def __init__(self,winParent, width=300, height=300):    
        super(ConfWidget, self).__init__()
        self.winParent=winParent
        self.groupboxstyle = """QGroupBox {
                        font: bold;
                        border: 1px solid silver;
                        border-radius: 6px;
                        margin-top: 6px;
                    }

                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 7px;
                        padding: 0px 5px 0px 5px;
                    }"""
        self.net_configurator = NetworkConfiurator()
        self.net_framework = 'Keras'
        self.net_type = 'Classification'
        self.net_cropped = False
        self.net_model_v = None
        self.net_model_w = None

        self.initUI()


    def initUI(self):

        self.setMinimumSize(680, 500)
        self.setMaximumSize(680, 500)
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.exec_widget = QtWidgets.QWidget()
        self.train_class_widget = QtWidgets.QWidget()
        self.train_reg_widget = QtWidgets.QWidget()

        tabwidget = QtWidgets.QTabWidget()
        tabwidget.addTab(self.exec_widget, "Execution configuration")
        tabwidget.addTab(self.train_class_widget, "Train Calssification")
        tabwidget.addTab(self.train_reg_widget, "Train Regression")
        self.execTabUI()
        self.trainClassTabUI()
        self.trainRegTabUI()
        layout.addWidget(tabwidget, 0, 0)

    def execTabUI(self):
        layout = QtWidgets.QGridLayout()

        # Framework section
        framework_groupbox = QtWidgets.QGroupBox()
        framework_groupbox.setTitle("Select framework")
        framework_groupbox.setStyleSheet(self.groupboxstyle)
        l = QtWidgets.QHBoxLayout()
        keras_rb = QtWidgets.QRadioButton("Keras")
        tf_rb = QtWidgets.QRadioButton("TensorFlow")
        torch_rb = QtWidgets.QRadioButton("Pytorch")
        keras_rb.clicked.connect(lambda: self.rb_toggled(keras_rb, 'framework'))
        tf_rb.clicked.connect(lambda: self.rb_toggled(tf_rb, 'framework'))
        torch_rb.clicked.connect(lambda: self.rb_toggled(torch_rb, 'framework'))
        l.addWidget(keras_rb)
        l.addWidget(tf_rb)
        l.addWidget(torch_rb)
        framework_groupbox.setLayout(l)
        layout.addWidget(framework_groupbox, 0, 0)

        # Network type section
        type_groupbox = QtWidgets.QGroupBox()
        type_groupbox.setTitle("Select network type")
        type_groupbox.setStyleSheet(self.groupboxstyle)
        l = QtWidgets.QHBoxLayout()
        clf_rb = QtWidgets.QRadioButton("Classification")
        reg_rb = QtWidgets.QRadioButton("Regression")
        clf_rb.clicked.connect(lambda: self.rb_toggled(clf_rb, 'type'))
        reg_rb.clicked.connect(lambda: self.rb_toggled(reg_rb, 'type'))
        l.addWidget(clf_rb)
        l.addWidget(reg_rb)
        type_groupbox.setLayout(l)
        layout.addWidget(type_groupbox, 1, 0)

        # Models path section
        models_groupbox = QtWidgets.QGroupBox()
        models_groupbox.setTitle("Model paths")
        models_groupbox.setStyleSheet(self.groupboxstyle)
        lv = QtWidgets.QVBoxLayout()
        lh_v = QtWidgets.QHBoxLayout()
        lh_w = QtWidgets.QHBoxLayout()

        self.v_label = QtWidgets.QLabel("No model selected for v")
        self.w_label = QtWidgets.QLabel("No model selected for w")
        v_btn = QtWidgets.QPushButton("Browse...")
        v_btn.clicked.connect(lambda: self.getFiles('v'))
        w_btn = QtWidgets.QPushButton("Browse...")
        w_btn.clicked.connect(lambda: self.getFiles('w'))

        lh_v.addWidget(self.v_label)
        lh_v.addWidget(v_btn)
        lh_w.addWidget(self.w_label)
        lh_w.addWidget(w_btn)
        lv.addLayout(lh_v)
        lv.addLayout(lh_w)

        models_groupbox.setLayout(lv)
        layout.addWidget(models_groupbox, 2, 0)

        cropped_check = QtWidgets.QCheckBox("Cropped")
        layout.addWidget(cropped_check, 3, 0)
        cropped_check.clicked.connect(lambda: self.cropped_checked(cropped_check))

        save_button = QtWidgets.QPushButton("Save")
        layout.addWidget(save_button, 4, 0, alignment=QtCore.Qt.AlignRight)
        save_button.clicked.connect(self.saveBtnClk)
      
        self.exec_widget.setLayout(layout)

    def trainClassTabUI(self):
        pass

    def trainRegTabUI(self):
        pass


    def closeEvent(self, event):
        self.close()

    def getFiles(self, opt):
        dlg = QtWidgets.QFileDialog(caption='Open model file')
        dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
        dlg.setNameFilter("HDF5 models (*.h5)")
        dlg.fileMode()
        filenames = None
		
        if dlg.exec_():
            filenames = dlg.selectedFiles()

        if filenames:
            if opt == 'v':
                self.v_label.setText(filenames[0])
                self.net_model_v = filenames[0]
            else:
                self.w_label.setText(filenames[0])
                self.net_model_w = filenames[0]

    def saveBtnClk(self):
        network = self.net_configurator.config_from_gui(
            self.net_framework,
            self.net_type,
            self.net_cropped,
            self.net_model_v,
            self.net_model_w
        )
        self.winParent.getAlgorithm().setNetwork(network)
    
    def rb_toggled(self, b, mode):
        if mode == 'framework':
            self.net_framework = b.text()
        elif mode == 'type':
            self.net_type = b.text()
    
    def cropped_checked(self, b):
        if b.isChecked():
            print("True")
            self.net_cropped = True
        else:
            print("False")
            self.net_cropped = False

              
