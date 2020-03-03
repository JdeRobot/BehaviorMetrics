from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from network_configurator import NetworkConfigurator

class ConfWidget(QtWidgets.QWidget):

    
    def __init__(self,winParent, width=300, height=300):    
        super(ConfWidget, self).__init__()
        self.winParent=winParent
        self.groupboxstyle = """QGroupBox {
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
        # execution parameters
        self.net_configurator = NetworkConfigurator()
        self.net_framework = 'Keras'
        self.net_type = 'Classification'
        self.net_cropped = 'cropped'
        self.net_model_v = None
        self.net_model_w = None

        # classification train parameters
        self.variable = 'v'
        self.classes = '4'
        self.net_model = 'other'
        self.dataset_mode = 'normal'
        self.train_cropped = 'cropped'

        # regression train parameters
        self.type_image = 'cropped'
        self.type_net = 'pilotnet'

        self.initUI()


    def initUI(self):

        self.setWindowTitle("Network_configuration")
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
        l.addWidget(keras_rb, alignment=QtCore.Qt.AlignCenter)
        l.addWidget(tf_rb, alignment=QtCore.Qt.AlignCenter)
        l.addWidget(torch_rb, alignment=QtCore.Qt.AlignCenter)
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
        l.addWidget(clf_rb, alignment=QtCore.Qt.AlignCenter)
        l.addWidget(reg_rb, alignment=QtCore.Qt.AlignCenter)
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

        self.cropped_check = QtWidgets.QCheckBox("Cropped")
        layout.addWidget(self.cropped_check, 3, 0)
        self.cropped_check.clicked.connect(lambda: self.cropped_checked(self.cropped_check))

        self.lbl = QtWidgets.QLabel()
        self.lbl.setStyleSheet("QLabel { font-weight: bold; font-size: 18px; color: green}")
        self.lbl.setText("Setting up network...")
        self.lbl.hide()
        # self.movie = QtGui.QMovie('/tmp/loader.gif')
        # self.lbl.setMovie(self.movie)
        # self.movie.start()
        layout.addWidget(self.lbl, 4, 0)

        save_button = QtWidgets.QPushButton("Save")
        layout.addWidget(save_button, 4, 0, alignment=QtCore.Qt.AlignRight)
        save_button.clicked.connect(self.saveBtnClk)
      
        self.exec_widget.setLayout(layout)

    def trainClassTabUI(self):
        layout = QtWidgets.QGridLayout()

        # Variable section
        variable_groupbox = QtWidgets.QGroupBox()
        variable_groupbox.setTitle("Select variable")
        variable_groupbox.setStyleSheet(self.groupboxstyle)
        l = QtWidgets.QHBoxLayout()
        
        linear_rb = QtWidgets.QRadioButton("Linear velocity (v)")
        angular_rb = QtWidgets.QRadioButton("Angular velocity (w)")
        linear_rb.clicked.connect(lambda: self.rb_toggled(linear_rb, 'train_v'))
        angular_rb.clicked.connect(lambda: self.rb_toggled(angular_rb, 'train_w'))
        l.addWidget(linear_rb, alignment=QtCore.Qt.AlignCenter)
        l.addWidget(angular_rb, alignment=QtCore.Qt.AlignCenter)
        variable_groupbox.setLayout(l)
        layout.addWidget(variable_groupbox, 0, 0)

        # classes section
        classes_groupbox = QtWidgets.QGroupBox()
        classes_groupbox.setTitle("Number of classes")
        classes_groupbox.setStyleSheet(self.groupboxstyle)
        l = QtWidgets.QHBoxLayout()

        self.cb_v = QtWidgets.QComboBox()
        self.cb_v.setEnabled(False)
        self.cb_v.addItems(["4", "5"])
        self.cb_v.currentIndexChanged.connect(lambda: self.selectionchange(self.cb_v))
        self.cb_w = QtWidgets.QComboBox()
        self.cb_w.setEnabled(False)
        self.cb_w.addItems(["2", "7", "9"])
        self.cb_w.currentIndexChanged.connect(lambda: self.selectionchange(self.cb_w))
		
        l.addWidget(self.cb_v, alignment=QtCore.Qt.AlignCenter)
        l.addWidget(self.cb_w, alignment=QtCore.Qt.AlignCenter)
        classes_groupbox.setLayout(l)
        layout.addWidget(classes_groupbox, 1, 0)

        # dataset type section
        dataset_groupbox = QtWidgets.QGroupBox()
        dataset_groupbox.setTitle("Select dataset mode")
        dataset_groupbox.setStyleSheet(self.groupboxstyle)
        l = QtWidgets.QHBoxLayout()
        normal_rb = QtWidgets.QRadioButton("Normal")
        balanced_rb = QtWidgets.QRadioButton("Balanced")
        biased_rb = QtWidgets.QRadioButton("Biased")
        normal_rb.clicked.connect(lambda: self.rb_toggled(normal_rb, 'dataset'))
        balanced_rb.clicked.connect(lambda: self.rb_toggled(balanced_rb, 'dataset'))
        biased_rb.clicked.connect(lambda: self.rb_toggled(biased_rb, 'dataset'))
        l.addWidget(normal_rb, alignment=QtCore.Qt.AlignCenter)
        l.addWidget(balanced_rb, alignment=QtCore.Qt.AlignCenter)
        l.addWidget(biased_rb, alignment=QtCore.Qt.AlignCenter)
        dataset_groupbox.setLayout(l)
        layout.addWidget(dataset_groupbox, 2, 0)

        # networks section
        type_groupbox = QtWidgets.QGroupBox()
        type_groupbox.setTitle("Select network type")
        type_groupbox.setStyleSheet(self.groupboxstyle)
        l = QtWidgets.QHBoxLayout()
        
        self.listWidget = QtWidgets.QListWidget(self)
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        for elem in ['CNN', 'LeNet5', 'SmallerVGGNet']:
            item = QtWidgets.QListWidgetItem(self.listWidget)
            item.setText("{0}".format(elem))
            self.listWidget.addItem(item)

        self.listWidget.itemSelectionChanged.connect(self.itemchanged)
        l.addWidget(self.listWidget)
        type_groupbox.setLayout(l)
        layout.addWidget(type_groupbox, 3, 0)        

        cropped_check = QtWidgets.QCheckBox("Cropped")
        layout.addWidget(cropped_check, 4, 0)
        cropped_check.clicked.connect(lambda: self.cropped_check_train(cropped_check))

        save_button = QtWidgets.QPushButton("Save")
        layout.addWidget(save_button, 5, 0, alignment=QtCore.Qt.AlignRight)
        save_button.clicked.connect(self.saveBtnClkTrain)
      
        self.train_class_widget.setLayout(layout)

    def trainRegTabUI(self):
        layout = QtWidgets.QGridLayout()

        # networks section
        type_groupbox = QtWidgets.QGroupBox()
        type_groupbox.setTitle("Select network type")
        type_groupbox.setStyleSheet(self.groupboxstyle)
        l = QtWidgets.QHBoxLayout()
        
        self.listWidget_reg = QtWidgets.QListWidget(self)
        self.listWidget_reg.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        for elem in ['PilotNet', 'Stacked PilotNet', 'Tiny PilotNet', 'LSTM Tiny PilotNet', 'DeepLSTM Tiny PilotNet', 'LSTM', 'ControlNet', 'Temporal', 'Stacked Temporal']:
            item = QtWidgets.QListWidgetItem(self.listWidget_reg)
            item.setText("{0}".format(elem))
            self.listWidget_reg.addItem(item)

        self.listWidget_reg.itemSelectionChanged.connect(self.itemchanged_reg)
        l.addWidget(self.listWidget_reg)
        type_groupbox.setLayout(l)
        layout.addWidget(type_groupbox, 0, 0)        

        cropped_check = QtWidgets.QCheckBox("Cropped")
        layout.addWidget(cropped_check, 1, 0)
        cropped_check.clicked.connect(lambda: self.cropped_check_train(cropped_check))

        save_button = QtWidgets.QPushButton("Save")
        layout.addWidget(save_button, 4, 0, alignment=QtCore.Qt.AlignRight)
        save_button.clicked.connect(self.saveBtnClkTrain2)
      
        self.train_reg_widget.setLayout(layout)


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
        self.lbl.show()
        self.repaint()
        if self.cropped_check.isChecked():
            self.net_cropped = True
        else:
            self.net_cropped = False
        network = self.net_configurator.config_from_gui(
            self.net_framework,
            self.net_type,
            self.net_cropped,
            self.net_model_v,
            self.net_model_w
        )

        self.winParent.getAlgorithm().setNetwork(network)
        self.winParent.network_connector.setNetworkRuntime(network)
        self.close()
    
    def saveBtnClkTrain(self):
        self.winParent.setClassTrainParams(
            self.variable,
            self.classes,
            self.net_model,
            self.dataset_mode,
            self.train_cropped
        )
        self.winParent.config_set = True
        self.winParent.seeConfigButton.setEnabled(True)
        self.winParent.trainButton.setEnabled(True)
        self.close()

    def saveBtnClkTrain2(self):
        self.winParent.setRegTrainParams(self.type_image, self.type_net )
        self.winParent.config_set = True
        self.winParent.seeConfigButton.setEnabled(True)
        self.winParent.trainButton.setEnabled(True)
        self.close()
    
    def rb_toggled(self, b, mode):
        if mode == 'framework':
            self.net_framework = b.text()
        elif mode == 'type':
            self.net_type = b.text()
        elif mode == 'train_v':
            self.variable = 'v'
            self.cb_v.setEnabled(True)
            self.cb_w.setEnabled(False)
        elif mode == 'train_w':
            self.variable = 'w'
            self.cb_v.setEnabled(False)
            self.cb_w.setEnabled(True)
        elif mode == 'dataset':
            self.dataset_mode = b.text().lower()
    
    def cropped_checked(self, b):
        if b.isChecked():
            self.net_cropped = True
        else:
            self.net_cropped = False
    
    def cropped_check_train(self, b):
        if b.isChecked():
            self.train_cropped = 'cropped'
            self.type_image = 'cropped'
        else:
            self.train_cropped = 'normal'
            self.type_image = 'normal'
    
    def selectionchange(self, cb):
        self.classes = int(cb.currentText())

    def itemchanged(self):
        name = self.listWidget.selectedItems()[0].text()
        if name == 'CNN':
            net_name = 'other'
        elif name == 'SmallerVGGNet':
            net_name = 'smaller_vgg'
        elif name == 'LeNet5':
            net_name = 'lenet'

        self.net_model = net_name

    def itemchanged_reg(self):
        name = self.listWidget_reg.selectedItems()[0].text()
        if name == 'PilotNet':
            net_name = 'pilotnet'
        elif name == 'Stacked PilotNet':
            net_name = 'stacked'
        elif name == 'Tiny PilotNet':
            net_name = 'tinypilotnet'
        elif name == 'LSTM Tiny PilotNet':
            net_name = 'lstm_tinypilotnet'
        elif name == 'DeepLSTM Tiny PilotNet':
            net_name = 'deepestlstm_tinypilotnet'
        elif name == 'LSTM':
            net_name = 'lstm'
        elif name == 'ControlNet':
            net_name = 'controlnet'
        elif name == 'Temporal':
            net_name = 'temporal'
        elif name == 'Stacked Temporal':
            net_name = 'stacked_dif'
        
        self.type_net = net_name
              