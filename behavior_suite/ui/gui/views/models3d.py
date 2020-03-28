from PyQt5.Qt3DCore import QTransform, QEntity
from PyQt5.Qt3DExtras import Qt3DWindow, QPhongMaterial, QOrbitCameraController
from PyQt5.Qt3DRender import QPointLight, QMesh
from PyQt5.QtCore import pyqtSignal, pyqtProperty, QUrl, QPropertyAnimation
from PyQt5.QtGui import QMatrix4x4, QVector3D, QColor
from PyQt5.QtWidgets import QWidget, QHBoxLayout


def fuzzyCompareDouble(p1, p2):
    """
    compares 2 double as points
    """
    return abs(p1 - p2) * 100000. <= min(abs(p1), abs(p2))


class OrbitTransformController(QTransform):
    targetChanged = pyqtSignal()
    angleChanged = pyqtSignal()
    radiusChanged = pyqtSignal()

    def __init__(self, parent):
        QTransform.__init__(self, parent)
        self.m_target = QTransform()
        self.m_matrix = QMatrix4x4()
        self.m_radius = 1.0
        self.m_angle = 0.0

    def target(self):
        return self.m_target

    def setTarget(self, target):
        if self.m_target == target:
            return
        self.m_target = target
        self.targetChanged.emit()

    def setRadius(self, radius):
        if fuzzyCompareDouble(radius, self.m_radius):
            return
        self.m_radius = radius
        self.radiusChanged.emit()

    def radius(self, ):
        return self.m_radius

    def setAngle(self, angle):
        if fuzzyCompareDouble(angle, self.m_angle):
            return
        self.m_angle = angle
        self.updateMatrix()
        self.angleChanged.emit()

    def angle(self):
        return self.m_angle

    def updateMatrix(self, ):
        self.m_matrix.setToIdentity()
        self.m_matrix.rotate(self.m_angle, QVector3D(0.0, 1.0, 0.0))
        self.m_matrix.translate(self.m_radius, 0.0, 0.0)
        self.m_target.setMatrix(self.m_matrix)

    angle = pyqtProperty(float, fget=angle, fset=setAngle, notify=angleChanged)
    radius = pyqtProperty(float, fget=radius, fset=setRadius, notify=radiusChanged)
    target = pyqtProperty(QTransform, fget=target, fset=setTarget, notify=angleChanged)


class View3D(QWidget):
    def __init__(self, robot_type, parent=None):
        super(View3D, self).__init__(parent)
        self.view = Qt3DWindow()
        self.parent = parent
        self.view.defaultFrameGraph().setClearColor(QColor(51, 51, 51))
        self.container = self.createWindowContainer(self.view)
        self.setStyleSheet('background-color: white')
        self.robot_type = robot_type
        self.robot_entity = None
        self.setMouseTracking(True)

        vboxlayout = QHBoxLayout()
        vboxlayout.addWidget(self.container)
        self.setLayout(vboxlayout)

        self.scene = self.createScene()

        # Camera.
        self.initialiseCamera(self.scene)

        self.view.setRootEntity(self.scene)

    #     t1 = threading.Thread(target=self.print_campos)
    #     t1.start()

    # def print_campos(self):
    #     while True:
    #         print(self.robot_type, self.camera.position())
    #         time.sleep(0.5)

    def initialiseCamera(self, scene):
        # Camera.
        self.camera = self.view.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16.0 / 9.0, 0.1, 1000.0)
        if self.robot_type == 'car':
            self.camera.setPosition(QVector3D(0.3, 1.5, 4.0))
        elif self.robot_type == 'f1':
            self.camera.setPosition(QVector3D(0.3, 1.7, 4.5))
        elif self.robot_type == 'drone' or self.robot_type == 'drone_l':
            self.camera.setPosition(QVector3D(0.2, 0.1, 0.5))
        elif self.robot_type == 'roomba':
            self.camera.setPosition(QVector3D(0.0, 0.2, 0.6))
        elif self.robot_type == 'turtlebot':
            self.camera.setPosition(QVector3D(0.0, 0.4, 0.8))
        elif self.robot_type == 'pepper':
            self.camera.setPosition(QVector3D(0.17, 1.3, 1.6))

        if self.robot_type == 'pepper':
            self.camera.setViewCenter(QVector3D(0.0, 0.6, 0.0))
        elif self.robot_type == 'turtlebot':
            self.camera.setViewCenter(QVector3D(0.0, 0.1, 0.0))
        else:
            self.camera.setViewCenter(QVector3D(0.0, 0.0, 0.0))

    def activate_camera(self, scene):
        # # For camera controls.
        camController = QOrbitCameraController(scene)
        camController.setLinearSpeed(250.0)
        camController.setLookSpeed(250.0)
        camController.setCamera(self.camera)

    def change_window(self):
        print('finished robots, emiting---')
        self.parent.emit_and_destroy()
        self.sphereRotateTransformAnimation.deleteLater()

    def start_animation(self):
        self.stop_animation()
        self.sphereRotateTransformAnimation.start()

    def start_animation_with_duration(self, duration):
        self.stop_animation()
        self.sphereRotateTransformAnimation.setDuration(duration)
        self.sphereRotateTransformAnimation.setLoopCount(1)
        self.start_animation()
        self.sphereRotateTransformAnimation.finished.connect(self.change_window)

    def stop_animation(self):
        self.sphereRotateTransformAnimation.stop()

    def set_animation_speed(self, speed):
        """ speed of a 360deg rotation in seconds """
        self.sphereRotateTransformAnimation.setDuration(speed)

    def createScene(self):
        # Root entity
        rootEntity = QEntity()

        light_entity = QEntity(rootEntity)
        light = QPointLight(light_entity)
        light.setColor(QColor(255, 255, 255))
        light.setIntensity(0.6)
        trans = QTransform()
        trans.setTranslation(QVector3D(0, 0, 2))

        light_entity.addComponent(trans)
        light_entity.addComponent(light)

        # Material
        material = QPhongMaterial(rootEntity)
        material.setAmbient(QColor(100, 100, 100))
        # material.setShininess(0)

        self.robot_entity = QEntity(rootEntity)
        f1_mesh = QMesh()
        f1_mesh.setSource(QUrl('qrc:/assets/'+self.robot_type+'.obj'))

        self.robot_entity.addComponent(f1_mesh)

        self.robot_entity.addComponent(material)

        sphereTransform = QTransform()

        controller = OrbitTransformController(sphereTransform)
        controller.setTarget(sphereTransform)
        controller.setRadius(0.0)

        self.sphereRotateTransformAnimation = QPropertyAnimation(sphereTransform)
        self.sphereRotateTransformAnimation.setTargetObject(controller)
        self.sphereRotateTransformAnimation.setPropertyName(b"angle")
        self.sphereRotateTransformAnimation.setStartValue(0)
        self.sphereRotateTransformAnimation.setEndValue(360)
        self.sphereRotateTransformAnimation.setDuration(10000)
        self.sphereRotateTransformAnimation.setLoopCount(-1)
        self.robot_entity.addComponent(sphereTransform)
        self.start_animation()

        return rootEntity
