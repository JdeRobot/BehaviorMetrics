from brains.CARLA.utils.test_utils import (
    traffic_light_to_int,
    model_control,
    calculate_delta_yaw,
)
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from brains.CARLA.utils.high_level_command import HighLevelCommandLoader
from os import path
from utils.logger import logger
from traceback import print_exc

import numpy as np

import torch
import time
import math
import carla

from torchvision.models import resnet18, ResNet18_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.nn as nn

from torchvision import transforms
import cv2
import onnxruntime as ort

import logging
# logging.basicConfig(level=logging.INFO)


PRETRAINED_MODELS = ROOT_PATH + "/" + PRETRAINED_MODELS_DIR + "CARLA/"

# ----- PREPROCESAMIENTO -----
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# TIME_CYCLE = 0.1  # seconds, 10 Hz

class Brain:

    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        self.motors = actuators.get_motor("motors_0")
        self.camera_rgb = sensors.get_camera("camera_0")  # rgb front view camera
        self.camera_seg = sensors.get_camera("camera_2")  # segmentation camera
        self.handler = handler
        self.inference_times = []
        self.gpu_inference = config["GPU"]
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.gpu_inference) else "cpu"
        )
        self.red_light_counter = 0
        self.running_light = False

        client = carla.Client("localhost", 2000)
        client.set_timeout(100.0)
        self.world = client.get_world()
        self.map = self.world.get_map()
        
        # contar ticks
        # self.delta = self.world.get_settings().fixed_delta_seconds or 0.05
        # self.prev_frame = None
        
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

        self.vehicle = None
        while self.vehicle is None:
            for vehicle in self.world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name") == "ego_vehicle":
                    self.vehicle = vehicle
                    break
            if self.vehicle is None:
                print("Waiting for vehicle with role_name 'ego_vehicle'")
                time.sleep(1)  # sleep for 1 second before checking again

        if model:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            monolithic_model_path = PRETRAINED_MODELS + model
            print("Loading model from: ", monolithic_model_path)
            
            # onnx model
            providers = [('CUDAExecutionProvider', {})] if ort.get_available_providers().__contains__('CUDAExecutionProvider') else ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(str(monolithic_model_path), providers=providers)
            
            print(f"onnx ort session providers: {self.ort_session.get_providers()}")

            # nombre de la(s) entrada(s) y salida(s)
            self._in_name  = self.ort_session.get_inputs()[0].name
            self._out_name = self.ort_session.get_outputs()[0].name
            
        if "Route" in config:
            route = config["Route"]
            print("route: ", route)
        else:
            route = None

        self.hlc_loader = HighLevelCommandLoader(self.vehicle, self.map, route=route)
        self.prev_hlc = 0
        self.prev_yaw = None
        self.delta_yaw = 0

        self.target_point = [160.0,108.3,0.42,0.00,0.00,180.00]
        self.termination_code = 0  # 0: not terminated; 1: arrived at target; 2: wrong turn
    
        self._last_tick = None # para calcular el tiempo entre ticks
        
        ## debugging, tick time calculation
        # Al cargar el modelo ONNX (por ejemplo en __init__ o setup)
        dummy_input = np.random.rand(1, 3, 66, 200).astype(np.float32)   # Tamaño esperado

        for _ in range(10):
            _ = self.ort_session.run([self._out_name], {self._in_name: dummy_input})


    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        if data.shape[0] != data.shape[1]:
            if data.shape[0] > data.shape[1]:
                difference = data.shape[0] - data.shape[1]
                extra_left, extra_right = int(difference / 2), int(difference / 2)
                extra_top, extra_bottom = 0, 0
            else:
                difference = data.shape[1] - data.shape[0]
                extra_left, extra_right = 0, 0
                extra_top, extra_bottom = int(difference / 2), int(difference / 2)

            data = np.pad(
                data,
                ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        self.handler.update_frame(frame_id, data)


    def predict_controls(self, image_seg): # se saco model como argumento
        # Asegurarse de que sea una imagen BGR como la cargada por cv2.imread
        if image_seg is None:
            raise ValueError("Segmented image is None (sensor may not be ready).")

        if not isinstance(image_seg, np.ndarray):
            print(f"[DEBUG] Segmentation image type: {type(image_seg)}")
            
            
        calzada_color =np.array([128, 64, 128], dtype=np.uint8)
        mask = cv2.inRange(image_seg, calzada_color, calzada_color)
        
        masked_image = np.zeros_like(image_seg)
        masked_image[mask > 0] = [255, 255, 255]
        
        # cropped = masked_image[200:-1, :]
        # resized = cv2.resize(masked_image[200:-1, :], (200, 66))   # (66,200) to (240,640) # funciona bien
        
        h = masked_image.shape[0]
        y0 = min(200, max(0, h - 2))           # evita salirte si h < 200
        crop = masked_image[y0:-1, :]
        if crop is None or crop.size == 0:     # fallback si el recorte quedó vacío
            crop = masked_image
        resized = cv2.resize(crop, (200, 66))  # OpenCV: (width, height)
     
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        rgb_like = cv2.merge([gray, gray, gray])
        
        # rgb_like /= 255.0
        
        input_tensor = torch.tensor(rgb_like, dtype=torch.float32).permute(2, 0, 1) 
        
        input_np = input_tensor.unsqueeze(0).cpu().numpy()        # [1,3,66,200] float32
        
        start_infer = time.perf_counter() # for debbuging, tick time calculation
        
        # ONNX inference
        pred = self.ort_session.run([self._out_name], {self._in_name: input_np})[0]  # → (1,2)
        
        ## Debugging
        infer_time = (time.perf_counter() - start_infer) * 1000  # en ms
        # print(f"Inferencia ONNX: {infer_time:.3f} ms")
        

        steer, throttle = pred[0].astype(np.float32)
        # print(f"steer: {steer}, throttle: {throttle}")

        return float(steer), float(throttle) 

    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""      
        rgb_image = self.camera_rgb.getImage()  #.data  # rgb_image shape:  (768, 1024, 3)     
        seg_image = self.camera_seg.getImage()  #.data   # seg_image shape:  (80, 400, 3)   
        
        self.update_frame("frame_0", rgb_image)
        self.update_frame("frame_1", seg_image)


        steer, throttle = self.predict_controls(seg_image) # se saco self.monolithic_model como argumento       
        # print(f"Predicted - Steer: {steer:.3f}, Throttle: {throttle:.3f}")
        
        vehicle_location = self.vehicle.get_transform().location
        # calculate distance to target point
        # print(f'vehicle location: ({vehicle_location.x}, {vehicle_location.y})')
        # print(f'target point: ({self.target_point[0]}, {self.target_point[1]})')
        
        # arrived = False # flag to indicate if the vehicle has arrived at the target point
        # print(f"[DEBUG] vehicle: ({vehicle_location.x:.2f}, {vehicle_location.y:.2f}) "
        #         f"target: ({self.target_point[0]:.2f}, {self.target_point[1]:.2f})")
        
        if self.target_point is not None:
            # Aceptar lista o Location
            if isinstance(self.target_point, (list, tuple, np.ndarray)):
                tx, ty = self.target_point[0], self.target_point[1]
            elif hasattr(self.target_point, "x"):
                tx, ty = self.target_point.x, self.target_point.y
            else:
                raise ValueError(f"Unsupported type for target_point: {type(self.target_point)}")

            # Coordenadas actuales
            vx, vy = vehicle_location.x, vehicle_location.y
            # print(f"x: {vx} -> {tx}, y: {vy} -> {ty}")

            # Calcular distancia euclidiana 2D
            distance_to_target = math.hypot(tx - vx, ty - vy)

            if distance_to_target < 1.5:
                self.termination_code = 1
                print(f"======== Arrived at target point ({tx:.1f}, {ty:.1f}) | {distance_to_target:.2f} m away ==========")
                        
        
        self.motors.sendThrottle(throttle)
        self.motors.sendSteer(steer)
        self.motors.sendBrake(0.0)
        
       

