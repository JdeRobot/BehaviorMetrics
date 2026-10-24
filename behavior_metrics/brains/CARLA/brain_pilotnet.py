import math
import os
import time
from os import path
from traceback import print_exc

import carla
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from pydash import now

from brains.CARLA.utils.high_level_command import HighLevelCommandLoader
from brains.CARLA.utils.test_utils import (
    calculate_delta_yaw,
    model_control,
    traffic_light_to_int,
)
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from utils.logger import logger


class PilotNetLegacy(nn.Module):
    def __init__(self, num_outputs=1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        return self.linear_layers(x)


PRETRAINED_MODELS = ""


class Brain:
    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        self.motors = actuators.get_motor("motors_0")
        self.camera_rgb = sensors.get_camera("camera_0")
        self.camera_seg = sensors.get_camera("camera_2")
        self.handler = handler

        self.inference_times = []
        self.preprocess_times = []
        self.total_times = []

        self.gpu_inference = config.get("GPU", True) if config is not None else True
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.gpu_inference) else "cpu"
        )
        torch.backends.cudnn.benchmark = True

        self.use_fp16 = bool(config.get("FP16", False)) if config is not None else False
        if self.use_fp16 and self.device.type != "cuda":
            self.use_fp16 = False

        self._tick = 0
        self._vis_every = int(config.get("VisEvery", 1)) if config is not None else 1
        if self._vis_every < 1:
            self._vis_every = 1

        # Input geometry
        if config is not None and "ImageSize" in config:
            img_w, img_h = config["ImageSize"]
            self.out_w = int(img_w)
            self.out_h = int(img_h)
        else:
            self.out_w = 200
            self.out_h = 66

        if config is not None and "CropPixels" in config:
            self.crop_pixels = int(config["CropPixels"])
            self.top_ratio = None
        else:
            self.crop_pixels = 200
            self.top_ratio = (
                float(config.get("TopRatio", 0.40)) if config is not None else None
            )

        if config is not None and "RoadRGB" in config:
            self.road_rgb = np.array(config["RoadRGB"], dtype=np.uint8)
        else:
            self.road_rgb = np.array([128, 64, 128], dtype=np.uint8)

        # Throttle / steer control policy (FIXED, CLIP, or MODEL)
        self.throttle_mode = (
            str(config.get("ThrottleMode", "MODEL")).upper()
            if config is not None
            else "MODEL"
        )
        self.fixed_throttle = (
            float(config.get("FixedThrottle", 0.35)) if config is not None else 0.35
        )
        self.throttle_min = (
            float(config.get("ThrottleMin", 0.25)) if config is not None else 0.25
        )
        self.throttle_max = (
            float(config.get("ThrottleMax", 0.45)) if config is not None else 0.45
        )

        self.use_steer_clip = (
            bool(config.get("UseSteerClip", True)) if config is not None else True
        )
        self.steer_clip_min = (
            float(config.get("SteerClipMin", -0.35)) if config is not None else -0.25
        )
        self.steer_clip_max = (
            float(config.get("SteerClipMax", 0.35)) if config is not None else 0.25
        )

        # Policy execution frequency
        self.sim_hz = float(config.get("SimHz", 20.0)) if config is not None else 20.0
        self.policy_hz = (
            float(config.get("PolicyHz", 5.0)) if config is not None else 5.0
        )
        if self.policy_hz <= 0:
            raise ValueError("PolicyHz must be > 0")

        self.control_period_ticks = max(1, int(round(self.sim_hz / self.policy_hz)))

        self.last_pred_steer = 0.0
        self.last_pred_throttle = 0.0
        self.last_steer_cmd = 0.0
        self.last_throttle_cmd = 0.0
        self.force_inference_next = False

        print(
            f"[INFO] sim_hz={self.sim_hz:.2f} | "
            f"policy_hz={self.policy_hz:.2f} | "
            f"control_period_ticks={self.control_period_ticks}"
        )

        # Debug configuration
        self.debug_enabled = (
            bool(config.get("DebugPreprocess", False)) if config is not None else False
        )
        self.debug_every = (
            int(config.get("DebugEvery", 30)) if config is not None else 30
        )
        self.debug_dir = (
            str(config.get("DebugDir", "/tmp/carla_pilotnet_debug"))
            if config is not None
            else "/tmp/carla_pilotnet_debug"
        )
        self.debug_save_images = (
            bool(config.get("DebugSaveImages", False)) if config is not None else False
        )
        self.debug_print_stats = (
            bool(config.get("DebugPrintStats", False)) if config is not None else False
        )
        self.debug_extractor_mode = (
            str(config.get("RoadExtractorMode", "exact_rgb_from_bgr"))
            if config is not None
            else "exact_rgb_from_bgr"
        )
        self.debug_color_tol = (
            int(config.get("RoadColorTolerance", 5)) if config is not None else 5
        )
        os.makedirs(self.debug_dir, exist_ok=True)

        # CARLA world connection
        self.red_light_counter = 0
        self.running_light = False

        client = carla.Client("localhost", 2000)
        client.set_timeout(100.0)
        self.world = client.get_world()
        self.map = self.world.get_map()

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
                time.sleep(1)

        # Model initialization
        self.use_onnx = False
        self.model_torch = None
        self.ort_session = None
        self.model_type = None

        if model:
            self.device = torch.device(
                "cuda" if (torch.cuda.is_available() and self.gpu_inference) else "cpu"
            )

            monolithic_model_path = PRETRAINED_MODELS + model
            print("Loading model from:", monolithic_model_path)

            ext = os.path.splitext(monolithic_model_path)[1].lower()
            model_name_lower = os.path.basename(monolithic_model_path).lower()

            if ext == ".onnx":
                self.use_onnx = True

                if "pilotnet" in model_name_lower:
                    self.model_type = "pilotnet_legacy_3ch"
                    self.model_input_channels = 3
                else:
                    raise ValueError(f"Unsupported ONNX model: {model_name_lower}")

                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1

                if (
                    self.gpu_inference
                    and "CUDAExecutionProvider" in ort.get_available_providers()
                ):
                    providers = [("CUDAExecutionProvider", {})]
                else:
                    providers = ["CPUExecutionProvider"]

                self.ort_session = ort.InferenceSession(
                    str(monolithic_model_path), sess_options=so, providers=providers
                )
                print(f"[INFO] ONNX providers: {self.ort_session.get_providers()}")

                self._in_name = self.ort_session.get_inputs()[0].name
                self._out_name = self.ort_session.get_outputs()[0].name

                dummy_input = np.random.rand(
                    1, self.model_input_channels, self.out_h, self.out_w
                ).astype(np.float32)

                for _ in range(20):
                    _ = self.ort_session.run(
                        [self._out_name], {self._in_name: dummy_input}
                    )

                print(f"[INFO] ONNX warm-up complete | model_type={self.model_type}")

            elif ext in [".pth", ".pt"]:
                self.use_onnx = False

                if "pilotnet" in model_name_lower:
                    self.model_type = "pilotnet_legacy_3ch"
                    self.model_input_channels = 3
                    net = PilotNetLegacy(num_outputs=2)
                    print("[INFO] Loading PilotNetLegacy (.pth) model")
                else:
                    raise ValueError(
                        f"Unsupported .pth model: {model_name_lower}. "
                        "Expected 3-channel PilotNet legacy model."
                    )

                state = torch.load(monolithic_model_path, map_location=self.device)
                net.load_state_dict(state)
                net.to(self.device)
                net.eval()

                if self.device.type == "cuda":
                    net = net.to(memory_format=torch.channels_last)

                self.model_torch = net

                dummy = torch.randn(
                    1,
                    self.model_input_channels,
                    self.out_h,
                    self.out_w,
                    device=self.device,
                )
                if self.device.type == "cuda":
                    dummy = dummy.contiguous(memory_format=torch.channels_last)

                with torch.inference_mode():
                    for _ in range(10):
                        if self.use_fp16 and self.device.type == "cuda":
                            with torch.autocast(
                                device_type="cuda", dtype=torch.float16
                            ):
                                _ = self.model_torch(dummy)
                        else:
                            _ = self.model_torch(dummy)

                print("[INFO] Torch model warm-up complete")
                print("[INFO] Device:", next(self.model_torch.parameters()).device)
                print("[INFO] FP16:", self.use_fp16)
                print("[INFO] model_type:", self.model_type)

            else:
                raise ValueError(f"Unsupported model extension: {ext}")

        # Route & navigation setup
        if config is not None and "Route" in config:
            route = config["Route"]
            print("route:", route)
        else:
            route = None

        self.hlc_loader = HighLevelCommandLoader(self.vehicle, self.map, route=route)
        self.prev_hlc = 0
        self.prev_yaw = None
        self.delta_yaw = 0

        self.target_point = [193.75, 118.40, 0.50, 0.00, 0.00, 0.00]
        self.termination_code = 0

        self._prev_dist_to_target = None
        self._min_dist_to_target = float("inf")
        self._seen_target_zone = False
        self.target_radius = (
            float(config.get("TargetRadius", 5.5)) if config is not None else 5.5
        )

        # Initial acceleration phase
        target_speed = 0.10  # km/h
        self.start_speed_kmh = (
            float(config.get("StartSpeedKmh", target_speed))
            if config is not None
            else target_speed
        )
        self.start_throttle = (
            float(config.get("StartThrottle", 1.0)) if config is not None else 1.0
        )
        self.start_steer = (
            float(config.get("StartSteer", 0.0)) if config is not None else 0.0
        )
        self.speed_hysteresis_kmh = (
            float(config.get("SpeedHysteresisKmh", 1.0)) if config is not None else 1.0
        )
        self.control_mode = "ACCEL"

    def _should_debug(self):
        return self.debug_enabled and (self._tick % self.debug_every == 0)

    def _save_debug_image(self, name, img):
        if not self.debug_save_images:
            return
        out_path = os.path.join(self.debug_dir, f"{self._tick:06d}_{name}.png")
        cv2.imwrite(out_path, img)

    def _print_array_stats(self, name, arr):
        if not self.debug_print_stats:
            return
        arr_np = np.asarray(arr)
        print(
            f"[DEBUG] {name}: "
            f"shape={arr_np.shape}, dtype={arr_np.dtype}, "
            f"min={arr_np.min()}, max={arr_np.max()}, "
            f"mean={arr_np.mean():.6f}, sum={arr_np.sum():.6f}"
        )

    def _extract_road_mask_debug(self, image_seg: np.ndarray):
        mode = self.debug_extractor_mode
        tol = self.debug_color_tol
        road = self.road_rgb.astype(np.int16)

        if mode == "exact_rgb_from_bgr":
            seg_rgb = cv2.cvtColor(image_seg, cv2.COLOR_BGR2RGB)
            mask = np.all(seg_rgb == self.road_rgb, axis=-1).astype(np.uint8) * 255
            return mask, seg_rgb, mode

        if mode == "exact_rgb_no_convert":
            seg_rgb = image_seg.copy()
            mask = np.all(seg_rgb == self.road_rgb, axis=-1).astype(np.uint8) * 255
            return mask, seg_rgb, mode

        if mode == "tolerance_rgb_no_convert":
            seg_rgb = image_seg.copy().astype(np.int16)
            mask = (
                np.all(np.abs(seg_rgb - road) <= tol, axis=-1).astype(np.uint8) * 255
            )
            return mask, seg_rgb.astype(np.uint8), mode

        if mode == "tolerance_rgb_from_bgr":
            seg_rgb = cv2.cvtColor(image_seg, cv2.COLOR_BGR2RGB).astype(np.int16)
            mask = (
                np.all(np.abs(seg_rgb - road) <= tol, axis=-1).astype(np.uint8) * 255
            )
            return mask, seg_rgb.astype(np.uint8), mode

        raise ValueError(f"Unsupported RoadExtractorMode: {mode}")

    def _crop_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]

        if self.crop_pixels is not None:
            y0 = min(max(self.crop_pixels, 0), h - 1)
        elif self.top_ratio is not None:
            y0 = int(h * self.top_ratio)
            y0 = min(max(y0, 0), h - 1)
        else:
            y0 = 200

        return mask[y0:h, :]

    def _preprocess_segmentation_pilotnet_legacy(
        self, image_seg: np.ndarray
    ) -> np.ndarray:
        """Preprocess semantic segmentation into a normalized 3-channel (1, 3, H, W) tensor."""
        if image_seg is None:
            raise ValueError("Segmented image is None (sensor may not be ready).")

        mask, seg_dbg_rgb, mode_used = self._extract_road_mask_debug(image_seg)
        cropped = self._crop_mask(mask)

        resized = cv2.resize(
            cropped, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST
        )

        x = resized.astype(np.float32) / 255.0
        x = np.stack([x, x, x], axis=0)
        x = np.expand_dims(x, axis=0)

        if self._should_debug():
            print("\n[DEBUG] =========================================")
            print(f"[DEBUG] tick={self._tick}")
            print(f"[DEBUG] extractor_mode={mode_used}")
            print(
                f"[DEBUG] road_rgb={self.road_rgb.tolist()} | tol={self.debug_color_tol}"
            )
            self._print_array_stats("seg_image_raw", image_seg)
            self._print_array_stats("seg_dbg_rgb", seg_dbg_rgb)
            self._print_array_stats("mask_raw", mask)
            self._print_array_stats("mask_cropped", cropped)
            self._print_array_stats("mask_resized", resized)
            self._print_array_stats("model_input", x)

            nonzero = int(np.count_nonzero(mask))
            total = int(mask.size)
            frac = nonzero / max(total, 1)
            print(f"[DEBUG] mask nonzero={nonzero}/{total} ({100.0 * frac:.3f}%)")

            self._save_debug_image("seg_raw", image_seg)
            self._save_debug_image(
                "seg_dbg_rgb", cv2.cvtColor(seg_dbg_rgb, cv2.COLOR_RGB2BGR)
            )
            self._save_debug_image("mask_raw", mask)
            self._save_debug_image("mask_cropped", cropped)
            self._save_debug_image("mask_resized", resized)

        return x.astype(np.float32, copy=False)

    def _preprocess_segmentation(self, image_seg: np.ndarray) -> np.ndarray:
        if self.model_type == "pilotnet_legacy_3ch":
            return self._preprocess_segmentation_pilotnet_legacy(image_seg)

        raise ValueError(f"Unsupported model_type for preprocessing: {self.model_type}")

    def predict_controls(self, image_seg: np.ndarray):
        t0 = time.perf_counter()
        input_np = self._preprocess_segmentation(image_seg)
        t1 = time.perf_counter()

        if self.use_onnx:
            t2 = time.perf_counter()
            pred = self.ort_session.run([self._out_name], {self._in_name: input_np})[0]
            t3 = time.perf_counter()
        else:
            inp_t = torch.from_numpy(input_np).to(self.device, non_blocking=True)

            if self.device.type == "cuda":
                inp_t = inp_t.contiguous(memory_format=torch.channels_last)

            t2 = time.perf_counter()
            with torch.inference_mode():
                if self.use_fp16 and self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model_torch(inp_t)
                else:
                    out = self.model_torch(inp_t)

            pred = out.detach().float().cpu().numpy()
            t3 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t3 - t2) * 1000.0
        tot_ms = (t3 - t0) * 1000.0

        self.preprocess_times.append(pre_ms)
        self.inference_times.append(inf_ms)
        self.total_times.append(tot_ms)

        steer = float(pred[0][0])
        throttle = 0.35

        if self._should_debug():
            print(
                f"[DEBUG] pred raw -> steer={float(steer):.6f}, "
                f"throttle={float(throttle):.6f}"
            )
            print(
                f"[DEBUG] timings -> pre={pre_ms:.2f} ms, "
                f"inf={inf_ms:.2f} ms, total={tot_ms:.2f} ms"
            )

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        return steer, throttle

    def _apply_control_policy(self, pred_steer: float, pred_throttle: float):
        steer_cmd = pred_steer
        if self.use_steer_clip:
            steer_cmd = float(
                np.clip(steer_cmd, self.steer_clip_min, self.steer_clip_max)
            )

        mode = self.throttle_mode
        if mode == "FIXED":
            throttle_cmd = self.fixed_throttle
        elif mode == "CLIP":
            throttle_cmd = float(
                np.clip(pred_throttle, self.throttle_min, self.throttle_max)
            )
        elif mode == "MODEL":
            throttle_cmd = pred_throttle
        else:
            raise ValueError(f"Unsupported ThrottleMode: {mode}")

        return steer_cmd, throttle_cmd

    def update_frame(self, frame_id, data):
        if data is None:
            return

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

    def dist_point_to_segment(self, px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby
        if ab2 == 0:
            return math.hypot(px - ax, py - ay)
        t = (apx * abx + apy * aby) / ab2
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * abx, ay + t * aby
        return math.hypot(px - cx, py - cy)

    def execute(self):
        rgb_image = self.camera_rgb.getImage()
        seg_image = self.camera_seg.getImage()

        self._tick += 1
        if self.handler is not None and (self._tick % self._vis_every == 0):
            self.update_frame("frame_0", rgb_image)
            self.update_frame("frame_1", seg_image)

        vehicle_location = self.vehicle.get_transform().location
        tx, ty = self.target_point[0], self.target_point[1]
        vx, vy = vehicle_location.x, vehicle_location.y

        offset = 3.5
        ax, ay = tx - offset, ty
        bx, by = tx + offset, ty

        distance_to_target = self.dist_point_to_segment(vx, vy, ax, ay, bx, by)

        if distance_to_target <= 1.0:
            self.termination_code = 1
            self.motors.sendThrottle(0.0)
            self.motors.sendSteer(0.0)
            self.motors.sendBrake(1.0)
            time.sleep(0.2)
            print(
                f"======== Arrived at target point ({tx:.1f}, {ty:.1f}) "
                f"| {distance_to_target:.2f} m away =========="
            )
            time.sleep(0.2)
            return

        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

        if self.control_mode == "ACCEL":
            if speed_kmh >= (self.start_speed_kmh + self.speed_hysteresis_kmh):
                self.control_mode = "MODEL"
                self.force_inference_next = True
                self.last_pred_steer = 0.0
                self.last_pred_throttle = 0.0
                self.last_steer_cmd = 0.0
                self.last_throttle_cmd = (
                    self.fixed_throttle if self.throttle_mode == "FIXED" else 0.0
                )
                print(
                    f"[INFO] StartSpeed reached ({speed_kmh:.1f} km/h). Activating model inference."
                )

        if self.control_mode == "ACCEL":
            throttle_cmd = self.start_throttle
            steer_cmd = self.start_steer
        else:
            run_inference = self.force_inference_next or (
                self._tick % self.control_period_ticks == 0
            )

            if run_inference:
                pred_steer, pred_throttle = self.predict_controls(seg_image)
                steer_cmd, throttle_cmd = self._apply_control_policy(
                    pred_steer, pred_throttle
                )

                self.last_pred_steer = pred_steer
                self.last_pred_throttle = pred_throttle
                self.last_steer_cmd = steer_cmd
                self.last_throttle_cmd = throttle_cmd
                self.force_inference_next = False
            else:
                steer_cmd = self.last_steer_cmd
                throttle_cmd = self.last_throttle_cmd

        self.motors.sendThrottle(throttle_cmd)
        self.motors.sendSteer(steer_cmd)
        self.motors.sendBrake(0.0)
