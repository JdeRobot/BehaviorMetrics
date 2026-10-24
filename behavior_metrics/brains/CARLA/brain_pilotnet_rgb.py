import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class PilotNet(nn.Module):
    def __init__(self):
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

    def forward(self, image):
        image = self.conv_layers(image)
        image = image.reshape(image.size(0), -1)
        return self.linear_layers(image)


class Brain:
    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        if not model:
            raise ValueError("Parameters.Model debe indicar el checkpoint PilotNet.")

        self.camera = sensors.get_camera("camera_0")
        self.motors = actuators.get_motor("motors_0")
        config = config or {}
        use_gpu = bool(config.get("GPU", True))
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.throttle = float(config.get("FixedThrottle", 0.35))
        self.steer_min = float(config.get("SteerClipMin", -1.0))
        self.steer_max = float(config.get("SteerClipMax", 1.0))

        model_path = os.path.expanduser(str(model))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No existe el checkpoint: {model_path}")

        self.model = PilotNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"[INFO] PilotNet RGB cargado desde {model_path} en {self.device}")

    def _preprocess(self, image):
        if not isinstance(image, np.ndarray) and hasattr(image, "data"):
            image = image.data
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Se esperaba imagen RGB HxWx3; se recibió {image.shape}")
        if image.shape[0] < 600 or image.shape[1] < 800:
            raise ValueError(
                f"La imagen debe medir al menos 800x600; se recibió {image.shape[1]}x{image.shape[0]}"
            )

        image = image[300:600, 0:800, :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = TF.to_pil_image(image)
        image = TF.resize(
            image,
            size=[66, 200],
            interpolation=InterpolationMode.BILINEAR,
        )
        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return image.unsqueeze(0).to(self.device)

    def execute(self):
        image = self.camera.getImage()
        if image is None:
            return

        input_tensor = self._preprocess(image)
        with torch.inference_mode():
            steer = float(self.model(input_tensor).item())

        steer = float(np.clip(steer, self.steer_min, self.steer_max))
        self.motors.sendThrottle(float(np.clip(self.throttle, 0.0, 1.0)))
        self.motors.sendSteer(steer)
        self.motors.sendBrake(0.0)