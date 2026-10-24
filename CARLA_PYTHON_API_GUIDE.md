# Complete Guide: BehaviorMetrics with CARLA Python API (No ROS)

This guide documents how to configure, run, customize, and evaluate autonomous driving models (such as **PilotNet**) in **BehaviorMetrics** using the **CARLA 0.9.15 Python API** directly (without ROS 1 or ROS 2 dependencies).

---

## 1. Environment Setup and Critical Variables

### 1.1 Activate the Virtual Environment (`bmetrics_env`)
From the repository root (`/home/daniel/code/BehaviorMetrics`), activate the virtual environment:

```bash
cd ~/code/BehaviorMetrics
source bmetrics_env/bin/activate
```

To recreate the environment on another machine with the exact packages required for the CARLA Python API (without ROS packages):

```bash
python3.10 -m venv bmetrics_env
source bmetrics_env/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements_api_carla.txt
```

### 1.2 Set `CARLA_ROOT` and Disable ROS (`ROS_VERSION`)
BehaviorMetrics automatically selects between **ROS** and the **CARLA Python API** based on the `ROS_VERSION` environment variable:
* If `ROS_VERSION` is set to `1` or `2` (for example, if `~/.bashrc` sources `/opt/ros/humble/setup.bash`), the application will attempt to use ROS nodes and `ros2 bag`, which fails in pure Python API mode.
* Additionally, `environment.py` requires `CARLA_ROOT` pointing to the directory containing `CarlaUE4.sh`.

Before running any experiment, execute:

```bash
# 1. Explicitly disable ROS to force Python API mode
unset ROS_VERSION
export ROS_VERSION=None

# 2. Set the path to your CARLA installation directory
export CARLA_ROOT=$HOME/code/CarlaSim

# 3. Enter the executable source directory
cd ~/code/BehaviorMetrics/behavior_metrics
```

> **Tip:** You can also run `source var_entorno.sh` inside `behavior_metrics/` to export these variables automatically.

---

## 2. Experiment Configuration (`.yml`)

Configuration files are located in [`behavior_metrics/configs/CARLA/`](file:///home/daniel/code/BehaviorMetrics/behavior_metrics/configs/CARLA). Using [`configs/CARLA/eval_pilotnet.yml`](file:///home/daniel/code/BehaviorMetrics/behavior_metrics/configs/CARLA/eval_pilotnet.yml) as a reference, here is how each section works:

### 2.1 Vehicle Sensors and Actuators (`Behaviors.Robot`)
```yaml
Behaviors:
    Robot:
        Name: 'ego_vehicle'
        Model: 'tesla.model3'
        Sensors:
            Cameras:
                Camera_0:
                    Name: 'camera_0'
                    Type: 'rgb'
                    Topic: 'Front'
                    Position: '2.0 0 1.4 0 0 0' # x y z yaw pitch roll relative to the vehicle
                    Width: 800
                    Height: 600
                    FOV: 90
                    SensorTick: 0.05
            Pose3D:
                Pose3D_0:
                    Name: 'pose3d_0'
                    Topic: '/carla/ego_vehicle/odometry'
        Actuators:
            CARLA_Motors:
                Motors_0:
                    Name: 'motors_0'
                    Topic: ''
                    MaxV: 15
                    MaxW: 0.5
```
* **Important:** If your `Brain` crops images expecting an `800x600` resolution, ensure `Camera_0` defines `Width: 800` and `Height: 600`.

### 2.2 Initial Pose (`InitPosition` and `SpawnPoint`)
Defines where the ego vehicle spawns on the map:
```yaml
        # Format: 'X Y Z Roll Pitch Yaw'
        InitPosition: '225.25 -364.11 0.28 0.00 -0.30 0.00'
```

### 2.3 Selecting the Controller (`BrainPath`) and `Parameters`
```yaml
        BrainPath: '/home/daniel/code/BehaviorMetrics/behavior_metrics/brains/CARLA/brain_pilotnet_rgb.py'
        PilotTimeCycle: 50  # Control cycle in milliseconds (50 ms = 20 Hz)
        AsyncMode: False    # False = Synchronous mode controlled step-by-step

        Parameters:
            Model: "/path/to/your/model/pilotnet_balanced_weights.pth"
            ImageCropped: True
            ImageSize: [200, 66]
            TopRatio: 0.40
            ImageNormalized: True
            PredictionsNormalized: True
            GPU: True
            FixedThrottle: 0.35
            UseSteerClip: True
            SteerClipMin: -0.95
            SteerClipMax: 0.95
```
* Everything defined under `Parameters:` is passed as the `config` dictionary to the `Brain.__init__` constructor, and `Parameters.Model` is passed as the `model` argument.

### 2.4 Map and Output Directory (`Simulation` and `Stats`)
```yaml
    Simulation:
        World: configs/CARLA/eval_pilotnet.yml
        Map: 'Town04_Opt'       # CARLA map name (e.g., Town02, Town04_Opt)
        RandomSpawnPoint: False
    Stats:
        Out: './'               # Output directory where <YYYYMMDD-HHMMSS> folders are created
```

---

## 3. Creating or Modifying a `Brain`

CARLA controllers are located in [`behavior_metrics/brains/CARLA/`](file:///home/daniel/code/BehaviorMetrics/behavior_metrics/brains/CARLA). For example, [`brain_pilotnet_rgb.py`](file:///home/daniel/code/BehaviorMetrics/behavior_metrics/brains/CARLA/brain_pilotnet_rgb.py) follows this structure:

```python
class Brain:
    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        # 1. Retrieve camera and motor interfaces defined in the YAML
        self.camera = sensors.get_camera("camera_0")
        self.motors = actuators.get_motor("motors_0")

        # 2. Read custom parameters from config (YAML Parameters section)
        config = config or {}
        self.throttle = float(config.get("FixedThrottle", 0.35))
        self.steer_min = float(config.get("SteerClipMin", -1.0))
        self.steer_max = float(config.get("SteerClipMax", 1.0))

        # 3. Load PyTorch (.pth) model on GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PilotNet().to(self.device)
        checkpoint = torch.load(model, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def execute(self):
        # Called on every pilot cycle (e.g., every 50 ms)
        image = self.camera.getImage()
        if image is None:
            return

        # Preprocess image (crop [300:600, 0:800], resize to 66x200, normalize)
        input_tensor = self._preprocess(image)

        # Model inference
        with torch.inference_mode():
            steer = float(self.model(input_tensor).item())

        # Send control commands to CARLA via CarlaApiMotors
        steer = float(np.clip(steer, self.steer_min, self.steer_max))
        self.motors.sendThrottle(self.throttle)
        self.motors.sendSteer(steer)
        self.motors.sendBrake(0.0)
```

---

## 4. Using the Graphical User Interface (GUI) Step-by-Step

### 4.1 Launch the Application
From `behavior_metrics/` with the environment variables set, run:

```bash
python3 driver_carla.py -c configs/CARLA/eval_pilotnet.yml -g
```

This automatically:
1. Starts the `CarlaUE4.sh` server in the background (if not already running).
2. Opens the Pygame window (`carla_world_generator.py`) that loads the configured map (`Town04_Opt`), weather, and `ego_vehicle`.
3. Opens the main **Behavior Metrics** GUI window.

### 4.2 Running and Recording an Experiment
1. **Configure Camera Views (Right Panel):**
   * In any of the camera frames on the right, select the sensor dropdown (e.g., `camera_0`), check `rgbimage`, and click **Confirm** to display the live camera feed.
2. **Start Recording Metrics (`Stats` Groupbox, Top-Left):**
   * Verify that `Save path:` is set to `./` (or choose an output folder using `...`).
   * Click the **Play** button inside the **Stats** section.
   * The **"Recording..."** indicator will appear, immediately creating a `<YYYYMMDD-HHMMSS>/` directory containing `<timestamp>.csv` and `<timestamp>.meta.json`.
3. **Start the Vehicle (`Simulation` Groupbox, Bottom-Left):**
   * Click the **Play** button inside the **Simulation** section.
   * The `Brain` will start running inference and driving the vehicle.
4. **Pause or Reset the Simulation (`Simulation`):**
   * **Pause Button (in `Simulation`):** Cuts throttle, engages the handbrake, and zeroes out the vehicle's linear and angular velocity immediately. Clicking it again releases the brake and resumes driving.
   * **Reset Button (circular arrows):** Stops the vehicle and teleports it back to its initial spawn pose (`InitPosition`).
5. **Stop Recording and Generate Plots (`Stats`):**
   * Once the test finishes, pause the vehicle (**Pause** in `Simulation`).
   * Click the **Pause** button inside the **Stats** section.
   * This closes the `.csv`, computes all evaluation metrics, generates `<timestamp>.json` and the `.png` trajectory plots, and displays the summary metrics window.
6. **Closing BehaviorMetrics Cleanly:**
   * Click the **`X`** button at the top-right corner of the main **Behavior Metrics** window.
   * *(Safety feature: If you close the window with `X` while `Stats` recording is still active, the application automatically stops the recording and saves the `.json` and `.png` files before shutting down CARLA).*

---

## 5. Understanding the Output Files

Each recorded run creates a timestamped directory `YYYYMMDD-HHMMSS/` containing:

### 5.1 `<timestamp>.csv` (Tick-by-Tick Telemetry)
Contains one row per simulator tick with 25 columns:
* **Time:** `time_stamp` (simulated seconds), `frame`
* **3D Pose:** `x`, `y`, `z` (meters), `yaw`, `pitch`, `roll` (degrees)
* **Velocity:** `vx`, `vy`, `vz`, `speed_mps` (m/s), `speed_kmh` (km/h)
* **Control Commands:** `throttle` (0 to 1), `steer` (-1 to 1), `brake` (0 to 1), `reverse`, `gear`
* **Per-Tick Events:**
  * `collision_impulse`: Impact impulse magnitude if a collision occurred on that tick (`0.0` otherwise).
  * `collision_with`: Actor type involved in the collision (e.g., `static.wall`, `static.fence`).
  * `lane_invasion`: Crossed lane marking type on that tick (e.g., `Broken`, `Solid`).
* **Weather:** `weather_cloud`, `weather_rain`, `weather_wetness`, `weather_fog`

### 5.2 `<timestamp>.meta.json` (Experiment Metadata)
Stores the exact configuration snapshot of the run: `.yml` file, `.pth` model path, sensor setup, CARLA map, and weather parameters.

### 5.3 `<timestamp>.json` (Final Evaluation Summary)
Contains the quantitative performance metrics:
* `experiment_total_simulated_time` / `experiment_total_real_time`: Duration in simulated vs. wall-clock seconds.
* `completed_distance`: Total Euclidean distance traveled by the vehicle (meters).
* `effective_completed_distance`: Effective distance covered along valid lane waypoints (meters).
* `average_speed`, `max_speed`, `min_speed`: Speeds in **km/h**.
* `position_deviation_mean`: Mean lateral error (meters) relative to the nearest lane center.
* `position_deviation_total_err` / `position_deviation_mean_per_km`: Total and per-kilometer lateral deviation.
* `collisions` / `collisions_per_km`: Total collision events and rate per kilometer.
* `lane_invasions` / `lane_invasions_per_km`: Total lane invasion events and rate per kilometer.
* `suddenness_distance_*`: Control smoothness metrics (`steer`, `throttle`, `brake`, and combined). Lower values indicate smoother, less jerky driving.

### 5.4 Visual Trajectory Maps (`.png`)
1. **`<timestamp>.png` (Main Trajectory, Deviation, and Speed Map):**
   * **Soft gray squares (`Map waypoints`):** Road lane centerlines of the CARLA map.
   * **Green dots (`Map waypoints for position deviation`):** Nearest lane center waypoints used as reference for lateral error.
   * **Heatmap line (`Experiment waypoints`):** Actual vehicle trajectory colored by speed in km/h (**Light yellow** = slow, **Orange/Red** = medium speed, **Dark red/Black** = high speed).
   * **Large Green Circle:** `Experiment starting point`.
   * **Large Red Circle:** `Experiment finish point`.
   * **Bottom text:** Clean summary of the experiment metrics.
2. **`<timestamp>_collisions.png`:**
   * Generated when `collisions > 0`. Marks the exact `(X, Y)` locations of collisions with **large yellow circles**.
3. **`<timestamp>_lane_invasion.png`:**
   * Generated when `lane_invasions > 0`. Marks the exact locations where the vehicle crossed lane markings with **large yellow circles**.

---

## 6. Useful Commands and Troubleshooting

### Regenerate `.json` and `.png` Plots Manually from an Existing `.csv`
If you have a run folder (e.g., `20261024-124555`) and want to recompute its `.json` summary and `.png` plots:

```bash
cd ~/code/BehaviorMetrics/behavior_metrics
python3 -c '
import json
from utils import metrics_carla
from utils.metrics_carla import build_json_summary, save_json_summary

folder = "/home/daniel/code/BehaviorMetrics/20261024-124555"
ts = "20261024-124555"
with open(f"{folder}/{ts}.meta.json") as f:
    meta = json.load(f)
res = metrics_carla.get_metrics_python_api(meta, f"{folder}/{ts}.csv", None, f"{folder}/{ts}", None)
summary = build_json_summary(res, extras={"timestamp": ts})
save_json_summary(f"{folder}/{ts}.json", summary)
'
```

### Force-Close Background CARLA Processes
```bash
pkill -9 -f CarlaUE4
pkill -9 -f carla_world_generator.py
```
