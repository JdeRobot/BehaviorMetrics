---
permalink: /carla/tasks_and_metrics
title: "Supported tasks and evaluation metrics"


sidebar:
  nav: "docs"
---

In this page, we describe the supported tasks and the evaluation metrics that are specific for each of them.

## Common evaluation metrics

## Follow lane

### Lane keeping and steering stability (Python API mode)

In Python API mode (`ROS_VERSION=None`), the metrics CSV includes, for every simulation tick, the lane under the centre of the ego vehicle:

| Column | Meaning |
| --- | --- |
| `lane_road_id`, `lane_id` | Road and lane of the CARLA waypoint under the vehicle (empty if it is not on a driving lane) |
| `lane_heading_diff` | Vehicle yaw minus lane yaw, in degrees between -180 and 180. Negative means the vehicle points to the left of the lane direction |
| `on_driving_lane` | 1 if the vehicle centre is on a lane of type `Driving` |
| `in_right_lane` | 1 if it is also driving in the lane direction (\|`lane_heading_diff`\| < 90°, right-hand traffic). With `BM_RIGHT_LANE_MODE=rightmost`, it must also be the rightmost driving lane |

From these columns and the applied steering, the metrics JSON adds:

* **`right_lane_percentage`**: percentage of ticks with `in_right_lane` = 1.
* **`off_driving_lane_percentage`**: percentage of ticks with the vehicle centre outside any driving lane.
* **`steering_zero_crossings_per_second`**: number of sign changes of the applied steering divided by the simulated time. Values with an absolute steering below a deadband (`BM_STEER_DEADBAND`, 0.02 by default) are ignored, so small noise around zero does not count. The total count and the deadband used are stored as `steering_zero_crossings` and `steering_zero_crossings_deadband`.

Reading the two main metrics together:

| Zero crossings per second | Right lane percentage | Diagnosis |
| --- | --- | --- |
| Low | High | Stable and correct |
| Low | Low | Stable but biased (e.g. drifts to one side) |
| High | High | Nervous corrections |
| High | Low | Unstable |

Both metrics are computed over all recorded ticks. If the recording continues after a collision with the vehicle stopped, those ticks are included, so stop the recording at the end of the driving segment you want to evaluate.

Both environment variables are optional and are read when Behavior Metrics computes the metrics:

```bash
export BM_STEER_DEADBAND=0.02          # deadband for steering zero crossings
export BM_RIGHT_LANE_MODE=direction    # or: rightmost
```


## Driving in traffic

## Navigation

