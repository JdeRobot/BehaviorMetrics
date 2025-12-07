import os

import launch
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    host_arg = DeclareLaunchArgument(
        "host", default_value="localhost", description="Host for CARLA server"
    )
    port_arg = DeclareLaunchArgument(
        "port", default_value="2000", description="Port for CARLA server"
    )
    timeout_arg = DeclareLaunchArgument(
        "timeout", default_value="10", description="Timeout for CARLA connection"
    )
    role_name_arg = DeclareLaunchArgument(
        "role_name",
        default_value="ego_vehicle",
        description="Role name for the vehicle",
    )
    vehicle_filter_arg = DeclareLaunchArgument(
        "vehicle_filter",
        default_value="vehicle.*",
        description="Filter for vehicle spawning",
    )
    spawn_point_arg = DeclareLaunchArgument(
        "spawn_point",
        default_value="100.0, 2.0, 1.37, 0.0, 0.0, 180.0",
        description="Spawn point (x,y,z,roll,pitch,yaw) for the ego vehicle",
    )
    # DEbugging changes town to Town01
    town_arg = DeclareLaunchArgument(
        "town", default_value="Town02", description="CARLA town to load"
    )
    passive_arg = DeclareLaunchArgument(
        "passive", default_value="", description="Enable/disable passive mode"
    )
    sync_wait_arg = DeclareLaunchArgument(
        "synchronous_mode_wait_for_vehicle_control_command",
        default_value="False",
        description="Wait for vehicle control command in synchronous mode",
    )
    fixed_delta_arg = DeclareLaunchArgument(
        "fixed_delta_seconds",
        default_value="0.05",
        description="Fixed delta time for simulation steps",
    )

    # Incluir carla_ros_bridge.launch.py (adaptado a ROS2)
    carla_ros_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("carla_ros_bridge"),
                "carla_ros_bridge.launch.py",
            )
        ),
        launch_arguments={
            "host": LaunchConfiguration("host"),
            "port": LaunchConfiguration("port"),
            "town": LaunchConfiguration("town"),  # se podría concatenar con _Opt si se desea
            "timeout": LaunchConfiguration("timeout"),
            "passive": LaunchConfiguration("passive"),
            "synchronous_mode_wait_for_vehicle_control_command": LaunchConfiguration(
                "synchronous_mode_wait_for_vehicle_control_command"
            ),
            "fixed_delta_seconds": LaunchConfiguration("fixed_delta_seconds"),
        }.items(),
    )

    # Incluir carla_example_ego_vehicle.launch.py (ROS2)
    spawn_objects_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("carla_spawn_objects"),
                "carla_example_ego_vehicle.launch.py",
            )
        ),
        launch_arguments={
            "host": LaunchConfiguration("host"),
            "port": LaunchConfiguration("port"),
            "timeout": LaunchConfiguration("timeout"),
            "role_name": LaunchConfiguration("role_name"),
            "vehicle_filter": LaunchConfiguration("vehicle_filter"),
            "spawn_point": LaunchConfiguration("spawn_point"),
        }.items(),
    )

    # Incluir carla_manual_control.launch.py (ROS2)
    manual_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("carla_manual_control"),
                "carla_manual_control.launch.py",
            )
        ),
        launch_arguments={
            "role_name": LaunchConfiguration("role_name"),
        }.items(),
    )

    # Construir la LaunchDescription final
    return launch.LaunchDescription(
        [
            host_arg,
            port_arg,
            timeout_arg,
            role_name_arg,
            vehicle_filter_arg,
            spawn_point_arg,
            town_arg,
            passive_arg,
            sync_wait_arg,
            fixed_delta_arg,
            carla_ros_bridge_launch,
            spawn_objects_launch,
            manual_control_launch,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
