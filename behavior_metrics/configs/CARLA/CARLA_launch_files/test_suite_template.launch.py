#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declarar argumentos equivalentes a los <arg> del XML
    host_arg = DeclareLaunchArgument(
        'host',
        default_value='localhost',
        description='IP o nombre de host de CARLA'
    )
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='2000',
        description='Puerto de CARLA'
    )
    timeout_arg = DeclareLaunchArgument(
        'timeout',
        default_value='10',
        description='Timeout (segundos) de conexión'
    )
    town_arg = DeclareLaunchArgument(
        'town',
        default_value='Town01',  # o puedes dejarlo vacío si quieres
        description='Mapa/town que va a cargar CARLA'
    )
    spawn_point_arg = DeclareLaunchArgument(
        'spawn_point',
        default_value='0',
        description='Spawn point para ego_vehicle'
    )
    passive_arg = DeclareLaunchArgument(
        'passive',
        default_value='',
        description='Activar modo pasivo (true/false). Por defecto vacío'
    )
    synchronous_mode_arg = DeclareLaunchArgument(
        'synchronous_mode',
        default_value='True',
        description='Habilitar modo síncrono en CARLA'
    )
    synchronous_wait_arg = DeclareLaunchArgument(
        'synchronous_mode_wait_for_vehicle_control_command',
        default_value='False',
        description='Esperar comandos de control de vehículo en modo síncrono'
    )
    fixed_delta_arg = DeclareLaunchArgument(
        'fixed_delta_seconds',
        default_value='0.05',
        description='Delta de tiempo fijo en modo síncrono'
    )
    objects_def_file_arg = DeclareLaunchArgument(
        'objects_definition_file',
        default_value='$(env OBJECT_PATH)/main_car_custom_camera.json',
        description='JSON con la definición de objetos (vehículos, sensores, etc.)'
    )
    ego_vehicle_role_arg = DeclareLaunchArgument(
        'ego_vehicle_role_name',
        default_value='ego_vehicle',
        description='Rol del vehículo ego'
    )

    # Nodo para el bridge de ROS con CARLA
    carla_ros_bridge_node = Node(
        package='carla_ros_bridge',
        executable='bridge.py',
        name='carla_ros_bridge',
        output='screen',
        parameters=[{
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
            'timeout': LaunchConfiguration('timeout'),
            'passive': LaunchConfiguration('passive'),
            'synchronous_mode': LaunchConfiguration('synchronous_mode'),
            'synchronous_mode_wait_for_vehicle_control_command': LaunchConfiguration('synchronous_mode_wait_for_vehicle_control_command'),
            'fixed_delta_seconds': LaunchConfiguration('fixed_delta_seconds'),
            'register_all_sensors': True,
            'town': LaunchConfiguration('town'),
            'ego_vehicle_role_name': LaunchConfiguration('ego_vehicle_role_name')
        }]
    )

    # Nodo para spawnear los objetos (vehículos, sensores, etc.)
    spawn_objects_node = Node(
        package='carla_spawn_objects',
        executable='carla_spawn_objects.py',
        name='carla_spawn_objects',
        output='screen',
        parameters=[{
            'objects_definition_file': LaunchConfiguration('objects_definition_file'),
            # Observa cómo el "spawn_point_$(arg ego_vehicle_role_name)" del XML
            # se “mapea” ahora a un diccionario en Python. Normalmente, si
            # carla_spawn_objects.py procesa un parámetro llamado
            # "spawn_point_ego_vehicle", lo indicamos así:
            'spawn_point_ego_vehicle': LaunchConfiguration('spawn_point'),
            'spawn_sensors_only': False
        }]
    )

    # Nodo para el control manual (carla_manual_control.launch se traduce a un Node)
    # Si el package carla_manual_control trae un launch.py, puedes hacer un IncludeLaunchDescription.
    # Si es un ejecutable Python, lo llamas con Node. Aquí asumimos que es un script:
    manual_control_node = Node(
        package='carla_manual_control',
        executable='carla_manual_control',
        name='carla_manual_control',
        output='screen',
        # El script seguramente necesitará un argumento 'role_name' (equivalente a la arg del XML).
        # Muchos scripts en carla_manual_control usan un param o un arg 'role_name' o 'rolename':
        parameters=[{
            'role_name': LaunchConfiguration('ego_vehicle_role_name')
        }]
    )

    # Juntamos todo en el LaunchDescription
    return LaunchDescription([
        # Primero declaramos todos los argumentos
        host_arg,
        port_arg,
        timeout_arg,
        town_arg,
        spawn_point_arg,
        passive_arg,
        synchronous_mode_arg,
        synchronous_wait_arg,
        fixed_delta_arg,
        objects_def_file_arg,
        ego_vehicle_role_arg,
        # Luego añadimos los nodos
        carla_ros_bridge_node,
        spawn_objects_node,
        manual_control_node
    ])
