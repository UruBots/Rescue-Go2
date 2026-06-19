from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    mission_arg = DeclareLaunchArgument(
        'mission', default_value='Mision_1', description='Nombre de la Mision'
    )
    
    # Expose mapping.launch.py arguments
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Launch RViz2'
    )
    foxglove_arg = DeclareLaunchArgument(
        'foxglove', default_value='true', description='Launch Foxglove Bridge'
    )
    joystick_arg = DeclareLaunchArgument(
        'joystick', default_value='true', description='Launch joystick control'
    )

    # Include SDK mapping launch (connects to WebRTC robot, lidar processing, SLAM, etc.)
    go2_mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('go2_robot_sdk'),
                'launch',
                'mapping.launch.py'
            )
        ]),
        launch_arguments={
            'rviz': LaunchConfiguration('rviz'),
            'foxglove': LaunchConfiguration('foxglove'),
            'joystick': LaunchConfiguration('joystick'),
        }.items()
    )

    mapper_node = Node(
        package='urubots_vision',
        executable='robocup_mapper',
        parameters=[{'mission_name': LaunchConfiguration('mission')}],
        output='screen'
    )

    vision_node = Node(
        package='urubots_vision',
        executable='vision_detector',
        parameters=[{'mission_name': LaunchConfiguration('mission')}],
        output='screen'
    )

    geotiff_node = Node(
        package='urubots_vision',
        executable='geotiff_mapper',
        parameters=[{'mission_name': LaunchConfiguration('mission')}],
        output='screen'
    )

    tts_node = Node(
        package='speech_processor',
        executable='tts_node',
        name='tts_node',
        output='screen',
        parameters=[{
            'piper_bin':    os.path.expanduser('~/piper_voices/piper/piper'),
            'piper_model':  os.path.expanduser('~/piper_voices/es_ES-sharvard-medium.onnx'),
            'local_playback': True,   # True = habla por los parlantes del PC
        }]
    )

    return LaunchDescription([
        mission_arg,
        rviz_arg,
        foxglove_arg,
        joystick_arg,
        go2_mapping_launch,
        mapper_node,
        vision_node,
        geotiff_node,
        tts_node,
    ])
