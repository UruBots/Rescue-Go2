from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
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
    slam_arg = DeclareLaunchArgument(
        'slam', default_value='true', description='Launch SLAM Toolbox mapping'
    )

    # Expose urubots_vision optimization arguments
    audio_arg = DeclareLaunchArgument(
        'audio', default_value='true', description='Launch TTS audio alerts node'
    )
    vision_arg = DeclareLaunchArgument(
        'vision', default_value='true', description='Launch HAZMAT & AprilTags vision detector'
    )
    robocup_mapper_arg = DeclareLaunchArgument(
        'robocup_mapper', default_value='true', description='Launch RoboCup 3D PLY mapper'
    )
    geotiff_arg = DeclareLaunchArgument(
        'geotiff', default_value='true', description='Launch GeoTIFF 2D mapper'
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
            'slam': LaunchConfiguration('slam'),
        }.items()
    )

    mapper_node = Node(
        package='urubots_vision',
        executable='robocup_mapper',
        condition=IfCondition(LaunchConfiguration('robocup_mapper')),
        parameters=[{'mission_name': LaunchConfiguration('mission')}],
        output='screen'
    )

    vision_node = Node(
        package='urubots_vision',
        executable='vision_detector',
        condition=IfCondition(LaunchConfiguration('vision')),
        parameters=[{'mission_name': LaunchConfiguration('mission')}],
        output='screen'
    )

    geotiff_node = Node(
        package='urubots_vision',
        executable='geotiff_mapper',
        condition=IfCondition(LaunchConfiguration('geotiff')),
        parameters=[{'mission_name': LaunchConfiguration('mission')}],
        output='screen'
    )

    tts_node = Node(
        package='speech_processor',
        executable='tts_node',
        name='tts_node',
        condition=IfCondition(LaunchConfiguration('audio')),
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
        slam_arg,
        audio_arg,
        vision_arg,
        robocup_mapper_arg,
        geotiff_arg,
        go2_mapping_launch,
        mapper_node,
        vision_node,
        geotiff_node,
        tts_node,
    ])
