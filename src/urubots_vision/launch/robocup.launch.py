from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    mission_arg = DeclareLaunchArgument(
        'mission', default_value='Mision_1', description='Nombre de la Mision'
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

    return LaunchDescription([
        mission_arg,
        mapper_node,
        vision_node,
        geotiff_node
    ])
