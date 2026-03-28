"""
Launch file for manual SLAM with Velodyne VLP-16.
Launches: Velodyne driver + PointCloud + LaserScan + rf2o odometry + slam_toolbox + RViz2
Usage: ros2 launch velodyne_slam_launch.py
"""

import os
import yaml

import ament_index_python.packages
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():

    # =============================================
    # 1. VELODYNE DRIVER
    # =============================================
    driver_share_dir = ament_index_python.packages.get_package_share_directory('velodyne_driver')
    driver_params_file = os.path.join(driver_share_dir, 'config', 'VLP16-velodyne_driver_node-params.yaml')
    velodyne_driver_node = Node(
        package='velodyne_driver',
        executable='velodyne_driver_node',
        name='velodyne_driver_node',
        output='both',
        parameters=[driver_params_file]
    )

    # =============================================
    # 2. VELODYNE POINTCLOUD (raw packets -> PointCloud2)
    # =============================================
    convert_share_dir = ament_index_python.packages.get_package_share_directory('velodyne_pointcloud')
    convert_params_file = os.path.join(convert_share_dir, 'config', 'VLP16-velodyne_transform_node-params.yaml')
    with open(convert_params_file, 'r') as f:
        convert_params = yaml.safe_load(f)['velodyne_transform_node']['ros__parameters']
    convert_params['calibration'] = os.path.join(convert_share_dir, 'params', 'VLP16db.yaml')
    velodyne_transform_node = Node(
        package='velodyne_pointcloud',
        executable='velodyne_transform_node',
        name='velodyne_transform_node',
        output='both',
        parameters=[convert_params]
    )

    # =============================================
    # 3. VELODYNE LASERSCAN (PointCloud2 -> LaserScan)
    # =============================================
    laserscan_share_dir = ament_index_python.packages.get_package_share_directory('velodyne_laserscan')
    laserscan_params_file = os.path.join(laserscan_share_dir, 'config', 'default-velodyne_laserscan_node-params.yaml')
    velodyne_laserscan_node = Node(
        package='velodyne_laserscan',
        executable='velodyne_laserscan_node',
        name='velodyne_laserscan_node',
        output='both',
        parameters=[laserscan_params_file]
    )

    # =============================================
    # 4. STATIC TF: base_link -> velodyne
    # (identity transform since we're carrying it by hand)
    # =============================================
    static_tf_base_to_velodyne = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_velodyne_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'velodyne']
    )

    # =============================================
    # 5. RF2O LASER ODOMETRY
    # Generates odometry from laser scan matching (no wheels needed)
    # =============================================
    rf2o_odometry_node = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry_node',
        output='screen',
        parameters=[{
            'laser_scan_topic': '/scan',
            'odom_topic': '/odom',
            'publish_tf': True,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'init_pose_from_topic': '',
            'freq': 10.0,
        }],
    )

    # =============================================
    # 6. SLAM TOOLBOX (Online Async mode)
    # =============================================
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            # General
            'solver_plugin': 'solver_plugins::CeresSolver',
            'ceres_linear_solver': 'SPARSE_NORMAL_CHOLESKY',
            'ceres_preconditioner': 'SCHUR_JACOBI',
            'ceres_trust_strategy': 'LEVENBERG_MARQUARDT',
            'ceres_dogleg_type': 'TRADITIONAL_DOGLEG',
            'ceres_loss_function': 'None',

            # Input
            'odom_frame': 'odom',
            'map_frame': 'map',
            'base_frame': 'base_link',
            'scan_topic': '/scan',
            'use_scan_matching': True,
            'use_scan_barycenter': True,

            # Throttle
            'mode': 'mapping',
            'map_update_interval': 3.0,
            'resolution': 0.05,
            'max_laser_range': 50.0,
            'minimum_travel_distance': 0.3,
            'minimum_travel_heading': 0.3,

            # Tuning
            'transform_publish_period': 0.05,
            'tf_buffer_duration': 30.0,
            'stack_size_to_use': 40000000,

            # Loop closure
            'loop_search_maximum_distance': 3.0,
            'do_loop_closing': True,

            # Debug
            'debug_logging': False,
            'throttle_scans': 1,

            'transform_timeout': 0.2,
            'scan_buffer_size': 10,
            'minimum_time_interval': 0.5,
        }],
    )

    # =============================================
    # 7. RVIZ2 con configuración para SLAM
    # =============================================
    rviz_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'velodyne_slam_rviz.rviz'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path] if os.path.exists(rviz_config_path) else [],
        output='screen'
    )

    return LaunchDescription([
        # Velodyne nodes
        velodyne_driver_node,
        velodyne_transform_node,
        velodyne_laserscan_node,
        # TF
        static_tf_base_to_velodyne,
        # Odometry from laser
        rf2o_odometry_node,
        # SLAM
        slam_toolbox_node,
        # Visualization
        rviz_node,
        # Shutdown on driver exit
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=velodyne_driver_node,
                on_exit=[EmitEvent(event=Shutdown())],
            )
        ),
    ])
