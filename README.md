# Rescue-Go2

Unitree Go2 rescue robot software suite - RoboCup 2026 UruBots Team. Includes 3D SLAM mapping, 2D GeoTIFF generation, and HAZMAT/Object detection via ROS2.

Developed by UruBots - UTEC, Uruguay

---

## Project Structure

```
Rescue-Go2/
├── vision/                         # Offline IA models and standalone testing
│   ├── best_hazmat.pt              # Custom trained HAZMAT YOLOv8 model
│   └── yolov8n.pt                  # Base COCO model for real objects
│
├── src/
│   ├── urubots_vision/             # RoboCup 2026 Core Package
│   │   ├── robocup_mapper.py       # 3D PointCloud Generator (.ply)
│   │   ├── geotiff_mapper.py       # 2D GeoTIFF Map Generator (.tiff)
│   │   └── vision_detector.py      # HAZMAT/AprilTag/Object Detector (.csv)
│   │
│   └── go2_ros2_sdk/               # Unitree Go2 ROS2 SDK (submodule)
│
├── BITACORA_URUBOTS.txt            # Development changelog
└── README.md
```

---

## Features

### Vision - HAZMAT & Real Object Detection
- YOLOv8 custom model for HAZMAT placards detection.
- YOLO COCO native detection for real objects (Backpacks, Persons, Suitcases).
- AprilTag (tag36h11 family) 3D coordinate projection.
- Automatic CSV generation formatted specifically for RoboCup 2026 judges.

### Mapping - 3D PLY & 2D GeoTIFF
- Generates 3D PointCloud (.ply) colored with camera feed.
- Dynamic origin translation and 90-degree rotations to comply with RoboCup rules.
- Generates 2D SLAM maps as .tiff images with embedded team info and custom padding.

### Robot Integration
- Fully integrated with the Unitree Go2 via WebRTC (go2_ros2_sdk).
- Retrieves odometry, LiDAR point clouds, and camera feeds directly from the robot.

---

## Quick Start

### Prerequisites
- Ubuntu 22.04+
- ROS2 Humble/Jazzy
- Python 3.10+
- Unitree Go2 robot

### 1. Build ROS2 packages

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. Connect to the Robot
Ensure your computer is connected to the Unitree Go2 WiFi network. The robot's default IP is 192.168.12.1.

### 3. Run the RoboCup Mission
Launch the main suite. Replace "Mision_1" with the current mission name.

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch urubots_vision robocup.launch.py mission:="Mision_1"
```

### 4. Finish and Export
When the mission is complete, press Ctrl+C ONCE. Wait 5-10 seconds for the system to downsample and export the 3D map. You will see a success message when the PLY, TIFF, and CSV files are safely saved in your workspace root.

---

## Testing Models Offline
If you want to test the HAZMAT model locally using your laptop webcam without connecting to the robot:

```bash
python3 ~/ros2_ws/test_hazmat_gui.py
```

---

## Git Submodules

This repo uses git submodules for third-party ROS2 packages:
- `src/go2_ros2_sdk` - Unitree Go2 SDK

---

## License
MIT License — see LICENSE for details.
