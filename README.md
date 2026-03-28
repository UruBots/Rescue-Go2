# 🤖 Rescue-Go2

**Unitree Go2 rescue robot** — SLAM with Velodyne VLP-16, HAZMAT/QR detection with YOLOv8, and robot control via ROS2.

> Developed by [UruBots](https://github.com/UruBots) — UTEC, Uruguay

---

## 📁 Project Structure

```
Rescue-Go2/
├── vision/                         # Computer vision modules
│   ├── combined_viewer.py          # Combined QR + HAZMAT detector (UDP stream)
│   ├── hazmat_udp_viewer.py        # HAZMAT-only detector (UDP stream)
│   ├── qr_udp_viewer.py            # QR-only detector (UDP stream)
│   ├── qr_camera_viewer.py         # QR detector (local camera)
│   ├── detector_imagen.py          # HAZMAT detector GUI (static images)
│   └── detector_tiempo_real.py     # HAZMAT real-time detector (webcam)
│
├── ros2_ws/
│   └── src/
│       ├── velodyne_slam_launch.py  # Full SLAM launch file
│       ├── velodyne_slam_rviz.rviz  # RViz config for SLAM
│       ├── velodyne/                # Velodyne VLP-16 driver (submodule)
│       └── rf2o_laser_odometry/     # Laser-based odometry (submodule)
│
└── README.md
```

---

## 🧩 Features

### 👁️ Vision — HAZMAT & QR Detection

- **YOLOv8n** model trained on 6,131 images covering **49 HAZMAT placard classes**
- Real-time detection via UDP stream from onboard Raspberry Pi camera
- Combined viewer detects both QR codes and HAZMAT placards simultaneously
- YOLO runs in a **separate process** to keep the display responsive

### 🗺️ SLAM — Velodyne VLP-16

Complete SLAM pipeline for mapping with a handheld or robot-mounted LiDAR:

```
Velodyne VLP-16 → velodyne_driver → PointCloud2 → LaserScan
    → rf2o_laser_odometry (scan matching) → slam_toolbox → Map + RViz
```

- **rf2o**: Generates odometry from laser scan matching (no wheels/IMU needed)
- **SLAM Toolbox**: Online async mode, 0.05m resolution, loop closure enabled
- Single launch file starts the complete pipeline

---

## 🚀 Quick Start

### Prerequisites

- Ubuntu 22.04+
- ROS2 Humble/Jazzy
- Python 3.10+

### 1. Clone (with submodules)

```bash
git clone --recurse-submodules https://github.com/UruBots/Rescue-Go2.git
cd Rescue-Go2
```

> If you already cloned without `--recurse-submodules`:
> ```bash
> git submodule update --init --recursive
> ```

### 2. Build ROS2 packages

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. Download YOLO model

Download the trained HAZMAT model (`best_hazmat.pt`) and place it in the `vision/` directory:

```bash
# TODO: Add download link once model is hosted
# Place best_hazmat.pt in vision/
```

### 4. Install Python dependencies (for vision)

```bash
pip install ultralytics opencv-python pyzbar numpy
```

---

## 📖 Usage

### Run SLAM with Velodyne

Connect the Velodyne VLP-16 via Ethernet, then:

```bash
cd ros2_ws
source install/setup.bash
ros2 launch src/velodyne_slam_launch.py
```

This starts: Velodyne driver → PointCloud → LaserScan → Odometry → SLAM → RViz

### Run Vision (Combined QR + HAZMAT)

Start the UDP stream from the Raspberry Pi, then:

```bash
cd vision
python3 combined_viewer.py udp://0.0.0.0:1234
```

### Other Vision Scripts

```bash
# HAZMAT only (UDP stream)
python3 hazmat_udp_viewer.py

# QR only (UDP stream)
python3 qr_udp_viewer.py

# QR (local camera)
python3 qr_camera_viewer.py

# HAZMAT GUI (static image analysis)
python3 detector_imagen.py

# HAZMAT real-time (webcam)
python3 detector_tiempo_real.py
```

---

## 🔧 Hardware

| Component | Model |
|-----------|-------|
| Robot | Unitree Go2 |
| LiDAR | Velodyne VLP-16 |
| Camera | Raspberry Pi Camera (UDP stream) |
| GPU (training) | NVIDIA GTX 1650 |

---

## 📦 Git Submodules

This repo uses git submodules for third-party ROS2 packages:

| Submodule | Source | Branch |
|-----------|--------|--------|
| `ros2_ws/src/velodyne` | [ros-drivers/velodyne](https://github.com/ros-drivers/velodyne) | `ros2` |
| `ros2_ws/src/rf2o_laser_odometry` | [MAPIRlab/rf2o_laser_odometry](https://github.com/MAPIRlab/rf2o_laser_odometry) | `ros2` |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
