#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge
import datetime
import os

def quat_to_mat(q, t):
    """Convierte quaternion y traslación a matriz 4x4"""
    x, y, z, w = q
    mat = np.eye(4)
    mat[0,0] = 1 - 2*y*y - 2*z*z
    mat[0,1] = 2*x*y - 2*z*w
    mat[0,2] = 2*x*z + 2*y*w
    mat[1,0] = 2*x*y + 2*z*w
    mat[1,1] = 1 - 2*x*x - 2*z*z
    mat[1,2] = 2*y*z - 2*x*w
    mat[2,0] = 2*x*z - 2*y*w
    mat[2,1] = 2*y*z + 2*x*w
    mat[2,2] = 1 - 2*x*x - 2*y*y
    mat[0,3] = t[0]
    mat[1,3] = t[1]
    mat[2,3] = t[2]
    return mat

class RoboCupMapper(Node):
    def __init__(self):
        super().__init__('robocup_mapper')
        
        # Parámetros editables
        self.declare_parameter('mission_name', 'MissionX')
        self.declare_parameter('team_name', 'UruBots')
        self.declare_parameter('voxel_size', 0.03) # 3cm de resolución
        
        self.mission = self.get_parameter('mission_name').value
        self.team = self.get_parameter('team_name').value
        self.voxel_size = self.get_parameter('voxel_size').value
        
        self.fixed_frame = 'odom'
        self.pc_topic = '/point_cloud2'
        self.img_topic = '/go2_camera/color/image'
        self.info_topic = '/go2_camera/color/camera_info'
        
        # Fallbacks si cambian los tópicos
        self.declare_parameter('image_fallback', '/camera/image_raw')
        self.declare_parameter('info_fallback', '/camera/camera_info')
        self.img_fallback = self.get_parameter('image_fallback').value
        self.info_fallback = self.get_parameter('info_fallback').value
        
        # TF
        self.tf_buffer = Buffer(rclpy.time.Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Variables de estado
        self.global_pcd = o3d.geometry.PointCloud()
        self.latest_image = None
        self.camera_info = None
        self.bridge = CvBridge()
        self.start_time = datetime.datetime.now()
        
        from rclpy.qos import qos_profile_sensor_data
        
        # Suscripciones con QoS de Sensor (Best Effort) requerido por Unitree
        self.create_subscription(CameraInfo, self.info_topic, self.info_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.info_fallback, self.info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, self.img_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Image, self.img_fallback, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.pc_topic, self.pc_callback, qos_profile_sensor_data)
        
        self.get_logger().info(f"🏆 RoboCup 3D Mapper Iniciado - Equipo: {self.team}")
        self.get_logger().info("==================================================")
        self.get_logger().info("🔥 Bono de Color: ACTIVADO (Esperando imagen...)")
        self.get_logger().info("📡 Caminá despacio para mapear.")
        self.get_logger().info("🛑 Presiona Ctrl+C cuando termines para guardar el .ply de RoboCup.")
        self.get_logger().info("==================================================")

    def info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("✅ Cámara Info detectada. Colorización Lista.")

    def image_callback(self, msg):
        self.latest_image = msg

    def pc_callback(self, msg):
        # 1. Transformación de LiDAR a Marco Fijo (Odometría del inicio = (0,0,0))
        try:
            t_odom_radar = self.tf_buffer.lookup_transform(
                self.fixed_frame, 
                msg.header.frame_id, 
                rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().debug(f"TF Error: {e}")
            return

        # 2. Extraer puntos geométricos X, Y, Z
        p_list = [ [p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True) ]
        points = np.array(p_list, dtype=np.float64)
        if len(points) == 0:
            return

        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(points)
        
        # 3. Proyección RGB para ganar puntos extra en RoboCup
        # Por defecto la nube es de un gris técnico
        colors = np.ones((len(points), 3)) * 0.7
        
        if self.latest_image is not None and self.camera_info is not None:
            try:
                t_cam_radar = self.tf_buffer.lookup_transform(
                    self.camera_info.header.frame_id,
                    msg.header.frame_id,
                    rclpy.time.Time()
                )
                
                # Imagen a OpenCV
                cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='rgb8')
                cv_img_float = cv_img.astype(float) / 255.0
                
                # Intrínsecos de cámara
                K = np.array(self.camera_info.k).reshape((3, 3))
                
                # Construir matriz de proyección Lidar -> Cámara
                trans = t_cam_radar.transform.translation
                rot = t_cam_radar.transform.rotation
                q = [rot.x, rot.y, rot.z, rot.w]
                t = [trans.x, trans.y, trans.z]
                mat = quat_to_mat(q, t)
                
                points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
                points_cam = (mat @ points_hom.T).T
                
                # Z > 0 significa enfrente de la cámara
                z_mask = points_cam[:, 2] > 0.1
                valid_idx = np.where(z_mask)[0]
                
                if len(valid_idx) > 0:
                    # Perspectiva 2D
                    uv = (K @ points_cam[valid_idx, :3].T).T
                    u = (uv[:, 0] / uv[:, 2]).astype(int)
                    v = (uv[:, 1] / uv[:, 2]).astype(int)
                    
                    height, width = cv_img.shape[:2]
                    bounds_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
                    
                    final_idx = valid_idx[bounds_mask]
                    final_u = u[bounds_mask]
                    final_v = v[bounds_mask]
                    
                    # Pintar los puntos frontales
                    colors[final_idx] = cv_img_float[final_v, final_u, :]
            except Exception:
                pass # Falla silenciosa si no hay TF de cámara en este frame
                
        scan_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 4. Transformar nube local al mundo real
        trans = t_odom_radar.transform.translation
        rot = t_odom_radar.transform.rotation
        q = [rot.x, rot.y, rot.z, rot.w]
        t = [trans.x, trans.y, trans.z]
        mat_world = quat_to_mat(q, t)
        
        scan_pcd.transform(mat_world)
        
        # 5. Acumular y optimizar
        self.global_pcd += scan_pcd
        # Sub-muestreo periódico para evitar saturar memoria RAM
        if len(self.global_pcd.points) > 2000000:
            self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=self.voxel_size)

    def save_robocup_map(self):
        self.get_logger().info("\n💾 ¡Guardando mapa RoboCup, un momento!...")
        
        if len(self.global_pcd.points) == 0:
            self.get_logger().error("❌ El mapa está vacío.")
            return

        # Voxel final para limpiar y estandarizar densidad según reglas
        final_pcd = self.global_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # REGLA ROBOCUP: Rotar matemáticamente para que el Norte sea el eje +Y
        # Originalmente ROS2 tiene el frente como +X.
        R = final_pcd.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
        final_pcd.rotate(R, center=(0,0,0))
        
        # REGLA ROBOCUP: El origen (0,0,0) debe ser el Centro del Frente a la altura del Suelo.
        points_np = np.asarray(final_pcd.points)
        # Buscar el suelo cerca del robot (X e Y cerca de 0)
        mask_center = (np.abs(points_np[:, 0]) < 0.5) & (np.abs(points_np[:, 1]) < 0.5)
        floor_z = np.min(points_np[mask_center, 2]) if np.any(mask_center) else -0.3
        
        # Trasladar Z para que el suelo sea 0, y Y para que el origen sea el frente (Go2 mide ~0.7m de largo -> offset 0.35m)
        final_pcd.translate((0, -0.35, -floor_z))
        
        points = np.asarray(final_pcd.points)
        colors = np.asarray(final_pcd.colors) * 255.0
        colors = colors.astype(np.uint8)
        
        time_str = self.start_time.strftime("%H-%M-%S")
        year = self.start_time.strftime("%Y")
        mapas_dir = os.path.expanduser('~/ros2_ws/Rescue-Go2/mapas')
        os.makedirs(mapas_dir, exist_ok=True)
        filename = os.path.join(mapas_dir, f"RoboCup{year}-{self.team}-{self.mission}-{time_str}-map.ply")
        
        # Generar archivo estrictamente ASCII PLY como lo dictan los jueces
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Team {self.team}\n")
            f.write(f"comment Start time {time_str}\n")
            f.write(f"comment Mission {self.mission}\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i]
                # Floats limpios en Metros
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
                
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"✅ ¡MAPA EXPORTADO CON ÉXITO PARA LOS JUECES!")
        self.get_logger().info(f"📂 Archivo: {filename}")
        self.get_logger().info(f"📏 Puntos totales: {len(points)}")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = RoboCupMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_robocup_map()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
