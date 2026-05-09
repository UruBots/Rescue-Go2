#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np
import datetime
import os
import csv
from pupil_apriltags import Detector
from ultralytics import YOLO

def quat_to_mat(q, t):
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

class VisionDetector(Node):
    def __init__(self):
        super().__init__('vision_detector')
        
        self.declare_parameter('mission_name', 'Mision_1')
        self.declare_parameter('team_name', 'UruBots')
        
        self.mission = self.get_parameter('mission_name').value
        self.team = self.get_parameter('team_name').value
        
        # Detector Setup
        self.bridge = CvBridge()
        self.get_logger().info("Cargando Modelos de IA...")
        
        # YOLO Unificado para Hazmat y Objetos Reales
        unified_path = os.path.expanduser('~/ros2_ws/Rescue-Go2/vision/best_all.pt')
        if os.path.exists(unified_path):
            self.unified_model = YOLO(unified_path)
            self.get_logger().info("✅ Súper-Modelo UNIFICADO (best_all) cargado.")
        else:
            self.get_logger().error("❌ No se encontró best_all.pt en la carpeta vision/")
            self.unified_model = None
        
        # AprilTag (pupil_apriltags soporta tag36h11 nativamente)
        self.at_detector = Detector(families='tag36h11')
        
        # TF y Topics
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/point_cloud2', self.pc_callback, qos_profile_sensor_data)
        
        self.camera_info = None
        self.latest_cloud = None
        self.latest_cloud_frame = None
        
        self.detections = [] # Guardará tuplas (time, type, name, x, y, z)
        self.detection_id = 1
        self.start_time = datetime.datetime.now()
        
        self.get_logger().info("==================================================")
        self.get_logger().info("👁️ Visión Activa. Buscando Hazmat, AprilTags y Objetos.")
        self.get_logger().info("==================================================")

    def info_callback(self, msg):
        self.camera_info = msg

    def pc_callback(self, msg):
        self.latest_cloud = msg
        self.latest_cloud_frame = msg.header.frame_id

    def is_duplicate(self, name, x, y, z, threshold=2.0):
        """Evita registrar el mismo objeto varias veces si esta a menos de 2 metros"""
        for d in self.detections:
            dx = d['x'] - x
            dy = d['y'] - y
            dz = d['z'] - z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < threshold and d['name'] == name:
                return True
        return False

    def register_detection(self, det_type, name, bbox):
        if not self.camera_info or not self.latest_cloud:
            return
            
        try:
            # TF Cámara a Odom
            t_odom_cam = self.tf_buffer.lookup_transform(
                'odom', 
                self.camera_info.header.frame_id, 
                rclpy.time.Time()
            )
            # TF Lidar a Cámara
            t_cam_radar = self.tf_buffer.lookup_transform(
                self.camera_info.header.frame_id, 
                self.latest_cloud_frame, 
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"⚠️ [TF] Transformación fallida para {name}: {str(e)}")
            return

        # Extraer puntos del lidar
        p_list = [ [p[0], p[1], p[2]] for p in pc2.read_points(self.latest_cloud, field_names=("x", "y", "z"), skip_nans=True) ]
        points = np.array(p_list, dtype=np.float64)
        if len(points) == 0:
            return

        # Proyectar Lidar a Píxeles
        trans = t_cam_radar.transform.translation
        rot = t_cam_radar.transform.rotation
        mat = quat_to_mat([rot.x, rot.y, rot.z, rot.w], [trans.x, trans.y, trans.z])
        
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
        points_cam = (mat @ points_hom.T).T
        z_mask = points_cam[:, 2] > 0.1
        valid_idx = np.where(z_mask)[0]
        
        if len(valid_idx) == 0:
            return
            
        K = np.array(self.camera_info.k).reshape((3, 3))
        uv = (K @ points_cam[valid_idx, :3].T).T
        u = (uv[:, 0] / uv[:, 2]).astype(int)
        v = (uv[:, 1] / uv[:, 2]).astype(int)
        
        x1, y1, x2, y2 = bbox
        # Encontrar puntos que caen dentro de la Bounding Box
        box_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        in_box_idx = valid_idx[box_mask]
        
        if len(in_box_idx) > 0:
            # Calcular Centroide 3D local en base a la nube de puntos
            median_pt = np.median(points[in_box_idx], axis=0)
            
            # Convertir a ODOM
            t_odom_radar = self.tf_buffer.lookup_transform('odom', self.latest_cloud_frame, rclpy.time.Time())
            tr = t_odom_radar.transform.translation
            ro = t_odom_radar.transform.rotation
            mat_world = quat_to_mat([ro.x, ro.y, ro.z, ro.w], [tr.x, tr.y, tr.z])
            
            pt_hom = np.array([median_pt[0], median_pt[1], median_pt[2], 1.0])
            world_pt = mat_world @ pt_hom
            
            x_odom, y_odom, z_odom = world_pt[0], world_pt[1], world_pt[2]
            
            # Guardar coordenadas en odom puro (sin rotar)
            # El GeoTIFF aplicará la misma transformación que usa para el path
            final_x = x_odom
            final_y = y_odom
            final_z = z_odom
            
            if not self.is_duplicate(name, final_x, final_y, final_z):
                time_str = datetime.datetime.now().strftime("%H:%M:%S")
                self.detections.append({
                    'detection': self.detection_id,
                    'time': time_str,
                    'type': det_type,
                    'name': name,
                    'x': final_x,
                    'y': final_y,
                    'z': final_z,
                    'robot': 'UruBots_Go2',
                    'mode': 'T' # T para teleop
                })
                self.get_logger().info(f"✅ DETECCIÓN {self.detection_id}: {name} en X:{final_x:.2f} Y:{final_y:.2f} Z:{final_z:.2f}")
                self.detection_id += 1
        else:
            self.get_logger().warn(f"⚠️ [3D] {name} visto en cámara, pero el Láser no encontró puntos en esa caja (¿Muy cerca?).")


    def image_callback(self, msg):
        if not hasattr(self, 'first_image_received'):
            self.get_logger().info("📸 ¡Primera imagen de la cámara del perro RECIBIDA con éxito!")
            self.first_image_received = True

        if self.latest_cloud is None:
            # Si aún no tenemos LiDAR, no procesamos visión para no gastar CPU en vano
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen: {e}")
            return
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Detectar AprilTags
        if self.camera_info:
            K = self.camera_info.k
            camera_params = [K[0], K[4], K[2], K[5]] # fx, fy, cx, cy
            tags = self.at_detector.detect(gray_img, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.15)
            for tag in tags:
                bbox = [min(tag.corners[:,0]), min(tag.corners[:,1]), max(tag.corners[:,0]), max(tag.corners[:,1])]
                self.register_detection('ar_code', str(tag.tag_id), bbox)
                
        # 2. Detectar Hazmats y Objetos Reales (Modelo Unificado)
        if self.unified_model:
            results = self.unified_model(cv_img, verbose=False)
            
            # Lista de clases que consideramos "Objetos Reales" segun el reglamento
            # Nombres exactos del modelo best_all.pt: Backpack, baby face, babyface, fire_extinguisher, gas tank, helmet
            real_object_classes = ['backpack', 'baby face', 'babyface', 'fire_extinguisher', 'gas tank', 'helmet', 'person', 'suitcase']
            
            for r in results:
                for box in r.boxes:
                    if box.conf[0] > 0.5:
                        b = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        name = self.unified_model.names[cls].lower()
                        
                        # Determinar tipo de objeto para el CSV de RoboCup
                        if any(ro in name for ro in real_object_classes):
                            det_type = 'real_object'
                            self.get_logger().info(f"🔎 [2D] Objeto Real visto: {name} (Confianza: {box.conf[0]*100:.1f}%)")
                        else:
                            det_type = 'hazmat_sign'
                            self.get_logger().info(f"🔎 [2D] Hazmat visto: {name} (Confianza: {box.conf[0]*100:.1f}%)")
                            
                        self.register_detection(det_type, name, b)

    def save_csv(self):
        time_str = self.start_time.strftime("%H-%M-%S")
        date_str = self.start_time.strftime("%Y-%m-%d")
        year = self.start_time.strftime("%Y")
        mapas_dir = os.path.expanduser('~/ros2_ws/Rescue-Go2/mapas')
        os.makedirs(mapas_dir, exist_ok=True)
        filename = os.path.join(mapas_dir, f"RoboCup{year}-{self.team}-{self.mission}-{time_str}-pois.csv")
        
        with open(filename, 'w', newline='') as f:
            f.write('"pois"\n')
            f.write('"1.3"\n')
            f.write(f'"{self.team}"\n')
            f.write('"Uruguay"\n')
            f.write(f'"{date_str}"\n')
            f.write(f'"{time_str}"\n')
            f.write(f'"{self.mission}"\n')
            f.write('detection,time,type,name,x,y,z,robot,mode\n')
            
            writer = csv.writer(f)
            for d in self.detections:
                writer.writerow([d['detection'], d['time'], d['type'], d['name'], 
                               f"{d['x']:.4f}", f"{d['y']:.4f}", f"{d['z']:.4f}", 
                               f"\"{d['robot']}\"", d['mode']])
                               
        self.get_logger().info(f"✅ CSV DE ROBOCUP GUARDADO: {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = VisionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_csv()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
