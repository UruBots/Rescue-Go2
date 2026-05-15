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
        self._pending_hits = [] # Guardará (time, name, x, y) para exigir 3 detecciones antes de registrar
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

    def is_duplicate(self, name, x, y, z, threshold=1.0):
        """Evita registrar el mismo objeto varias veces si esta a menos de 1 metro"""
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
        
        # --- COOLDOWN POR CLASE + DISTANCIA DEL ROBOT ---
        # No registrar el mismo objeto si el robot NO se movio mas de 2.5m
        # desde la ultima vez que lo detecto (independiente del tiempo).
        now = datetime.datetime.now()
        if not hasattr(self, '_cooldown'):
            self._cooldown = {}  # name -> {'time': datetime, 'rx': float, 'ry': float}
        
        # Obtener posicion actual del robot
        robot_x, robot_y = 0.0, 0.0
        try:
            t_robot = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            robot_x = t_robot.transform.translation.x
            robot_y = t_robot.transform.translation.y
        except Exception:
            pass
        
        if name in self._cooldown:
            cd = self._cooldown[name]
            elapsed = (now - cd['time']).total_seconds()
            dist_moved = ((robot_x - cd['rx'])**2 + (robot_y - cd['ry'])**2)**0.5
            
            # Si pasaron menos de 60s Y no se movio mas de 1.0m, ignorar
            if elapsed < 60.0 and dist_moved < 1.0:
                # RESETEAR TIMER: sigue viendolo, asi que el cooldown empieza de nuevo ahora
                self._cooldown[name]['time'] = now
                return
            
        try:
            # --- BUG FIX 1: usar el timestamp REAL de la nube para el lookup TF ---
            # rclpy.time.Time() = ultimo transform disponible (puede ser del futuro o pasado)
            # cloud_stamp = momento exacto en que el lidar capturo esa scan
            cloud_stamp = rclpy.time.Time.from_msg(self.latest_cloud.header.stamp)
            
            t_cam_radar = self.tf_buffer.lookup_transform(
                self.camera_info.header.frame_id,
                self.latest_cloud_frame,
                cloud_stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            t_odom_radar = self.tf_buffer.lookup_transform(
                'odom',
                self.latest_cloud_frame,
                cloud_stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception:
            # Fallback si el stamp exacto no esta en el buffer TF
            try:
                t_cam_radar = self.tf_buffer.lookup_transform(
                    self.camera_info.header.frame_id, self.latest_cloud_frame, rclpy.time.Time())
                t_odom_radar = self.tf_buffer.lookup_transform(
                    'odom', self.latest_cloud_frame, rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn(f"⚠️ [TF] Transformación fallida para {name}: {str(e)}")
                return

        # Extraer puntos del lidar
        p_list = [[p[0], p[1], p[2]] for p in pc2.read_points(
            self.latest_cloud, field_names=("x", "y", "z"), skip_nans=True)]
        points = np.array(p_list, dtype=np.float64)
        if len(points) == 0:
            return

        # Proyectar Lidar a Pixeles usando la TF del instante correcto
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
        box_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        in_box_idx = valid_idx[box_mask]
        
        if len(in_box_idx) > 0:
            median_pt = np.median(points[in_box_idx], axis=0)
            
            tr = t_odom_radar.transform.translation
            ro = t_odom_radar.transform.rotation
            mat_world = quat_to_mat([ro.x, ro.y, ro.z, ro.w], [tr.x, tr.y, tr.z])
            
            pt_hom = np.array([median_pt[0], median_pt[1], median_pt[2], 1.0])
            world_pt = mat_world @ pt_hom
            
            x_odom, y_odom, z_odom = world_pt[0], world_pt[1], world_pt[2]
            
            # --- FILTRO DE DISTANCIA MAXIMA ---
            # Si el objeto aparece a mas de 6m del robot, probablemente es una proyeccion espuria
            dist_from_robot = np.sqrt((x_odom - robot_x)**2 + (y_odom - robot_y)**2)
            if dist_from_robot > 6.0:
                self.get_logger().warn(f"⚠️ [{name}] Proyeccion espuria descartada (dist={dist_from_robot:.1f}m del robot)")
                return
            
            if not self.is_duplicate(name, x_odom, y_odom, z_odom):
                # --- SISTEMA DE CONFIRMACION (3 HITS MINIMO PARA OBJETOS REALES) ---
                if det_type != 'hazmat_sign':
                    # Limpiar hits muy viejos (> 5 segundos)
                    self._pending_hits = [h for h in self._pending_hits if (now - h[0]).total_seconds() < 5.0]
                    
                    # Buscar hits similares recientes
                    hits_similares = [h for h in self._pending_hits if h[1] == name and np.sqrt((h[2]-x_odom)**2 + (h[3]-y_odom)**2) < 1.0]
                    
                    if len(hits_similares) < 2:  # Necesitamos 2 anteriores + 1 actual = 3
                        self._pending_hits.append((now, name, x_odom, y_odom))
                        return  # No registrar todavia
                    
                    # Si llegamos aca, tenemos 3 hits confirmados! 
                    # Limpiamos los hits de este objeto para no volver a activarlo al instante
                    self._pending_hits = [h for h in self._pending_hits if not (h[1] == name and np.sqrt((h[2]-x_odom)**2 + (h[3]-y_odom)**2) < 1.0)]
                
                time_str = datetime.datetime.now().strftime("%H:%M:%S")
                self.detections.append({
                    'detection': self.detection_id,
                    'time': time_str,
                    'type': det_type,
                    'name': name,
                    'x': x_odom,
                    'y': y_odom,
                    'z': z_odom,
                    'robot': 'UruBots_Go2',
                    'mode': 'T'
                })
                self._cooldown[name] = {'time': now, 'rx': robot_x, 'ry': robot_y}
                self.get_logger().info(f"✅ DETECCIÓN {self.detection_id}: {name} en X:{x_odom:.2f} Y:{y_odom:.2f} Z:{z_odom:.2f}")
                self.detection_id += 1
        else:
            self.get_logger().warn(f"⚠️ [3D] {name} visto en cámara, pero el Láser no encontró puntos en esa caja (¿Muy cerca?).")

    def image_callback(self, msg):
        if not hasattr(self, 'first_image_received'):
            self.get_logger().info("📸 ¡Primera imagen de la cámara del perro RECIBIDA con éxito!")
            self.first_image_received = True
            self.image_count = 0
            
        self.image_count += 1
        if self.image_count % 150 == 0:  # Cada ~10 segundos a 15fps
            self.get_logger().info("🔄 Recibiendo y procesando stream de video...")

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
                    conf = box.conf[0]
                    b = box.xyxy[0].cpu().numpy()
                    
                    # Filtro de Bounding Box minima (ignorar cajitas muy chicas, suelen ser ruido)
                    width = b[2] - b[0]
                    height = b[3] - b[1]
                    if width < 30 or height < 30:  # pixeles
                        continue
                        
                    cls = int(box.cls[0])
                    name = self.unified_model.names[cls].lower()
                    
                    # Umbrales especificos por clase
                    # Clases propensas a falsos positivos necesitan mas confianza
                    threshold = 0.55  # 55% base (para hazmats)
                    if name in ['helmet', 'gas tank', 'backpack', 'person', 'suitcase', 'baby face', 'babyface']:
                        threshold = 0.82  # 82% para estas
                    elif name == 'fire_extinguisher':
                        threshold = 0.75  # 75% para extintor
                        
                    if conf > threshold:
                        # Determinar tipo de objeto para el CSV de RoboCup
                        if any(ro in name for ro in real_object_classes):
                            det_type = 'real_object'
                            self.get_logger().info(f"🔎 [2D] Objeto Real visto: {name} (Confianza: {box.conf[0]*100:.1f}%)")
                        else:
                            det_type = 'hazmat_sign'
                            self.get_logger().info(f"🔎 [2D] Hazmat visto: {name} (Confianza: {box.conf[0]*100:.1f}%)")
                            
                        self.register_detection(det_type, name, b)

    def cluster_detections(self, cluster_radius=3.0):
        """Agrupa detecciones del mismo tipo que esten cerca (por drift de odom)
        y las fusiona en una sola usando la posicion mediana."""
        if not self.detections:
            return []
        
        used = [False] * len(self.detections)
        clusters = []
        
        for i, d in enumerate(self.detections):
            if used[i]:
                continue
            group = [d]
            used[i] = True
            for j, d2 in enumerate(self.detections):
                if used[j] or i == j:
                    continue
                if d2['name'] != d['name']:
                    continue
                dist = ((d['x'] - d2['x'])**2 + (d['y'] - d2['y'])**2)**0.5
                if dist < cluster_radius:
                    group.append(d2)
                    used[j] = True
            
            # Tomar la mediana de X, Y, Z del grupo (mas robusta que el promedio)
            xs = sorted(g['x'] for g in group)
            ys = sorted(g['y'] for g in group)
            zs = sorted(g['z'] for g in group)
            mid = len(group) // 2
            merged = dict(group[0])
            merged['x'] = xs[mid]
            merged['y'] = ys[mid]
            merged['z'] = zs[mid]
            clusters.append(merged)
        
        return clusters

    def save_csv(self):
        time_str = self.start_time.strftime("%H-%M-%S")
        date_str = self.start_time.strftime("%Y-%m-%d")
        year = self.start_time.strftime("%Y")
        mapas_dir = os.path.expanduser('~/ros2_ws/Rescue-Go2/mapas')
        os.makedirs(mapas_dir, exist_ok=True)
        filename = os.path.join(mapas_dir, f"RoboCup{year}-{self.team}-{self.mission}-{time_str}-pois.csv")
        
        # Fusionar detecciones cercanas del mismo tipo antes de guardar
        final_detections = self.cluster_detections(cluster_radius=1.0)
        self.get_logger().info(f"📊 Detecciones brutas: {len(self.detections)} → tras clustering: {len(final_detections)}")
        
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
            for i, d in enumerate(final_detections, 1):
                writer.writerow([i, d['time'], d['type'], d['name'], 
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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
