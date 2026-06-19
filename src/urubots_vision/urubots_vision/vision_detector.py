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
        
        # Motion detection and tracking (RoboCup Section C)
        self.image_pub = self.create_publisher(Image, '/vision/motion_detection/image', qos_profile_sensor_data)
        self.prev_gray = None
        self.trail_points = []
        self.centroid_history = []
        self.prev_centroid = None
        self.missed_frames = 0

        self.get_logger().info("==================================================")
        self.get_logger().info("👁️ Visión Activa. Buscando Hazmat, AprilTags y Objetos.")
        self.get_logger().info("==================================================")

    def info_callback(self, msg):
        self.camera_info = msg

    def pc_callback(self, msg):
        self.latest_cloud = msg
        self.latest_cloud_frame = msg.header.frame_id

    def is_duplicate(self, name, x, y, z, threshold=2.5):
        """Evita registrar el mismo objeto varias veces si esta a menos de 2.5 metros"""
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
                    hits_similares = [h for h in self._pending_hits if h[1] == name and np.sqrt((h[2]-x_odom)**2 + (h[3]-y_odom)**2) < 2.5]
                    
                    if len(hits_similares) < 2:  # Necesitamos 2 anteriores + 1 actual = 3
                        self._pending_hits.append((now, name, x_odom, y_odom))
                        return  # No registrar todavia
                    
                    # Si llegamos aca, tenemos 3 hits confirmados! 
                    # Limpiamos los hits de este objeto para no volver a activarlo al instante
                    self._pending_hits = [h for h in self._pending_hits if not (h[1] == name and np.sqrt((h[2]-x_odom)**2 + (h[3]-y_odom)**2) < 2.5)]
                
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

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error decodificando imagen: {e}")
            return

        # ==========================================
        # 3. Motion Detection and Target Tracking (RoboCup Section C)
        # ==========================================
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        annotated_img = cv_img.copy()
        target_contour = None
        
        if self.prev_gray is not None:
            frame_diff = cv2.absdiff(self.prev_gray, gray)
            _, thresh = cv2.threshold(frame_diff, 12, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=1)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the solid moving target contour (excluding hollow disk edges or stretched arms)
            candidates = []
            for c in contours:
                area = cv2.contourArea(c)
                if 120 < area < 8000:
                    (x, y, w, h) = cv2.boundingRect(c)
                    aspect_ratio = float(w) / h if h > 0 else 1.0
                    if 0.5 < aspect_ratio < 2.0:
                        # Check solidity (solid target vs hollow/thin circular edges)
                        hull = cv2.convexHull(c)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / hull_area if hull_area > 0 else 0
                        if solidity > 0.70:
                            cx = x + w // 2
                            cy = y + h // 2
                            candidates.append((c, area, cx, cy))
            
            # Select the best target contour using centroid distance tracking
            if len(candidates) > 0:
                self.missed_frames = 0
                if self.prev_centroid is not None:
                    best_cand = None
                    min_dist = float('inf')
                    for cand in candidates:
                        dist = np.sqrt((cand[2] - self.prev_centroid[0])**2 + (cand[3] - self.prev_centroid[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_cand = cand
                    
                    if min_dist < 160:
                        target_contour = best_cand[0]
                        self.prev_centroid = (best_cand[2], best_cand[3])
                    else:
                        largest_cand = max(candidates, key=lambda x: x[1])
                        target_contour = largest_cand[0]
                        self.prev_centroid = (largest_cand[2], largest_cand[3])
                else:
                    largest_cand = max(candidates, key=lambda x: x[1])
                    target_contour = largest_cand[0]
                    self.prev_centroid = (largest_cand[2], largest_cand[3])
            else:
                self.missed_frames += 1
                if self.missed_frames > 20:
                    self.prev_centroid = None
            
            if target_contour is not None:
                (x, y, w, h) = cv2.boundingRect(target_contour)
                cx = x + w // 2
                cy = y + h // 2
                
                # Smooth box coordinates using history
                self.centroid_history.append((x, y, w, h, cx, cy))
                if len(self.centroid_history) > 5:
                    self.centroid_history.pop(0)
                
                avg_x = int(sum(pt[0] for pt in self.centroid_history) / len(self.centroid_history))
                avg_y = int(sum(pt[1] for pt in self.centroid_history) / len(self.centroid_history))
                avg_w = int(sum(pt[2] for pt in self.centroid_history) / len(self.centroid_history))
                avg_h = int(sum(pt[3] for pt in self.centroid_history) / len(self.centroid_history))
                avg_cx = int(sum(pt[4] for pt in self.centroid_history) / len(self.centroid_history))
                avg_cy = int(sum(pt[5] for pt in self.centroid_history) / len(self.centroid_history))
                
                # Robust rotation-invariant shape classification using minimum area bounding box
                area = cv2.contourArea(target_contour)
                rect = cv2.minAreaRect(target_contour)
                (box_w, box_h) = rect[1]
                rect_area = box_w * box_h
                extent_rotated = float(area) / rect_area if rect_area > 0 else 0
                
                # Secondary feature: vertex count
                peri = cv2.arcLength(target_contour, True)
                approx = cv2.approxPolyDP(target_contour, 0.04 * peri, True)
                vertices = len(approx)
                
                if extent_rotated < 0.65 or vertices == 3:
                    shape_name = "TRIANGLE"
                    color = (255, 180, 0) # Sky Blue
                elif extent_rotated >= 0.83 or vertices == 4:
                    side_ratio = max(box_w, box_h) / min(box_w, box_h) if min(box_w, box_h) > 0 else 1.0
                    if side_ratio < 1.2:
                        shape_name = "SQUARE"
                        color = (0, 255, 0) # Green
                    else:
                        shape_name = "RECTANGLE"
                        color = (0, 255, 255) # Yellow
                else:
                    shape_name = "CIRCLE"
                    color = (0, 165, 255) # Orange
                
                # Draw bounding box and centroid
                cv2.rectangle(annotated_img, (avg_x, avg_y), (avg_x + avg_w, avg_y + avg_h), color, 2)
                cv2.circle(annotated_img, (avg_cx, avg_cy), 5, (0, 0, 255), -1)
                
                # Update trail
                self.trail_points.append((avg_cx, avg_cy))
                if len(self.trail_points) > 120:
                    self.trail_points.pop(0)
                    
                cv2.putText(annotated_img, f"{shape_name} DETECTED", (avg_x, avg_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw tracking trail history
            for i in range(1, len(self.trail_points)):
                cv2.line(annotated_img, self.trail_points[i-1], self.trail_points[i], (0, 242, 254), 2)
                
            # Draw premium status HUD overlays
            cv2.rectangle(annotated_img, (10, 10), (320, 110), (20, 22, 34), -1)
            cv2.rectangle(annotated_img, (10, 10), (320, 110), (34, 38, 56), 1)
            
            cv2.putText(annotated_img, "ROBOCUP MOTION TRACKER", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 2)
            
            if target_contour is not None:
                cv2.putText(annotated_img, "STATUS: TRACKING ACTIVE", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated_img, f"PATH: {len(self.trail_points)} pts (360 deg tracking)", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_img, f"COORDINATES: X={avg_cx} Y={avg_cy}", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(annotated_img, "STATUS: SCANNING FOR MOTION", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                cv2.putText(annotated_img, "Waiting for rotating target...", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (138, 149, 165), 1)
        
        self.prev_gray = gray
        
        # Publish the annotated frame
        try:
            pub_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8")
            self.image_pub.publish(pub_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing annotated image: {e}")

        if self.latest_cloud is None:
            # Si aún no tenemos LiDAR, no procesamos el resto de la visión para no gastar CPU en vano
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
        final_detections = self.cluster_detections(cluster_radius=2.5)
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
