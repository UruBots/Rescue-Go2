#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import cv2
import numpy as np
import datetime
import os
import csv
import glob

class GeoTiffMapper(Node):
    def __init__(self):
        super().__init__('geotiff_mapper')
        
        self.declare_parameter('mission_name', 'Mision_1')
        self.declare_parameter('team_name', 'UruBots')
        self.mission = self.get_parameter('mission_name').value
        self.team = self.get_parameter('team_name').value
        
        # Suscripción al mapa 2D (SLAM Toolbox)
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile_sensor_data)
        
        # TF para rastrear el camino
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.track_path)
        
        self.latest_map = None
        self.robot_path = [] # Lista de (X, Y) en odom
        self.start_time = datetime.datetime.now()
        
        self.get_logger().info("🗺️ GeoTIFF 2D Mapper Iniciado. Rastreando ruta del robot...")

    def map_callback(self, msg):
        self.latest_map = msg

    def track_path(self):
        try:
            # Obtener la posicion del robot respecto al origen (odom)
            t = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            self.robot_path.append((x, y))
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

    def draw_marker(self, img, x_px, y_px, type_name, label, ppm):
        # Tamanos fijos en pixeles para que sean visibles
        marker_size = max(20, int(0.35 * ppm))
        half = marker_size // 2
        
        if type_name == 'ar_code':
            # Circulo Amarillo 255,200,0
            cv2.circle(img, (x_px, y_px), half, (0, 200, 255), -1)
            cv2.circle(img, (x_px, y_px), half, (0, 0, 0), 2)
            text = f"#{label}"
        elif type_name == 'hazmat_sign':
            # Rombo Naranja 255,100,30
            pts = np.array([
                [x_px, y_px - half], [x_px + half, y_px],
                [x_px, y_px + half], [x_px - half, y_px]
            ], np.int32)
            cv2.fillPoly(img, [pts], (30, 100, 255))
            cv2.polylines(img, [pts], True, (0, 0, 0), 2)
            text = label[:2].upper()
        else:  # real_object
            # Rombo Rojo 240,10,10
            pts = np.array([
                [x_px, y_px - half], [x_px + half, y_px],
                [x_px, y_px + half], [x_px - half, y_px]
            ], np.int32)
            cv2.fillPoly(img, [pts], (10, 10, 240))
            cv2.polylines(img, [pts], True, (0, 0, 0), 2)
            text = label[:2].upper()
            
        # Dibujar texto dentro del marcador
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, marker_size / 40.0)
        thickness = max(1, int(font_scale * 2))
        size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        tx = x_px - size[0] // 2
        ty = y_px + size[1] // 2
        cv2.putText(img, text, (tx, ty), font, font_scale, (255, 255, 255), thickness)
        
        # Dibujar nombre completo debajo del marcador
        name_scale = max(0.4, marker_size / 50.0)
        name_thick = max(1, int(name_scale * 2))
        name_size = cv2.getTextSize(label, font, name_scale, name_thick)[0]
        nx = x_px - name_size[0] // 2
        ny = y_px + half + name_size[1] + 5
        cv2.putText(img, label, (nx, ny), font, name_scale, (0, 0, 0), name_thick)

    def save_geotiff(self):
        self.get_logger().info("🎨 Dibujando Mapa GeoTIFF (Reglas RoboCup)...")
        if not self.latest_map:
            self.get_logger().error("No se ha recibido ningún mapa de SLAM.")
            return
            
        res = self.latest_map.info.resolution
        w = self.latest_map.info.width
        h = self.latest_map.info.height
        
        # Pixels per meter
        ppm = int(1.0 / res) if res > 0 else 20
        
        # Garantizar un tamaño mínimo (ej. 1200x1200 píxeles, ~60 metros) para que los textos no se pisen
        W = max(w, 1200)
        H = max(h, 1200)
        offset_x = (W - w) // 2
        offset_y = (H - h) // 2

        # Regla: Unexplored Checkerboard 100cm (1 metro) = ppm píxeles
        img = np.zeros((H, W, 3), dtype=np.uint8)
        c1 = (227, 226, 226) # Light Grey (BGR)
        c2 = (238, 237, 237) # Dark Grey (BGR)
        for i in range(0, W, ppm):
            for j in range(0, H, ppm):
                color = c1 if ((i//ppm) + (j//ppm)) % 2 == 0 else c2
                cv2.rectangle(img, (i, j), (i+ppm, j+ppm), color, -1)
                
        # Procesar Occupancy Grid
        data = np.array(self.latest_map.data).reshape((h, w))
        
        # Máscaras
        explored_mask = data >= 0
        walls_mask = data > 50
        free_mask = (data >= 0) & (data <= 50)
        
        # Crear capa del SLAM
        img_slam = np.zeros((h, w, 3), dtype=np.uint8)
        img_slam[free_mask] = (255, 255, 255)
        
        # Dibujar grilla negra fina (50cm) sobre área explorada
        grid_50 = ppm // 2
        for i in range(0, w, grid_50):
            img_slam[:, i] = np.where(explored_mask[:, i, np.newaxis], (191, 190, 190), img_slam[:, i])
        for j in range(0, h, grid_50):
            img_slam[j, :] = np.where(explored_mask[j, :, np.newaxis], (191, 190, 190), img_slam[j, :])
            
        # Pintar muros
        img_slam[walls_mask] = (120, 40, 0)
        
        # Superponer el SLAM explorado en el centro del mapa grande
        img[offset_y:offset_y+h, offset_x:offset_x+w] = np.where(
            explored_mask[:,:,np.newaxis], 
            img_slam, 
            img[offset_y:offset_y+h, offset_x:offset_x+w]
        )
        
        # Offset del mapa (para proyectar coordenadas métricas a píxeles)
        # La orientación en ROS es +X a la derecha, +Y arriba (dependiendo del marco).
        # En el mapa 2D, el origin está en la esquina inferior izquierda.
        origin_x = self.latest_map.info.origin.position.x
        origin_y = self.latest_map.info.origin.position.y
        
        def world_to_px(wx, wy):
            # Convertir coordenadas odom directamente a pixeles del mapa SLAM
            px = int((wx - origin_x) / res)
            py = int((wy - origin_y) / res)
            # Invertir Y porque en la imagen Y crece hacia abajo, y aplicar offset
            return (px + offset_x, offset_y + h - py)
            
        # Regla: Robot Path Magenta (120, 0, 140) -> BGR (140, 0, 120), grosor 2cm
        path_thick = max(1, int(0.02 * ppm))
        for i in range(1, len(self.robot_path)):
            pt1 = world_to_px(*self.robot_path[i-1])
            pt2 = world_to_px(*self.robot_path[i])
            cv2.line(img, pt1, pt2, (140, 0, 120), path_thick)
            
        # Regla: Initial Position (Green Arrow pointing UP) (0, 240, 0) -> BGR (0, 240, 0)
        if len(self.robot_path) > 0:
            start_px = world_to_px(*self.robot_path[0])
            cv2.arrowedLine(img, (start_px[0], start_px[1] + int(0.5*ppm)), start_px, (0, 240, 0), max(2, int(0.05*ppm)), tipLength=0.3)
            
        # Dibujar POIs desde CSV
        csv_files = glob.glob(os.path.expanduser(f"~/ros2_ws/Rescue-Go2/mapas/RoboCup*{self.mission}*pois.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=os.path.getctime)
            with open(latest_csv, 'r') as f:
                reader = csv.reader(f)
                headers = []
                for row in reader:
                    if len(row) > 0 and row[0] == 'detection':
                        headers = row
                        continue
                    if headers and len(row) >= 7:
                        # detection,time,type,name,x,y,z...
                        rtype = row[2]
                        name = row[3]
                        wx = float(row[4])
                        wy = float(row[5])
                        px, py = world_to_px(wx, wy)
                        self.draw_marker(img, px, py, rtype, name, ppm)
                        
        # Regla: Nombre de archivo en Azul Oscuro
        time_str = self.start_time.strftime("%H-%M-%S")
        year = self.start_time.strftime("%Y")
        filename = f"RoboCup{year}-{self.team}-{self.mission}-{time_str}-map.tiff"
        
        cv2.putText(img, filename, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (207, 44, 0), 2) # BGR
        
        # Regla: Escala de 1 metro y Orientación (se colocan relativas al nuevo ancho W)
        scale_x = W - ppm - 20
        scale_y = 40
        cv2.line(img, (scale_x, scale_y), (scale_x + ppm, scale_y), (140, 50, 0), 3)
        cv2.putText(img, "1 METER", (scale_x, scale_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 50, 0), 2)
        
        cv2.arrowedLine(img, (scale_x, scale_y + 40), (scale_x, scale_y + 40 - int(0.5*ppm)), (140, 50, 0), 2)
        cv2.putText(img, "X", (scale_x - 5, scale_y + 40 - int(0.5*ppm) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 50, 0), 2)
        cv2.arrowedLine(img, (scale_x, scale_y + 40), (scale_x - int(0.5*ppm), scale_y + 40), (140, 50, 0), 2)
        cv2.putText(img, "Y", (scale_x - int(0.5*ppm) - 20, scale_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 50, 0), 2)

        # Guardar archivo
        mapas_dir = os.path.expanduser('~/ros2_ws/Rescue-Go2/mapas')
        os.makedirs(mapas_dir, exist_ok=True)
        full_path = os.path.join(mapas_dir, filename)
        cv2.imwrite(full_path, img)
        self.get_logger().info(f"✅ MAPA 2D GEOTIFF CREADO: {full_path}")


def main(args=None):
    rclpy.init(args=args)
    node = GeoTiffMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_geotiff()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
