#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import numpy as np

class AutonomousWanderer(Node):
    def __init__(self):
        super().__init__('autonomous_wanderer')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.listener_callback,
            qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.last_scan = None
        self.get_logger().info('¡Nodo Explorador Autónomo Iniciado!')

    def listener_callback(self, msg):
        self.last_scan = msg

    def timer_callback(self):
        if self.last_scan is None:
            return
            
        # Obtener rangos de distancia del LiDAR
        ranges = np.array(self.last_scan.ranges)
        # Limpiar valores inf o nan
        ranges = np.where(np.isnan(ranges) | np.isinf(ranges), 20.0, ranges)
        
        # El sector frontal está en el centro de la lectura (por ejemplo, entre -30 y +30 grados)
        num_samples = len(ranges)
        middle = num_samples // 2
        span = int(num_samples * 30 / 360)  # +/- 30 grados
        
        front_min = middle - span
        front_max = middle + span
        
        front_ranges = ranges[front_min:front_max]
        min_distance = np.min(front_ranges)
        
        twist = Twist()
        # Si hay un obstáculo a menos de 1.2 metros en frente
        if min_distance < 1.2:
            twist.linear.x = 0.0
            twist.angular.z = 0.6  # Girar a la izquierda
            self.get_logger().info(f'⚠️ Obstáculo a {min_distance:.2f}m. Girando...', throttle_duration_sec=1.0)
        else:
            twist.linear.x = 0.35  # Avanzar de frente
            twist.angular.z = 0.0
            self.get_logger().info('🟢 Camino despejado. Avanzando...', throttle_duration_sec=2.0)
            
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousWanderer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Detener al robot al apagar el nodo
        stop_twist = Twist()
        node.publisher_.publish(stop_twist)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
