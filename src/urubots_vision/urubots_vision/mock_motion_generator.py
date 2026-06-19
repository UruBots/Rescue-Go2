#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from rclpy.qos import qos_profile_sensor_data

class MockMotionGenerator(Node):
    def __init__(self):
        super().__init__('mock_motion_generator')
        
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', qos_profile_sensor_data)
        self.bridge = CvBridge()
        
        # Publish at 15 Hz
        self.timer = self.create_timer(1.0 / 15.0, self.timer_callback)
        
        self.theta = 0.0
        self.shape_index = 0
        self.shapes = ["SQUARE", "TRIANGLE", "RECTANGLE", "CIRCLE"]
        
        self.get_logger().info("Mock Motion Generator started. Publishing synthetic rotating disk to /camera/image_raw.")

    def timer_callback(self):
        # Create a dark background frame (640x480)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 20
        
        # Draw a stationary disk (large grey circle) in the center
        center_x, center_y = 320, 240
        cv2.circle(frame, (center_x, center_y), 180, (50, 50, 50), -1)
        cv2.circle(frame, (center_x, center_y), 180, (80, 80, 80), 2)
        
        # Calculate target position (rotating on the disk)
        radius = 110
        target_x = int(center_x + radius * math.cos(self.theta))
        target_y = int(center_y + radius * math.sin(self.theta))
        
        # Draw target based on shape index
        shape_type = self.shapes[self.shape_index]
        
        if shape_type == "SQUARE":
            # Draw a rotated square of size 55x55
            size = 55
            d = size / 2
            local_corners = np.array([[-d, -d], [d, -d], [d, d], [-d, d]])
            rot_matrix = np.array([
                [math.cos(self.theta), -math.sin(self.theta)],
                [math.sin(self.theta), math.cos(self.theta)]
            ])
            rotated_corners = (rot_matrix @ local_corners.T).T
            pts = (rotated_corners + np.array([target_x, target_y])).astype(np.int32)
            cv2.fillPoly(frame, [pts], (0, 255, 0)) # Green
            
        elif shape_type == "TRIANGLE":
            # Draw an equilateral triangle rotated
            size = 65
            h = size * math.sqrt(3) / 2
            local_corners = np.array([[0, -2*h/3], [-size/2, h/3], [size/2, h/3]])
            rot_matrix = np.array([
                [math.cos(self.theta), -math.sin(self.theta)],
                [math.sin(self.theta), math.cos(self.theta)]
            ])
            rotated_corners = (rot_matrix @ local_corners.T).T
            pts = (rotated_corners + np.array([target_x, target_y])).astype(np.int32)
            cv2.fillPoly(frame, [pts], (255, 180, 0)) # Bright Sky Blue
            
        elif shape_type == "RECTANGLE":
            # Draw a narrow rectangle (65x35)
            w, h = 65, 30
            local_corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
            rot_matrix = np.array([
                [math.cos(self.theta), -math.sin(self.theta)],
                [math.sin(self.theta), math.cos(self.theta)]
            ])
            rotated_corners = (rot_matrix @ local_corners.T).T
            pts = (rotated_corners + np.array([target_x, target_y])).astype(np.int32)
            cv2.fillPoly(frame, [pts], (0, 255, 255)) # Yellow
            
        elif shape_type == "CIRCLE":
            # Draw a small orange circle target
            cv2.circle(frame, (target_x, target_y), 28, (0, 165, 255), -1) # Orange
            
        # Draw target details
        cv2.circle(frame, (target_x, target_y), 4, (0, 0, 255), -1)
        
        # Add overlay text for test info
        cv2.putText(frame, f"TEST DISK GENERATOR", (20, 440), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, f"SHAPE: {shape_type}", (20, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Update rotation angle
        self.theta += 0.05 # ~3 degrees per frame
        if self.theta >= 2 * math.pi:
            self.theta = 0.0
            # Switch to next shape on full rotation
            self.shape_index = (self.shape_index + 1) % len(self.shapes)
            self.get_logger().info(f"Full 360 degree rotation completed! Switching to: {self.shapes[self.shape_index]}")
            
        # Publish image
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing mock image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MockMotionGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
