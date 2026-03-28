#!/usr/bin/python3
"""
QR Code Camera Viewer for Unitree Go2
Opens a window showing the robot's camera feed with QR code detection.
When a QR is detected, draws a green bounding box and shows the content.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("pyzbar not found, using OpenCV QR detector (less reliable)")


class QRCameraViewer(Node):
    def __init__(self):
        super().__init__('qr_camera_viewer')

        self.declare_parameter('camera_topic', '/camera/image_raw')
        camera_topic = self.get_parameter('camera_topic').value

        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        self.last_detections = []

        # QoS compatible with Go2 camera (BEST_EFFORT)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos
        )

        self.get_logger().info(f'QR Camera Viewer started on {camera_topic}')
        self.get_logger().info('Press "q" on the window to quit')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Detect QR codes
        detections = self.detect_qr_codes(cv_image)

        # Draw detections
        for det in detections:
            x, y, w, h = det['bbox']
            data = det['data']

            # Green bounding box
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Label background
            label = f'QR: {data}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw filled rectangle behind text
            cv2.rectangle(cv_image, (x, y - th - 15), (x + tw + 10, y), (0, 255, 0), -1)
            cv2.putText(cv_image, label, (x + 5, y - 10), font, font_scale, (0, 0, 0), thickness)

            self.get_logger().info(f'QR Detected: {data}')

        # Status text
        status = f'QR Codes: {len(detections)}' if detections else 'Scanning for QR codes...'
        cv2.putText(cv_image, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show window
        cv2.imshow('Go2 Camera - QR Reader', cv_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quitting...')
            raise SystemExit

    def detect_qr_codes(self, image):
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if PYZBAR_AVAILABLE:
            decoded = pyzbar.decode(gray)
            for d in decoded:
                detections.append({
                    'data': d.data.decode('utf-8'),
                    'bbox': [d.rect.left, d.rect.top, d.rect.width, d.rect.height]
                })
        else:
            data, bbox, _ = self.qr_detector.detectAndDecode(gray)
            if data and bbox is not None:
                x = int(bbox[0][0][0])
                y = int(bbox[0][0][1])
                w = int(bbox[0][2][0]) - x
                h = int(bbox[0][2][1]) - y
                detections.append({
                    'data': data,
                    'bbox': [x, y, w, h]
                })

        return detections


def main(args=None):
    rclpy.init(args=args)
    node = QRCameraViewer()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
