#!/usr/bin/python3
"""
QR Code Camera Viewer for UDP stream (e.g. Raspberry Pi camera)
Usage: python3 qr_udp_viewer.py [udp_url]
Default: udp://0.0.0.0:1234
"""

import sys
import cv2
import numpy as np

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("pyzbar not found, using OpenCV QR detector (less reliable)")


def detect_qr_codes(image):
    """Detect QR codes in the image."""
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
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(gray)
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


def main():
    # URL del stream UDP
    url = sys.argv[1] if len(sys.argv) > 1 else "udp://0.0.0.0:1234"

    print(f"Connecting to {url} ...")
    print("Waiting for stream (make sure the Raspi is sending)...")
    print("Press 'q' to quit")

    # Abrir stream con baja latencia
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Could not open stream {url}")
        sys.exit(1)

    print("Stream connected!")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect QR codes
        detections = detect_qr_codes(frame)

        # Draw detections
        for det in detections:
            x, y, w, h = det['bbox']
            data = det['data']

            # Green bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Label background
            label = f'QR: {data}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x, y - th - 15), (x + tw + 10, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x + 5, y - 10), font, font_scale, (0, 0, 0), thickness)

            print(f"QR Detected: {data}")

        # Status text
        status = f'QR Codes: {len(detections)}' if detections else 'Scanning for QR codes...'
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Raspi Camera - QR Reader', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
