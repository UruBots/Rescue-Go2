#!/usr/bin/python3
"""
Combined QR + HAZMAT Detection Viewer for UDP stream
Reads stream once, detects both QR codes and HAZMAT placards.
Usage: python3 combined_viewer.py [udp_url]
Default: udp://0.0.0.0:1234
"""

import sys
import subprocess
import cv2
import numpy as np
import multiprocessing as mp
import time

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("pyzbar not found, using OpenCV QR detector")

MODEL_PATH = "best_hazmat.pt"
WIDTH = 640
HEIGHT = 480


def yolo_worker(frame_queue, result_queue, model_path):
    """Separate process for YOLO inference."""
    from ultralytics import YOLO
    model = YOLO(model_path)

    # Warmup with tiny image for speed
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model(dummy, verbose=False, conf=0.5)
    result_queue.put(("ready", None))

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except Exception:
            continue
        if frame is None:
            break

        results = model(frame, verbose=False, conf=0.5)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                detections.append((x1, y1, x2, y2, conf, cls_id, label))

        result_queue.put(("detections", detections))


def detect_qr(image):
    """Detect QR codes in image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    qr_results = []

    if PYZBAR_AVAILABLE:
        decoded = pyzbar.decode(gray)
        for d in decoded:
            qr_results.append({
                'data': d.data.decode('utf-8'),
                'bbox': (d.rect.left, d.rect.top, d.rect.width, d.rect.height)
            })
    else:
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(gray)
        if data and bbox is not None:
            x = int(bbox[0][0][0])
            y = int(bbox[0][0][1])
            w = int(bbox[0][2][0]) - x
            h = int(bbox[0][2][1]) - y
            qr_results.append({'data': data, 'bbox': (x, y, w, h)})

    return qr_results


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "udp://0.0.0.0:1234"

    # Start YOLO process and FFmpeg in PARALLEL
    frame_queue = mp.Queue(maxsize=2)
    result_queue = mp.Queue(maxsize=2)

    print("Starting YOLO process...")
    yolo_proc = mp.Process(target=yolo_worker, args=(frame_queue, result_queue, MODEL_PATH), daemon=True)
    yolo_proc.start()

    # Start ffmpeg IMMEDIATELY (don't wait for YOLO)
    print(f"Connecting to {url} ...")
    cmd = [
        'ffmpeg', '-fflags', 'nobuffer', '-flags', 'low_delay',
        '-probesize', '4096', '-analyzeduration', '100000',
        '-i', url, '-vf', f'scale={WIDTH}:{HEIGHT}',
        '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-an', '-sn', '-',
    ]
    ffproc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               bufsize=WIDTH * HEIGHT * 3)

    frame_size = WIDTH * HEIGHT * 3
    hazmat_colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
    ]

    yolo_ready = False
    current_hazmat = []
    last_send = 0

    print("Waiting for stream... Press 'q' or ESC to quit")

    while True:
        raw = ffproc.stdout.read(frame_size)
        if len(raw) != frame_size:
            print("Stream ended")
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

        # --- Check if YOLO is ready (non-blocking) ---
        if not yolo_ready:
            try:
                msg_type, _ = result_queue.get_nowait()
                if msg_type == "ready":
                    yolo_ready = True
                    print("YOLO ready!")
            except Exception:
                pass

        # --- QR Detection (fast, runs every frame) ---
        qr_codes = detect_qr(frame)
        for qr in qr_codes:
            x, y, w, h = qr['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            label = f'QR: {qr["data"]}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            cv2.rectangle(frame, (x, y - th - 15), (x + tw + 10, y), (255, 255, 0), -1)
            cv2.putText(frame, label, (x + 5, y - 10), font, 0.6, (0, 0, 0), 2)
            print(f"QR: {qr['data']}")

        # --- HAZMAT Detection (async via separate process) ---
        if yolo_ready:
            now = time.time()
            if now - last_send > 0.5:
                try:
                    frame_queue.put_nowait(frame.copy())
                    last_send = now
                except Exception:
                    pass

            try:
                while not result_queue.empty():
                    msg_type, data = result_queue.get_nowait()
                    if msg_type == "detections":
                        current_hazmat = data
            except Exception:
                pass

        for (x1, y1, x2, y2, conf, cls_id, label) in current_hazmat:
            color = hazmat_colors[cls_id % len(hazmat_colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f'{label} {conf:.0%}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 10), font, 0.7, (0, 0, 0), 2)

        # Status bar
        hz_count = len(current_hazmat)
        qr_count = len(qr_codes)
        status_parts = []
        if not yolo_ready:
            status_parts.append('YOLO loading...')
        if hz_count:
            status_parts.append(f'HAZMAT: {hz_count}')
        if qr_count:
            status_parts.append(f'QR: {qr_count}')
        status = ' | '.join(status_parts) if status_parts else 'Scanning...'
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('QR + HAZMAT Detector', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    frame_queue.put(None)
    ffproc.kill()
    cv2.destroyAllWindows()
    yolo_proc.join(timeout=2)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
