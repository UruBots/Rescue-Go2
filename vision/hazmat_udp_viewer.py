#!/usr/bin/python3
"""
HAZMAT Detection Viewer for UDP stream
Display runs in main process, YOLO runs in a separate process.
Usage: python3 hazmat_udp_viewer.py [udp_url]
Default: udp://0.0.0.0:1234
"""

import sys
import subprocess
import cv2
import numpy as np
import multiprocessing as mp
import time

MODEL_PATH = "best_hazmat.pt"
WIDTH = 640
HEIGHT = 480


def yolo_worker(frame_queue, result_queue, model_path):
    """Separate process that runs YOLO inference."""
    from ultralytics import YOLO
    model = YOLO(model_path)

    # Warmup
    dummy = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
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


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "udp://0.0.0.0:1234"

    # Start YOLO in separate process
    frame_queue = mp.Queue(maxsize=2)
    result_queue = mp.Queue(maxsize=2)

    print("Starting YOLO process...")
    proc = mp.Process(target=yolo_worker, args=(frame_queue, result_queue, MODEL_PATH), daemon=True)
    proc.start()

    # Wait for YOLO warmup
    print("Warming up YOLO model (this takes a few seconds)...")
    msg_type, _ = result_queue.get()
    print("YOLO ready!")

    # Start ffmpeg
    print(f"Connecting to {url} ...")
    cmd = [
        'ffmpeg',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-i', url,
        '-vf', f'scale={WIDTH}:{HEIGHT}',
        '-pix_fmt', 'bgr24',
        '-f', 'rawvideo',
        '-an', '-sn',
        '-',
    ]

    ffproc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=WIDTH * HEIGHT * 3
    )

    frame_size = WIDTH * HEIGHT * 3
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
    ]

    print("Waiting for stream... Press 'q' or ESC to quit")

    current_detections = []
    frame_count = 0
    last_send_time = 0

    while True:
        raw = ffproc.stdout.read(frame_size)
        if len(raw) != frame_size:
            print("Stream ended or error reading frames")
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
        frame_count += 1

        # Send frame to YOLO every ~0.5 seconds (non-blocking)
        now = time.time()
        if now - last_send_time > 0.5:
            try:
                frame_queue.put_nowait(frame)
                last_send_time = now
            except Exception:
                pass

        # Check for new detections (non-blocking)
        try:
            while not result_queue.empty():
                msg_type, data = result_queue.get_nowait()
                if msg_type == "detections":
                    current_detections = data
        except Exception:
            pass

        # Draw current detections on frame
        for (x1, y1, x2, y2, conf, cls_id, label) in current_detections:
            color = colors[cls_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            text = f'{label} {conf:.0%}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 10), font, 0.7, (0, 0, 0), 2)

        count = len(current_detections)
        status = f'HAZMAT: {count} detected' if count else 'Scanning...'
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('HAZMAT Detector', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Cleanup
    frame_queue.put(None)
    ffproc.kill()
    cv2.destroyAllWindows()
    proc.join(timeout=2)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
