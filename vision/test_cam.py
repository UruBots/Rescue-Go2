import cv2
import sys

def test_camera(index):
    print(f"Testing camera {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Failed to open camera {index}")
        return False
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame from camera {index}")
        return False
    
    print(f"Success! Frame shape: {frame.shape}")
    cap.release()
    return True

if __name__ == "__main__":
    success = False
    for i in range(4):
        if test_camera(i):
            success = True
    sys.exit(0 if success else 1)
