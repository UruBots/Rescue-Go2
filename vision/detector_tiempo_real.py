"""
╔══════════════════════════════════════════════════════════════╗
║          DETECTOR EN TIEMPO REAL - YOLOv8 (best.pt)         ║
║  Controles:                                                  ║
║    Q     → Salir                                             ║
║    S     → Guardar captura de pantalla                       ║
║    +/-   → Ajustar umbral de confianza                       ║
║    SPACE → Pausar / Reanudar                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
# Solución para cuelgues de cv2.imshow en Wayland/Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
from pathlib import Path
from datetime import datetime

# Iniciar Interfaz ANTES de cargar PyTorch/YOLO para evitar deadlocks de X11/Qt en Wayland
cv2.namedWindow("Detector YOLOv8 - Tiempo Real", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detector YOLOv8 - Tiempo Real", 1280, 720)

# ── Intentar importar ultralytics ──────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("\n❌ ERROR: 'ultralytics' no está instalado.")
    print("   Ejecuta en tu terminal:\n")
    print("       pip install ultralytics\n")
    input("Presiona ENTER para salir...")
    exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  (cambia lo que necesites)
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH      = Path(__file__).parent / "best_hazmat.pt"   # modelo entrenado hazmat
CAMERA_INDEX    = 0          # índice de cámara (0 = webcam principal)
CONF_THRESHOLD  = 0.45       # umbral de confianza inicial (0.0 – 1.0)
CONF_STEP       = 0.05       # paso al usar +/-
SHOW_FPS        = True       # mostrar FPS en pantalla
SHOW_LABELS     = True       # mostrar etiquetas sobre las cajas
SHOW_CONF       = True       # mostrar porcentaje de confianza
LINE_THICKNESS  = 2          # grosor de las cajas
FONT_SCALE      = 0.65       # tamaño del texto

# Paleta de colores HSV → BGR (se genera automáticamente para cada clase)
def color_for_class(class_id: int, num_classes: int) -> tuple[int, int, int]:
    """Genera un color único y vibrante para cada clase."""
    hue   = int(class_id * 180 / max(num_classes, 1)) % 180
    color_hsv = __import__("numpy").array([[[hue, 220, 255]]], dtype=__import__("numpy").uint8)
    bgr   = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ══════════════════════════════════════════════════════════════════════════════
#  CARGAR MODELO
# ══════════════════════════════════════════════════════════════════════════════
print("\n🔄  Cargando modelo:", MODEL_PATH)
if not MODEL_PATH.exists():
    print(f"\n❌  No se encontró '{MODEL_PATH}'.")
    print("    Asegúrate de que 'best.pt' está en la misma carpeta que este script.")
    input("\nPresiona ENTER para salir...")
    exit(1)

model = YOLO(str(MODEL_PATH))
class_names  = model.names          # dict {id: nombre}
num_classes  = len(class_names)
print(f"✅  Modelo cargado  |  Clases ({num_classes}): {list(class_names.values())}\n")

# Precalcular paleta de colores
palette = {cid: color_for_class(cid, num_classes) for cid in class_names}


# ══════════════════════════════════════════════════════════════════════════════
#  ABRIR CÁMARA
# ══════════════════════════════════════════════════════════════════════════════
# Autodetectar cámara — compatible con Windows y Linux
import sys
_backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_V4L2, cv2.CAP_ANY]

cap = None
for idx in [0, 1, 2]:
    for backend in _backends:
        c = cv2.VideoCapture(idx, backend)
        if not c.isOpened():
            c.release()
            continue
        # Warmup: leer hasta 30 frames hasta que la imagen sea real (no negra)
        ok = False
        for _ in range(30):
            ret, frm = c.read()
            if ret and frm is not None and frm.mean() > 1.0:
                ok = True
                break
            time.sleep(0.05)
        if ok:
            cap = c
            print(f"📷  Cámara encontrada: índice {idx}")
            break
        c.release()
    if cap:
        break

if cap is None:
    print("❌  No se encontró ninguna cámara o solo devuelve frames negros.")
    print("    Verificá que la cámara no esté usada por otra app (Teams, Discord, etc.)")
    input("\nPresiona ENTER para salir...")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)


print("📷  Cámara abierta correctamente")
print("ℹ️   Controles: Q=Salir | S=Captura | +/-=Confianza | SPACE=Pausa\n")


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════
def draw_overlay(frame, fps: float, conf: float, paused: bool):
    """Dibuja el panel de información en la esquina superior izquierda."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Fondo semitransparente
    cv2.rectangle(overlay, (0, 0), (300, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Textos
    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
    cv2.putText(frame, f"Conf: {conf:.0%}", (120, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    if paused:
        cv2.putText(frame, "⏸ PAUSA", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)


def draw_detections(frame, results):
    """Dibuja bounding boxes, etiquetas y confianza sobre el frame."""
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = class_names.get(cls_id, str(cls_id))
            color   = palette[cls_id]

            # Caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)

            # Etiqueta con fondo
            if SHOW_LABELS:
                text = f"{label} {conf:.0%}" if SHOW_CONF else label
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                              FONT_SCALE, 1)
                bg_y1 = max(y1 - th - 8, 0)
                cv2.rectangle(frame,
                              (x1, bg_y1),
                              (x1 + tw + 6, y1),
                              color, -1)
                text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                cv2.putText(frame, text,
                            (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                            text_color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
conf_threshold = CONF_THRESHOLD
paused         = False
prev_time      = time.time()
fps            = 0.0
screenshot_dir = Path(__file__).parent / "capturas"

print("▶️  Iniciando detección... Presiona Q para salir.\n")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Error al leer frame de la cámara. Intentando reconectar...")
            cap.release()
            time.sleep(1)
            # Usar CAP_ANY en lugar de CAP_DSHOW para compatibilidad multiplataforma
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
            continue

        # ── Inferencia YOLO ──────────────────────────────────────────────────
        results = model.predict(
            source   = frame,
            conf     = conf_threshold,
            verbose  = False,
            stream   = False,
        )

        # ── Dibujar detecciones ──────────────────────────────────────────────
        draw_detections(frame, results)

        # ── Calcular FPS ─────────────────────────────────────────────────────
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

    # ── Overlay de info ──────────────────────────────────────────────────────
    draw_overlay(frame, fps, conf_threshold, paused)

    cv2.imshow("Detector YOLOv8 - Tiempo Real", frame)

    # ── Teclado ──────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n👋  Saliendo...")
        break

    elif key == ord('s') or key == ord('S'):
        screenshot_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = screenshot_dir / f"captura_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"📸  Captura guardada: {path}")

    elif key == ord('+') or key == ord('='):
        conf_threshold = min(conf_threshold + CONF_STEP, 0.95)
        print(f"🔼  Confianza → {conf_threshold:.0%}")

    elif key == ord('-') or key == ord('_'):
        conf_threshold = max(conf_threshold - CONF_STEP, 0.05)
        print(f"🔽  Confianza → {conf_threshold:.0%}")

    elif key == ord(' '):
        paused = not paused
        print("⏸  PAUSADO" if paused else "▶️  Reanudado")

    # Cerrar con la X de la ventana
    if cv2.getWindowProperty("Detector YOLOv8 - Tiempo Real",
                             cv2.WND_PROP_VISIBLE) < 1:
        break


# ══════════════════════════════════════════════════════════════════════════════
#  LIMPIEZA
# ══════════════════════════════════════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()
print("✅  Recursos liberados. ¡Hasta luego!")
