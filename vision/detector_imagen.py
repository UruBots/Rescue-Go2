"""
╔══════════════════════════════════════════════════════════════╗
║        DETECTOR DE IMÁGENES - YOLOv8 (best.pt)              ║
║  Seleccioná una imagen y YOLO detecta los objetos en ella.  ║
╚══════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import time

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "Pillow"], check=True)
    from PIL import Image, ImageTk, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
    from ultralytics import YOLO

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH     = Path(__file__).parent / "best.pt"
CONF_THRESHOLD = 0.45
PREVIEW_MAX    = (860, 560)   # tamaño máximo del panel de imagen

# Paleta de colores HSV para las clases
def hsv_color(class_id, n_classes):
    h = int(class_id * 180 / max(n_classes, 1)) % 180
    arr = np.array([[[h, 210, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))   # → RGB para PIL

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
model = None
class_names = {}

def load_model():
    global model, class_names
    if not MODEL_PATH.exists():
        messagebox.showerror("Error", f"No se encontró el modelo:\n{MODEL_PATH}")
        return False
    model = YOLO(str(MODEL_PATH))
    class_names = model.names
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
class YoloImageApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Detector de Imágenes · YOLOv8")
        self.root.configure(bg="#0f1117")
        self.root.resizable(True, True)
        self.root.minsize(980, 700)

        self.current_image_path = None
        self.result_image_pil   = None
        self.conf_var           = tk.DoubleVar(value=CONF_THRESHOLD)
        self.status_var         = tk.StringVar(value="Esperando imagen...")
        self.detections         = []

        self._build_ui()
        self._load_model_async()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # ── Título ──────────────────────────────────────────────────────────
        header = tk.Frame(root, bg="#0f1117")
        header.pack(fill="x", padx=20, pady=(18, 0))

        tk.Label(header, text="🔍 Detector YOLOv8",
                 font=("Segoe UI", 22, "bold"),
                 fg="#e2e8f0", bg="#0f1117").pack(side="left")

        tk.Label(header, text="best.pt",
                 font=("Segoe UI", 12),
                 fg="#64748b", bg="#0f1117").pack(side="left", padx=(10, 0), pady=(6, 0))

        # ── Panel principal (imagen + sidebar) ──────────────────────────────
        body = tk.Frame(root, bg="#0f1117")
        body.pack(fill="both", expand=True, padx=20, pady=14)

        # Área de imagen
        img_frame = tk.Frame(body, bg="#1e2130", bd=0,
                             highlightthickness=2, highlightbackground="#2d3148")
        img_frame.pack(side="left", fill="both", expand=True)

        # Zona de drop / placeholder
        self.drop_label = tk.Label(
            img_frame,
            text="📂  Hacé clic en 'Abrir imagen' o arrastrá\nuna foto aquí",
            font=("Segoe UI", 14), fg="#4a5568", bg="#1e2130",
            cursor="hand2"
        )
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")
        self.drop_label.bind("<Button-1>", lambda e: self.open_image())

        self.img_label = tk.Label(img_frame, bg="#1e2130")
        self.img_label.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Sidebar ─────────────────────────────────────────────────────────
        sidebar = tk.Frame(body, bg="#0f1117", width=270)
        sidebar.pack(side="right", fill="y", padx=(14, 0))
        sidebar.pack_propagate(False)

        # Botón abrir
        self._btn(sidebar, "📂  Abrir imagen",
                  "#4f46e5", "#6366f1", self.open_image).pack(fill="x", pady=(0, 8))

        # Botón analizar
        self.btn_analyze = self._btn(sidebar, "⚡  Analizar",
                                     "#059669", "#10b981", self.analyze,
                                     state="disabled")
        self.btn_analyze.pack(fill="x", pady=(0, 8))

        # Botón guardar resultado
        self.btn_save = self._btn(sidebar, "💾  Guardar resultado",
                                  "#0369a1", "#0ea5e9", self.save_result,
                                  state="disabled")
        self.btn_save.pack(fill="x", pady=(0, 18))

        # Umbral de confianza
        conf_card = tk.Frame(sidebar, bg="#1e2130",
                             highlightthickness=1, highlightbackground="#2d3148")
        conf_card.pack(fill="x", pady=(0, 10))

        tk.Label(conf_card, text="Umbral de confianza",
                 font=("Segoe UI", 10, "bold"),
                 fg="#94a3b8", bg="#1e2130").pack(anchor="w", padx=12, pady=(10, 2))

        self.conf_label = tk.Label(conf_card,
                                   text=f"{self.conf_var.get():.0%}",
                                   font=("Segoe UI", 20, "bold"),
                                   fg="#6366f1", bg="#1e2130")
        self.conf_label.pack(anchor="w", padx=12)

        slider = ttk.Scale(conf_card, from_=0.05, to=0.95,
                           orient="horizontal", variable=self.conf_var,
                           command=self._on_conf_change)
        slider.pack(fill="x", padx=12, pady=(2, 10))

        # Resultados
        res_card = tk.Frame(sidebar, bg="#1e2130",
                            highlightthickness=1, highlightbackground="#2d3148")
        res_card.pack(fill="both", expand=True, pady=(0, 10))

        tk.Label(res_card, text="Detecciones",
                 font=("Segoe UI", 10, "bold"),
                 fg="#94a3b8", bg="#1e2130").pack(anchor="w", padx=12, pady=(10, 6))

        # Lista scrollable
        list_frame = tk.Frame(res_card, bg="#1e2130")
        list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        scrollbar = tk.Scrollbar(list_frame, bg="#1e2130", troughcolor="#1e2130")
        scrollbar.pack(side="right", fill="y")

        self.result_list = tk.Listbox(
            list_frame,
            font=("Segoe UI", 10),
            bg="#161824", fg="#e2e8f0",
            selectbackground="#4f46e5",
            selectforeground="white",
            bd=0, highlightthickness=0,
            yscrollcommand=scrollbar.set,
            activestyle="none"
        )
        self.result_list.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.result_list.yview)

        # Status bar
        status_bar = tk.Frame(root, bg="#0a0c12", height=28)
        status_bar.pack(fill="x", side="bottom")

        self.model_dot = tk.Label(status_bar, text="●", fg="#ef4444",
                                  bg="#0a0c12", font=("Segoe UI", 9))
        self.model_dot.pack(side="left", padx=(10, 2), pady=5)

        self.model_label = tk.Label(status_bar, text="Cargando modelo...",
                                    fg="#64748b", bg="#0a0c12",
                                    font=("Segoe UI", 9))
        self.model_label.pack(side="left")

        tk.Label(status_bar, textvariable=self.status_var,
                 fg="#64748b", bg="#0a0c12",
                 font=("Segoe UI", 9)).pack(side="right", padx=10)

        # Estilo del slider
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScale",
                        troughcolor="#2d3148",
                        background="#4f46e5",
                        sliderthickness=16)

    def _btn(self, parent, text, bg, hover_bg, cmd, state="normal"):
        btn = tk.Button(
            parent, text=text,
            font=("Segoe UI", 11, "bold"),
            bg=bg, fg="white",
            activebackground=hover_bg, activeforeground="white",
            bd=0, padx=10, pady=10,
            cursor="hand2" if state == "normal" else "arrow",
            relief="flat", state=state,
            command=cmd
        )
        def on_enter(e): btn.config(bg=hover_bg)
        def on_leave(e): btn.config(bg=bg)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    # ── Lógica ──────────────────────────────────────────────────────────────
    def _load_model_async(self):
        def task():
            ok = load_model()
            if ok:
                self.root.after(0, self._on_model_ready)
            else:
                self.root.after(0, lambda: self.model_label.config(
                    text="❌ Error al cargar modelo"))
        threading.Thread(target=task, daemon=True).start()

    def _on_model_ready(self):
        n = len(class_names)
        classes = ", ".join(list(class_names.values())[:6])
        more    = f" (+{n-6} más)" if n > 6 else ""
        self.model_dot.config(fg="#22c55e")
        self.model_label.config(
            text=f"Modelo listo  ·  {n} clase{'s' if n>1 else ''}:  {classes}{more}",
            fg="#22c55e"
        )

    def _on_conf_change(self, val):
        self.conf_label.config(text=f"{float(val):.0%}")

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.gif"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not path:
            return
        self.current_image_path = path
        self.result_image_pil   = None
        self.result_list.delete(0, "end")
        self.btn_save.config(state="disabled")

        img = Image.open(path)
        self._show_image(img)

        name = Path(path).name
        self.status_var.set(f"📄  {name}  ({img.width}×{img.height})")
        self.btn_analyze.config(state="normal",
                                cursor="hand2", bg="#059669")
        self.drop_label.place_forget()

    def _show_image(self, pil_img: Image.Image):
        """Escala la imagen y la muestra en el label."""
        w, h = pil_img.size
        mw, mh = PREVIEW_MAX
        scale = min(mw / w, mh / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized      = pil_img.resize((new_w, new_h), Image.LANCZOS)
        tk_img       = ImageTk.PhotoImage(resized)
        self.img_label.config(image=tk_img)
        self.img_label.image = tk_img   # evitar GC

    def analyze(self):
        if not self.current_image_path or model is None:
            return
        self.btn_analyze.config(state="disabled", text="⏳  Analizando...")
        self.status_var.set("Ejecutando YOLO...")
        self.result_list.delete(0, "end")

        def task():
            t0 = time.time()
            results = model.predict(
                source  = self.current_image_path,
                conf    = self.conf_var.get(),
                verbose = False
            )
            elapsed = time.time() - t0
            self.root.after(0, lambda: self._on_results(results, elapsed))

        threading.Thread(target=task, daemon=True).start()

    def _on_results(self, results, elapsed):
        # Dibujar sobre imagen original
        original = Image.open(self.current_image_path).convert("RGB")
        draw     = ImageDraw.Draw(original, "RGBA")
        n        = len(class_names)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                label   = class_names.get(cls_id, str(cls_id))
                color   = hsv_color(cls_id, n)

                # Caja con fondo semitransparente
                draw.rectangle([(x1, y1), (x2, y2)],
                               outline=color + (255,), width=3)
                draw.rectangle([(x1, y1-1), (x2, y1-1)],
                               fill=color + (160,))

                # Texto
                text = f"{label}  {conf:.0%}"
                try:
                    font = ImageFont.truetype("arial.ttf", 18)
                except:
                    font = ImageFont.load_default()

                bbox_t = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]

                tag_y1 = max(y1 - th - 8, 0)
                draw.rectangle(
                    [(x1, tag_y1), (x1 + tw + 10, y1)],
                    fill=color + (230,)
                )
                brightness = sum(color) / 3
                txt_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)
                draw.text((x1 + 5, tag_y1 + 2), text,
                          fill=txt_color, font=font)

                detections.append((label, conf))

        self.result_image_pil = original
        self._show_image(original)

        # Llenar lista
        self.result_list.delete(0, "end")
        if detections:
            counts = {}
            for lbl, cf in detections:
                counts.setdefault(lbl, []).append(cf)

            for lbl, confs in sorted(counts.items()):
                avg  = sum(confs) / len(confs)
                line = f"  {'●'}  {lbl}   ×{len(confs)}   ({avg:.0%})"
                self.result_list.insert("end", line)
        else:
            self.result_list.insert("end", "  ⚠️  Sin detecciones")

        total = len(detections)
        self.status_var.set(
            f"✅  {total} detección{'es' if total!=1 else ''} · {elapsed*1000:.0f} ms"
        )
        self.btn_analyze.config(state="normal", text="⚡  Analizar", bg="#059669")
        self.btn_save.config(state="normal", cursor="hand2")

    def save_result(self):
        if self.result_image_pil is None:
            return
        default = Path(self.current_image_path).stem + "_resultado.jpg"
        path = filedialog.asksaveasfilename(
            title="Guardar resultado",
            initialfile=default,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Todos", "*.*")]
        )
        if path:
            self.result_image_pil.save(path, quality=95)
            self.status_var.set(f"💾  Guardado: {Path(path).name}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = YoloImageApp(root)
    root.mainloop()
