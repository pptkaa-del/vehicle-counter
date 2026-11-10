"""Подсчёт транспорта с камеры видеонаблюдения (Windows).

Источник: RTSP/HTTP-поток IP-камеры, USB-камера (номер 0, 1, ...) или видеофайл.
Распознавание: YOLO (ultralytics) + трекер ByteTrack.
Подсчёт: каждый объект считается один раз при пересечении заданной линии.
"""
import os
import sys

# В собранном .exe без консоли stdout/stderr = None, а ultralytics пишет в лог
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# RTSP через TCP — меньше «битых» кадров на нестабильной сети
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import csv
import datetime
import json
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from counter import VehicleCounter, CLASS_MAP, LABELS_RU, ORDER

if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
LOG_DIR = os.path.join(APP_DIR, "logs")

# Цвета рамок (BGR)
COLORS = {
    "car": (60, 200, 60),
    "truck": (0, 140, 255),
    "bus": (255, 90, 30),
    "motorcycle": (200, 60, 200),
}
# Подписи на видео (OpenCV не умеет кириллицу)
SHORT = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "moto"}

MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt"]

DEFAULT_CONFIG = {
    "source": "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
    "model": "yolo11s.pt",
    "conf": 0.35,
    "imgsz": 640,
    "line": None,
    "count_moto": False,
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def parse_source(text):
    text = text.strip().strip('"')
    if text.isdigit():
        return int(text), True
    live = text.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))
    return text, live


# --------------------------------------------------------------------------- #
#  Захват кадров
# --------------------------------------------------------------------------- #
class FrameGrabber(threading.Thread):
    """Читает кадры в отдельном потоке.

    Для живой камеры хранится только последний кадр (нет накопления задержки),
    при обрыве связи — автоматическое переподключение.
    Для видеофайла обрабатывается каждый кадр.
    """

    def __init__(self, source, live, on_status):
        super().__init__(daemon=True)
        self.source = source
        self.live = live
        self.on_status = on_status
        self.q = queue.Queue(maxsize=1 if live else 8)
        self.running = True
        self.finished = False

    def _open(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def run(self):
        cap = None
        try:
            while self.running:
                if cap is None or not cap.isOpened():
                    self.on_status("Подключение к источнику...")
                    cap = self._open()
                    if not cap.isOpened():
                        if not self.live:
                            self.on_status("Не удалось открыть файл/камеру")
                            break
                        self.on_status("Нет связи с камерой, повтор через 3 с...")
                        time.sleep(3)
                        continue
                    self.on_status("Подключено")
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not self.live:
                        self.on_status("Видео закончилось")
                        break
                    self.on_status("Поток прерван, переподключение...")
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue
                if self.live:
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                    self.q.put(frame)
                else:
                    while self.running:
                        try:
                            self.q.put(frame, timeout=0.5)
                            break
                        except queue.Full:
                            pass
        finally:
            if cap is not None:
                cap.release()
            self.finished = True

    def stop(self):
        self.running = False


# --------------------------------------------------------------------------- #
#  Распознавание и подсчёт
# --------------------------------------------------------------------------- #
class Processor(threading.Thread):
    def __init__(self, app, source, live, model_path, conf, imgsz):
        super().__init__(daemon=True)
        self.app = app
        self.source = source
        self.live = live
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.running = True
        self.grabber = None

    def stop(self):
        self.running = False
        if self.grabber:
            self.grabber.stop()

    def run(self):
        app = self.app
        try:
            app.set_status("Загрузка нейросети (при первом запуске — скачивание)...")
            import torch
            import torchvision  # noqa: F401  регистрирует операторы (nms) в собранном exe
            from ultralytics import YOLO

            device = 0 if torch.cuda.is_available() else "cpu"
            model_path = self.model_path
            local = os.path.join(APP_DIR, model_path)
            if os.path.exists(local):
                model_path = local
            else:  # скачается в папку программы
                os.chdir(APP_DIR)
            model = YOLO(model_path)
            dev_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
            app.set_device(dev_name)
        except Exception as e:
            app.set_status(f"Ошибка загрузки модели: {e}")
            app.root.after(0, app.on_processor_stopped)
            return

        self.grabber = FrameGrabber(self.source, self.live, app.set_status)
        self.grabber.start()

        classes = list(CLASS_MAP.keys())
        fps_t = time.time()
        fps_n = 0
        fps = 0.0
        try:
            while self.running:
                try:
                    frame = self.grabber.q.get(timeout=0.5)
                except queue.Empty:
                    if self.grabber.finished and self.grabber.q.empty():
                        break
                    continue

                h, w = frame.shape[:2]
                app.frame_size = (w, h)
                with app.lock:
                    if app.counter.line is None:
                        app.counter.set_line(((int(w * 0.05), int(h * 0.6)),
                                              (int(w * 0.95), int(h * 0.6))))
                        app.cfg["line"] = [list(p) for p in app.counter.line]

                results = model.track(
                    frame, persist=True, classes=classes, conf=self.conf,
                    imgsz=self.imgsz, tracker="bytetrack.yaml",
                    device=device, verbose=False,
                )
                r = results[0]
                dets, boxes = [], []
                if r.boxes is not None and r.boxes.id is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.int().cpu().tolist()
                    cls = r.boxes.cls.int().cpu().tolist()
                    confs = r.boxes.conf.cpu().tolist()
                    for (x1, y1, x2, y2), tid, c, cf in zip(xyxy, ids, cls, confs):
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        dets.append((tid, c, cx, cy, cf))
                        boxes.append((int(x1), int(y1), int(x2), int(y2), tid))

                with app.lock:
                    events = app.counter.update(dets)
                    line = app.counter.line
                    counted = set(app.counter.counted_ids)
                    labels = {tid: app.counter.track_class(tid) for *_, tid in boxes}

                for ev in events:
                    app.log_event(*ev)

                # --- отрисовка ---
                if line:
                    cv2.line(frame, tuple(map(int, line[0])), tuple(map(int, line[1])),
                             (0, 255, 255), 3)
                    cv2.circle(frame, tuple(map(int, line[0])), 6, (0, 255, 255), -1)
                    cv2.circle(frame, tuple(map(int, line[1])), 6, (0, 255, 255), -1)
                for x1, y1, x2, y2, tid in boxes:
                    name = labels.get(tid) or "car"
                    color = COLORS[name]
                    thick = 3 if tid in counted else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                    text = f"{SHORT[name]} #{tid}" + (" +" if tid in counted else "")
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 1, cv2.LINE_AA)

                fps_n += 1
                now = time.time()
                if now - fps_t >= 1.0:
                    fps = fps_n / (now - fps_t)
                    fps_t, fps_n = now, 0
                cv2.putText(frame, f"FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)

                try:
                    app.out_q.get_nowait()
                except queue.Empty:
                    pass
                app.out_q.put(frame)
        except Exception as e:
            app.set_status(f"Ошибка обработки: {e}")
        finally:
            self.grabber.stop()
            app.root.after(0, app.on_processor_stopped)


# --------------------------------------------------------------------------- #
#  Интерфейс
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.lock = threading.Lock()
        line = self.cfg.get("line")
        self.counter = VehicleCounter(line=tuple(map(tuple, line)) if line else None)
        self.out_q = queue.Queue(maxsize=1)
        self.processor = None
        self.frame_size = None
        self.disp = (1.0, 0, 0)  # масштаб, смещение x, смещение y
        self.line_points = []
        self.picking_line = False
        self.session_start = datetime.datetime.now()
        self._photo = None

        root.title("Подсчёт транспорта")
        root.geometry("1280x760")
        root.minsize(900, 560)
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.poll()

    # ---------- построение окна ----------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("vista" if os.name == "nt" else "clam")
        except tk.TclError:
            pass
        style.configure("Big.TLabel", font=("Segoe UI", 26, "bold"))
        style.configure("Name.TLabel", font=("Segoe UI", 12))
        style.configure("Dir.TLabel", font=("Segoe UI", 9), foreground="#666")

        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Камера / источник:").pack(side=tk.LEFT)
        self.src_var = tk.StringVar(value=self.cfg["source"])
        ttk.Entry(top, textvariable=self.src_var, width=60).pack(side=tk.LEFT, padx=4,
                                                                 fill=tk.X, expand=True)
        ttk.Button(top, text="Файл...", command=self.pick_file).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="Модель:").pack(side=tk.LEFT, padx=(8, 2))
        self.model_var = tk.StringVar(value=self.cfg["model"])
        ttk.Combobox(top, textvariable=self.model_var, values=MODELS, width=12).pack(side=tk.LEFT)
        ttk.Label(top, text="Порог:").pack(side=tk.LEFT, padx=(8, 2))
        self.conf_var = tk.DoubleVar(value=self.cfg["conf"])
        ttk.Spinbox(top, from_=0.1, to=0.9, increment=0.05, textvariable=self.conf_var,
                    width=5).pack(side=tk.LEFT)
        self.start_btn = ttk.Button(top, text="▶ Старт", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=(10, 2))
        self.stop_btn = ttk.Button(top, text="■ Стоп", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.video = tk.Label(body, bg="#111", fg="#aaa", cursor="crosshair",
                              text="Укажите адрес камеры и нажмите «Старт»",
                              font=("Segoe UI", 14))
        self.video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video.bind("<Button-1>", self.on_video_click)

        side = ttk.Frame(body, padding=10, width=270)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        ttk.Label(side, text="Подсчёт", font=("Segoe UI", 16, "bold")).pack(anchor=tk.W)
        self.count_labels = {}
        self.dir_labels = {}
        self.moto_var = tk.BooleanVar(value=self.cfg.get("count_moto", False))
        for name in ORDER:
            fr = ttk.Frame(side, padding=(0, 6))
            fr.pack(fill=tk.X)
            if name == "motorcycle":
                self.moto_frame = fr
            sw = tk.Canvas(fr, width=14, height=40, highlightthickness=0)
            b, g, r = COLORS[name]
            sw.create_rectangle(0, 0, 14, 40, fill=f"#{r:02x}{g:02x}{b:02x}", width=0)
            sw.pack(side=tk.LEFT, padx=(0, 8))
            info = ttk.Frame(fr)
            info.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(info, text=LABELS_RU[name], style="Name.TLabel").pack(anchor=tk.W)
            d = ttk.Label(info, text="↓ 0   ↑ 0", style="Dir.TLabel")
            d.pack(anchor=tk.W)
            lbl = ttk.Label(fr, text="0", style="Big.TLabel")
            lbl.pack(side=tk.RIGHT)
            self.count_labels[name] = lbl
            self.dir_labels[name] = d

        self.sep_counts = ttk.Separator(side)
        self.sep_counts.pack(fill=tk.X, pady=6)
        fr = ttk.Frame(side)
        fr.pack(fill=tk.X)
        ttk.Label(fr, text="Всего", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        self.total_label = ttk.Label(fr, text="0", style="Big.TLabel")
        self.total_label.pack(side=tk.RIGHT)

        ttk.Checkbutton(side, text="Показывать мотоциклы", variable=self.moto_var,
                        command=self.toggle_moto).pack(anchor=tk.W, pady=(8, 0))

        ttk.Separator(side).pack(fill=tk.X, pady=8)
        ttk.Button(side, text="Задать линию подсчёта", command=self.begin_line).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Сбросить счётчики", command=self.reset_counts).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Экспорт итогов в CSV...", command=self.export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Открыть папку журналов", command=self.open_logs).pack(fill=tk.X, pady=2)

        self.device_var = tk.StringVar(value="Устройство: —")
        ttk.Label(side, textvariable=self.device_var, style="Dir.TLabel",
                  wraplength=240).pack(side=tk.BOTTOM, anchor=tk.W)

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(6, 2)).pack(side=tk.BOTTOM, fill=tk.X)
        self.toggle_moto()

    # ---------- потокобезопасные обновления ----------
    def set_status(self, text):
        self.root.after(0, self.status_var.set, text)

    def set_device(self, name):
        self.root.after(0, self.device_var.set, f"Устройство: {name}")

    # ---------- действия ----------
    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Видеофайл",
            filetypes=[("Видео", "*.mp4 *.avi *.mkv *.mov *.ts *.264"), ("Все файлы", "*.*")])
        if path:
            self.src_var.set(path)

    def _save_cfg(self):
        self.cfg["source"] = self.src_var.get().strip()
        self.cfg["model"] = self.model_var.get().strip() or DEFAULT_CONFIG["model"]
        try:
            self.cfg["conf"] = float(self.conf_var.get())
        except (tk.TclError, ValueError):
            self.cfg["conf"] = DEFAULT_CONFIG["conf"]
        self.cfg["count_moto"] = bool(self.moto_var.get())
        if self.counter.line:
            self.cfg["line"] = [list(map(int, p)) for p in self.counter.line]
        save_config(self.cfg)

    def start(self):
        if self.processor:
            return
        self._save_cfg()
        source, live = parse_source(self.cfg["source"])
        if not live and not isinstance(source, int) and not os.path.exists(source):
            messagebox.showerror("Ошибка", f"Файл не найден:\n{source}")
            return
        self.session_start = datetime.datetime.now()
        with self.lock:
            self.counter.tracks.clear()
        self.processor = Processor(self, source, live, self.cfg["model"],
                                   self.cfg["conf"], self.cfg["imgsz"])
        self.processor.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        if self.processor:
            self.processor.stop()
            self.set_status("Остановка...")

    def on_processor_stopped(self):
        self.processor = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def toggle_moto(self):
        if self.moto_var.get():
            self.moto_frame.pack(fill=tk.X, before=self.sep_counts)
        else:
            self.moto_frame.pack_forget()
        self.update_counts()

    def begin_line(self):
        if not self.frame_size:
            messagebox.showinfo("Линия", "Сначала запустите видео, затем задайте линию.")
            return
        self.picking_line = True
        self.line_points = []
        self.status_var.set("Кликните по видео ДВЕ точки — начало и конец линии подсчёта")

    def on_video_click(self, event):
        if not self.picking_line or not self.frame_size:
            return
        scale, ox, oy = self.disp
        fx = (event.x - ox) / scale
        fy = (event.y - oy) / scale
        w, h = self.frame_size
        fx = max(0, min(w - 1, fx))
        fy = max(0, min(h - 1, fy))
        self.line_points.append((int(fx), int(fy)))
        if len(self.line_points) == 1:
            self.status_var.set("Теперь кликните вторую точку линии")
        else:
            with self.lock:
                self.counter.set_line(tuple(self.line_points))
            self.picking_line = False
            self._save_cfg()
            self.status_var.set("Линия подсчёта сохранена")

    def reset_counts(self):
        if not messagebox.askyesno("Сброс", "Обнулить все счётчики?"):
            return
        with self.lock:
            self.counter.reset()
        self.session_start = datetime.datetime.now()
        self.update_counts()

    def log_event(self, tid, cls, direction):
        """Запись каждого проезда в журнал logs/проезды_ГГГГ-ММ-ДД.csv"""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            now = datetime.datetime.now()
            path = os.path.join(LOG_DIR, f"проезды_{now:%Y-%m-%d}.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f, delimiter=";")
                if new:
                    wr.writerow(["Дата", "Время", "ID", "Тип", "Направление"])
                wr.writerow([f"{now:%d.%m.%Y}", f"{now:%H:%M:%S}", tid, LABELS_RU[cls],
                             "↓ (1)" if direction == 0 else "↑ (2)"])
        except Exception as e:
            self.set_status(f"Ошибка записи журнала: {e}")

    def export_csv(self):
        now = datetime.datetime.now()
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=f"итоги_{now:%Y-%m-%d_%H-%M}.csv",
            filetypes=[("CSV (Excel)", "*.csv")])
        if not path:
            return
        with self.lock:
            counts = {k: list(v) for k, v in self.counter.counts.items()}
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f, delimiter=";")
            wr.writerow(["Период", f"{self.session_start:%d.%m.%Y %H:%M:%S} — {now:%d.%m.%Y %H:%M:%S}"])
            wr.writerow([])
            wr.writerow(["Тип", "Направление 1 (↓)", "Направление 2 (↑)", "Всего"])
            for name in ORDER:
                d1, d2 = counts[name]
                wr.writerow([LABELS_RU[name], d1, d2, d1 + d2])
            wr.writerow(["Итого", sum(c[0] for c in counts.values()),
                         sum(c[1] for c in counts.values()),
                         sum(sum(c) for c in counts.values())])
        self.status_var.set(f"Сохранено: {path}")

    def open_logs(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        if os.name == "nt":
            os.startfile(LOG_DIR)
        else:
            messagebox.showinfo("Журналы", LOG_DIR)

    # ---------- обновление экрана ----------
    def update_counts(self):
        with self.lock:
            counts = {k: list(v) for k, v in self.counter.counts.items()}
        total = 0
        for name in ORDER:
            d1, d2 = counts[name]
            self.count_labels[name].config(text=str(d1 + d2))
            self.dir_labels[name].config(text=f"↓ {d1}   ↑ {d2}")
            if name != "motorcycle" or self.moto_var.get():
                total += d1 + d2
        self.total_label.config(text=str(total))

    def poll(self):
        try:
            frame = self.out_q.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            if self.picking_line and len(self.line_points) == 1:
                cv2.circle(frame, self.line_points[0], 8, (0, 0, 255), -1)
            vw = max(self.video.winfo_width(), 10)
            vh = max(self.video.winfo_height(), 10)
            h, w = frame.shape[:2]
            scale = min(vw / w, vh / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            self.disp = (scale, (vw - nw) / 2, (vh - nh) / 2)
            rgb = cv2.cvtColor(cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.config(image=self._photo, text="")
            self.update_counts()
        self.root.after(15, self.poll)

    def on_close(self):
        self._save_cfg()
        if self.processor:
            self.processor.stop()
        self.root.after(200, self.root.destroy)


def selftest():
    """Проверка без окна: синтетическое видео с автобусом, проезжающим линию.

    Запуск: VehicleCounter.exe --selftest  -> код возврата 0 и selftest.log
    """
    import numpy as np
    from ultralytics.utils import ASSETS

    log_path = os.path.join(APP_DIR, "selftest.log")
    log = open(log_path, "w", encoding="utf-8")

    def say(*a):
        print(*a, file=log, flush=True)

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    W, H = 810, 900
    vid = os.path.join(APP_DIR, "selftest.mp4")
    out = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for i in range(60):
        m = np.float32([[1, 0, 0], [0, 1, -600 + i * 15]])
        out.write(cv2.warpAffine(img, m, (W, H), borderValue=(90, 90, 90)))
    out.release()

    class Root:
        def after(self, ms, fn, *a):
            fn(*a)

    class Stub:
        def __init__(self):
            self.root = Root()
            self.lock = threading.Lock()
            self.cfg = {}
            self.counter = VehicleCounter(line=((0, 450), (W, 450)))
            self.out_q = queue.Queue(maxsize=1)
            self.frame_size = None
            self.frames = 0

        def set_status(self, t):
            say("status:", t)

        def set_device(self, d):
            say("device:", d)

        def log_event(self, *e):
            say("event:", e)

        def on_processor_stopped(self):
            pass

    st = Stub()
    p = Processor(st, vid, False, "yolo11n.pt", 0.3, 640)
    p.start()
    p.join(600)
    say("counts:", st.counter.counts)
    ok = st.counter.total("bus") >= 1 and st.out_q.qsize() == 1
    say("RESULT:", "OK" if ok else "FAIL")
    log.close()
    try:
        os.remove(vid)
    except OSError:
        pass
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if os.name == "nt":
        try:  # чёткий интерфейс на мониторах с масштабированием
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
