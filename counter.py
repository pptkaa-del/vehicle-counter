"""Логика подсчёта транспорта по пересечению линии.

Модуль не зависит от нейросети и OpenCV, поэтому его легко тестировать.
"""
from collections import Counter

# Классы COCO, которые распознаёт YOLO
CLASS_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

LABELS_RU = {
    "car": "Легковые",
    "truck": "Грузовики",
    "bus": "Автобусы",
    "motorcycle": "Мотоциклы",
}

# Порядок вывода в интерфейсе
ORDER = ["car", "truck", "bus", "motorcycle"]


def _orient(a, b, c):
    """Знак векторного произведения: с какой стороны от прямой AB лежит точка C."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, q1, q2):
    """Пересекает ли отрезок движения p1->p2 отрезок линии подсчёта q1-q2."""
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    # точки движения по разные стороны линии, концы линии по разные стороны движения
    return (d1 > 0) != (d2 > 0) and d3 * d4 <= 0


class VehicleCounter:
    """Считает каждый трек (ID объекта) один раз при пересечении линии.

    Класс объекта определяется голосованием по всем кадрам трека,
    чтобы случайное «мигание» truck/car не портило статистику.
    """

    def __init__(self, line=None, max_age=150):
        self.line = line          # ((x1, y1), (x2, y2)) в координатах кадра
        self.max_age = max_age    # через сколько кадров забывать пропавший трек
        self.tracks = {}
        self.frame_idx = 0
        self.reset()

    def reset(self):
        # counts[класс] = [направление 1, направление 2]
        self.counts = {name: [0, 0] for name in CLASS_MAP.values()}
        self.counted_ids = set()

    def set_line(self, line):
        self.line = line
        # после смены линии предыдущие позиции не годятся
        for t in self.tracks.values():
            t["prev"] = None

    def total(self, name=None):
        if name:
            return sum(self.counts[name])
        return sum(sum(v) for v in self.counts.values())

    def track_class(self, tid):
        t = self.tracks.get(tid)
        if not t or not t["votes"]:
            return None
        return t["votes"].most_common(1)[0][0]

    def update(self, detections):
        """detections: список (track_id, class_id, cx, cy, conf).

        Возвращает список новых событий: (track_id, класс, направление 0/1).
        """
        self.frame_idx += 1
        events = []
        for tid, cls_id, cx, cy, conf in detections:
            name = CLASS_MAP.get(int(cls_id))
            if name is None:
                continue
            t = self.tracks.get(tid)
            if t is None:
                t = {"prev": None, "votes": Counter(), "last": 0}
                self.tracks[tid] = t
            t["votes"][name] += float(conf)
            p = (float(cx), float(cy))
            if self.line and t["prev"] is not None and tid not in self.counted_ids:
                a, b = self.line
                if segments_intersect(t["prev"], p, a, b):
                    cls = self.track_class(tid)
                    direction = 0 if _orient(a, b, p) > 0 else 1
                    self.counts[cls][direction] += 1
                    self.counted_ids.add(tid)
                    events.append((tid, cls, direction))
            t["prev"] = p
            t["last"] = self.frame_idx

        # удаляем давно пропавшие треки
        stale = [tid for tid, t in self.tracks.items()
                 if self.frame_idx - t["last"] > self.max_age]
        for tid in stale:
            del self.tracks[tid]
            self.counted_ids.discard(tid)
        return events
