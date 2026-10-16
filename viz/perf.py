# -*- coding: utf-8 -*-
"""
Профилировщик кадра 3D-окна: время по участкам кадра (Python) и остаток — движок
(Panda3D: отрисовка, обход сцены, ожидание вертикальной синхронизации).

    perf = FrameProfiler()
    with perf("physics"):
        ...
    perf.frame(utime.dt)            # в конце update(); dt — длительность кадра

F2 — показать/скрыть таблицу (правый верхний угол). При выходе summary() печатает
средние за прогон (без первых WARMUP с — загрузка, компиляция шейдеров).

Столбцы таблицы: среднее (скользящее, ≈ 1 с) и максимум за последние 2 с, мс.
«engine» = длительность кадра − сумма участков: всё, что вне update() владельца. Участки не вкладывать
друг в друга — иначе время посчитается дважды.
"""

import time
from collections import deque

import numpy as np
from ursina import Text, Color, camera, window

WARMUP = 2.0         # с — не учитывать в итоге (загрузка)
EMA = 0.05           # коэффициент скользящего среднего (≈ 20 кадров)
SHOW_DT = 0.25       # с — период обновления таблицы (пересборка текста дорогая)


class _Section:
    __slots__ = ("p", "name", "t0")

    def __init__(self, p, name):
        self.p, self.name = p, name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        d = self.p.cur
        d[self.name] = d.get(self.name, 0.0) + time.perf_counter() - self.t0


class FrameProfiler:

    def __init__(self, font=None):
        self.cur = {}                              # участок → с, текущий кадр
        self.ema, self.order = {}, []
        self.hist = deque(maxlen=240)              # (длительность кадра, {участок: с})
        self.total, self.n, self.t_run = {}, 0, 0.0
        self.frames_all = []                       # длительности кадров после прогрева
        self.counters = {}                         # произвольные числа (шагов физики на кадр …)
        self._t_show = 0.0
        self.txt = Text(parent=camera.ui, font=font or "VeraMono.ttf", scale=0.75,
                        origin=(0.5, 0.5), color=Color(1, 1, 0.6, 0.95), text="")
        self.txt.enabled = False

    def __call__(self, name):
        return _Section(self, name)

    def count(self, name, value):
        self.counters[name] = value

    def toggle(self):
        self.txt.enabled = not self.txt.enabled

    def frame(self, dt):
        """Конец кадра: dt — длительность кадра (utime.dt), с."""
        cur, self.cur = self.cur, {}
        cur["engine"] = max(0.0, dt - sum(cur.values()))
        for k, v in cur.items():
            if k not in self.ema:
                self.ema[k] = v
                self.order.append(k)
            self.ema[k] += EMA * (v - self.ema[k])
        self.ema["frame"] = self.ema.get("frame", dt) + EMA * (dt - self.ema.get("frame", dt))
        self.hist.append((dt, cur))
        self.t_run += dt
        if self.t_run > WARMUP:
            self.n += 1
            self.frames_all.append(dt)
            for k, v in cur.items():
                self.total[k] = self.total.get(k, 0.0) + v
        self._t_show += dt
        if self.txt.enabled and self._t_show >= SHOW_DT:
            self._t_show = 0.0
            self._show()

    def _show(self):
        win = list(self.hist)[-120:]                 # ≈ 2 с при 60 кадрах/с
        mx = lambda k: 1e3 * max(c.get(k, 0.0) for _, c in win)
        f = self.ema["frame"]
        dts = np.array([d for d, _ in win])
        low = 1.0 / np.percentile(dts, 99) if len(dts) else 0.0
        lines = [f"FPS {1 / max(f, 1e-6):5.1f}   1% low {low:5.1f}",
                 f"{'':10s}{'avg':>7s}{'max':>7s}  ms",
                 f"{'frame':10s}{1e3 * f:7.2f}{1e3 * dts.max():7.2f}"]
        for k in sorted(self.order, key=lambda k: -self.ema[k]):
            lines.append(f"{k:10s}{1e3 * self.ema[k]:7.2f}{mx(k):7.2f}")
        lines += [f"{k} {v}" for k, v in self.counters.items()]
        self.txt.text = "\n".join(lines)
        self.txt.position = (window.aspect_ratio / 2 - 0.01, 0.40)

    def summary(self) -> str:
        """Итог за прогон (после прогрева): средние, мс на кадр, и доля кадра."""
        if self.n == 0:
            return "perf: мало кадров"
        fr = np.array(self.frames_all)
        mean = fr.mean()
        out = [f"perf: {self.n} кадров, {fr.sum():.1f} с; FPS ср. {1 / mean:.1f}, "
               f"1% low {1 / np.percentile(fr, 99):.1f}; кадр ср. {1e3 * mean:.2f} мс, "
               f"макс. {1e3 * fr.max():.1f} мс"]
        for k, v in sorted(self.total.items(), key=lambda kv: -kv[1]):
            out.append(f"  {k:10s} {1e3 * v / self.n:7.2f} мс  {100 * v / fr.sum():5.1f} %")
        return "\n".join(out)
