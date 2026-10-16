# -*- coding: utf-8 -*-
"""
Окно наземной станции: PFD + карта с маршрутом + приборы САУ (замена окна карты
viz/map_panel.py, решение автора 2026-10-15; стиль — по мотивам QGroundControl,
PFD — по мотивам Airbus; подписи по-английски).

Работает в ОТДЕЛЬНОМ процессе (tkinter, стандартная библиотека) рядом с 3D-окном
scenarios/GameScenario3D.py и scenarios/Bench3D.py.

Состав окна:
  сверху     — строка статуса: FLYING / ON GROUND / PAUSED / CRASH, режим, ветер, Va, h, α,
               газ, время записи;
  слева PFD  — табло режимов (FMA, 5 колонок: тяга, вертикальный и боковой каналы,
               защита по α, САУ; зелёный — активный режим, голубой — уставка);
               авиагоризонт с маркером вектора воздушной скорости (смещён на α вниз,
               на β вбок), шкала крена; лента скорости с уставкой Va_ref и полосами
               «предупреждение по α / срыв» (скорости горизонтального полёта с n = 1
               при alpha_warning и alpha_crit); лента высоты с уставкой h_ref,
               вариометр; курсовая лента: ψ, путевой угол χ (ромб), уставка χ_ref;
  в центре   — карта (вид сверху) с маршрутом: панель кнопок слева, масштаб, компас
               с курсом и ветром, полоса чисел снизу;
  справа     — рули и газ (положение после привода и команда САУ), угол атаки и
               скольжения, производные энергии TECS, журнал событий.

Мышь на карте:
  ЛКМ по пустому месту — новая точка в конец маршрута (высота — «WP ALT» на панели)
  ЛКМ по точке и тянуть — перенести точку;  ПКМ по точке — удалить
  колесо над точкой — её высота ±10 м;  колесо над картой — масштаб (вся карта … фон 1:1)
  средняя кнопка и тянуть — сдвиг карты
Кнопки: FLY — САУ ведёт по точкам с первой; STOP — навигация выкл (САУ держит курс и
высоту); CLEAR; LOOP — замкнуть маршрут (без него — кружение у последней точки);
CENTER — карту на ЛА; WP ALT ▲▼ — высота новых точек ±10 м.

Связь — две очереди multiprocessing (протокол окна карты сохранён):
    q_cmd (окно → 3D):  ("route", {"wps": [(N, E, h), ...], "loop": bool}) | ("go", None) | ("stop", None)
    q_tel (3D → окно):  dict на конец кадра — см. GameScenario3D.Game._station_telemetry
info — постоянные данные ЛА (пределы рулей, пороги α, скорости полос) — station_info().

Фон карты — PNG вида сверху (World.save_top_image()) охватом extent =
((E_мин, E_макс), (N_мин, N_макс)), м, в ZOOM_MAX раз крупнее карты в окне.

Размер окна: при запуске — номинальный, но не больше экрана; окно можно растягивать.
Всё, кроме карты, рисуется в номинальных пикселях (окно с картой фон / ZOOM_MAX) и
масштабируется коэффициентом UI = min(высота / номинальная высота, ширина / мин. ширина);
шрифты — тоже ×UI. Карта занимает остаток ширины; её масштаб — целое прореживание фона
(1 пикс окна = sub пикс фона), наименьший — вся карта в окне.
Физики и САУ модуль не знает.
"""

import math
import queue
import multiprocessing as mp
import tkinter as tk
import tkinter.font as tkfont

TICK_MS = 40                   # период обновления, мс
TRAIL_DT = 0.5                 # шаг точек следа, с
TRAIL_MAX = 2400               # точек следа (20 мин)
PICK_PX = 11                   # радиус «попадания» в точку, пикс
ZOOM_MAX = 4                   # фон — в ZOOM_MAX раз крупнее номинальной карты в окне
SIDE_W = 470                   # ширина правой колонки, номинальные пикс
TOP_H = 44                     # высота строки статуса, номинальные пикс
PFD_K = 0.78                   # ширина PFD / номинальная высота
MAP_MIN = 600                  # ширина карты, ниже которой мельчает всё окно (плашки карты не налезают), номинальные пикс
OV = ("dyn", "ovl")            # теги элементов поверх карты (масштабируются ×UI)

# цвета (тёмная тема)
BG, BAR, CARD, EDGE, DIM, TXT = "#101215", "#1c1f24", "#22262c", "#3a414b", "#8d99a6", "#e8ecf0"
GREEN, AMBER, RED, CYAN, MAG, YEL = "#3ddc84", "#ffb020", "#ff4545", "#4dd0ff", "#ff5cf0", "#ffe14d"
SKY, GND, TAPE = "#2678c2", "#7a4220", "#3a3d42"
ROUTE, WP, WP_CUR, AC, TRAIL = "#e8a33d", "#e8a33d", "#ffffff", "#ff3b30", "#ff5050"
COL = {"g": GREEN, "c": CYAN, "w": TXT, "a": AMBER, "r": RED, "d": DIM}
EV_COL = {"red": RED, "orange": AMBER, "green": GREEN, "dodgerblue": CYAN, "gray": DIM, "purple": MAG}

UI = 1.0                       # масштаб интерфейса (пересчитывается при изменении размера окна)
PT = 96 / 72                   # пикс на пункт шрифта (tk scaling, уточняется при запуске)
F = lambda size, w="normal": ("Segoe UI", -max(7, round(size * UI * PT)), w)       # size — пункты при UI = 1
FM = lambda size, w="normal": ("Consolas", -max(7, round(size * UI * PT)), w)


# ===========================================================================
# Геометрия (чистые функции)
# ===========================================================================

def clip_halfplane(poly, p0, n):
    """Часть многоугольника poly, где (p − p0)·n ≥ 0 (отсечение Сазерленда — Ходжмана)."""
    d = lambda p: (p[0] - p0[0]) * n[0] + (p[1] - p0[1]) * n[1]
    out = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        da, db = d(a), d(b)
        if da >= 0:
            out.append(a)
        if da * db < 0:
            k = da / (da - db)
            out.append((a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1])))
    return out


def clip_segment(a, b, x0, y0, x1, y1):
    """Отрезок ab, обрезанный прямоугольником (Лианг — Барски); None — снаружи."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return (a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy)


def fmt_time(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ===========================================================================
# Окно
# ===========================================================================

class Station:

    def __init__(self, root, q_cmd, q_tel, bg_path, extent, info, h_new=150.0, pos="+0+0"):
        self.root, self.q_cmd, self.q_tel, self.info = root, q_cmd, q_tel, info
        (self.e0, self.e1), (self.n0, self.n1) = extent
        self.wps = []                     # [[N, E, h], ...]
        self.trail = []                   # [(N, E), ...]
        self.flown = 0.0                  # пройденный путь по следу, м
        self.last = None
        self.drag = None                  # индекс перетаскиваемой точки
        self.sel = None                   # последняя затронутая точка (карточка справа вверху)
        self.loop = False
        self.h_new = float(h_new)
        self._t_trail = 0.0
        self._pan = None
        self.buttons = []                 # (x0, y0, x1, y1, действие) на карте
        self._fonts = {}                  # кэш шрифтов для подбора размера
        self._size, self._job = None, None

        global PT
        PT = float(root.tk.call("tk", "scaling"))
        root.title("Ground station — PFD + map")
        root.configure(bg=BG)
        self.src = tk.PhotoImage(file=bg_path)
        self.SW, self.SH = self.src.width(), self.src.height()
        self.H0 = self.SH // ZOOM_MAX                       # номинальная высота приборов и карты

        self.top = tk.Canvas(root, bg=BAR, highlightthickness=0)
        self.top.grid(row=0, column=0, columnspan=3)
        self.pfd = tk.Canvas(root, bg="black", highlightthickness=0)
        self.pfd.grid(row=1, column=0)
        self.cv = tk.Canvas(root, bg="#666", highlightthickness=0)
        self.cv.grid(row=1, column=1)
        self.side = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.side.grid(row=1, column=2)

        self.bg = self.cv.create_image(0, 0, anchor="nw")
        self.view = None
        self.sub, self.vx, self.vy = ZOOM_MAX, 0, 0  # прореживание фона и левый верхний угол вида (пикс фона)
        cv = self.cv
        cv.bind("<Button-1>", self._press)
        cv.bind("<B1-Motion>", self._motion)
        cv.bind("<ButtonRelease-1>", self._release)
        cv.bind("<Button-3>", self._delete)
        cv.bind("<MouseWheel>", self._wheel)
        cv.bind("<Button-2>", lambda ev: setattr(self, "_pan", (ev.x, ev.y)))
        cv.bind("<B2-Motion>", self._pan_move)
        # номинальный размер, но не больше экрана (запас — заголовок окна и панель задач)
        Ww = min(round(PFD_K * self.H0) + self.H0 + SIDE_W, root.winfo_screenwidth() - 16)
        Wh = min(self.H0 + TOP_H, root.winfo_screenheight() - 110)
        root.geometry(f"{Ww}x{Wh}{pos}")
        root.minsize(640, 360)
        self._layout(Ww, Wh)
        root.bind("<Configure>", self._on_configure)
        root.after(TICK_MS, self._tick)

    # --- размер окна -------------------------------------------------------------
    def _on_configure(self, ev):
        if ev.widget is self.root and (ev.width, ev.height) != self._size:
            if self._job:
                self.root.after_cancel(self._job)
            self._job = self.root.after(120, lambda: self._layout(self.root.winfo_width(),
                                                                  self.root.winfo_height()))

    def _layout(self, Ww, Wh):
        """Размеры панелей и масштаб UI под окно Ww × Wh пикс; карта — снова целиком."""
        global UI
        self._size, self._job = (Ww, Wh), None
        H0 = self.H0
        UI = min(max(min(Wh / (H0 + TOP_H), Ww / (PFD_K * H0 + SIDE_W + MAP_MIN)), 0.4), 2.0)
        top = round(TOP_H * UI)
        self.PH = self.MH = max(Wh - top, 100)
        self.PW, sw = round(PFD_K * H0 * UI), round(SIDE_W * UI)
        self.MW = max(Ww - self.PW - sw, 100)
        self.top.configure(width=Ww, height=top)
        self.pfd.configure(width=self.PW, height=self.PH)
        self.cv.configure(width=self.MW, height=self.MH)
        self.side.configure(width=sw, height=self.PH)
        fit = max(1, math.ceil(max(self.SW / self.MW, self.SH / self.MH)))
        self.levels = [fit] + [s for s in (4, 2, 1) if s < fit]       # прореживание: вся карта … 1:1
        self._set_view(fit, 0, 0)
        self._draw_all()

    # --- координаты и вид карты ------------------------------------------------
    def _k(self):
        return 1.0 / self.sub                                # пикс окна на пикс фона

    def to_px(self, n, e):
        sx = (e - self.e0) / (self.e1 - self.e0) * self.SW
        sy = (self.n1 - n) / (self.n1 - self.n0) * self.SH
        return (sx - self.vx) * self._k(), (sy - self.vy) * self._k()

    def to_ne(self, x, y):
        sx, sy = self.vx + x / self._k(), self.vy + y / self._k()
        return (self.n1 - sy / self.SH * (self.n1 - self.n0),
                self.e0 + sx / self.SW * (self.e1 - self.e0))

    def _scale(self):
        return self.SW / (self.e1 - self.e0) * self._k()     # пикс окна / м

    def _set_view(self, sub, vx, vy):
        """Прореживание sub и левый верхний угол (vx, vy) в пикселях фона; вид — в пределах
        карты, а если карта меньше окна — по центру."""
        self.sub = sub
        w, h = self.MW * sub, self.MH * sub
        place = lambda v, size, S: (S - size) // 2 if size >= S else int(min(max(v, 0), S - size))
        self.vx, self.vy = place(vx, w, self.SW), place(vy, h, self.SH)
        x0, y0 = max(self.vx, 0), max(self.vy, 0)
        x1, y1 = min(self.vx + w, self.SW), min(self.vy + h, self.SH)
        self.view = tk.PhotoImage()
        self.view.tk.call(self.view, "copy", self.src, "-from", x0, y0, x1, y1, "-subsample", sub, sub)
        self.cv.coords(self.bg, (x0 - self.vx) // sub, (y0 - self.vy) // sub)
        self.cv.itemconfigure(self.bg, image=self.view)

    def _next_sub(self, d):
        """Соседний уровень масштаба: d > 0 — крупнее."""
        i = self.levels.index(self.sub) if self.sub in self.levels else 0
        return self.levels[min(max(i + (1 if d > 0 else -1), 0), len(self.levels) - 1)]

    def _zoom_at(self, x, y, sub):
        sx, sy = self.vx + x * self.sub, self.vy + y * self.sub         # точка под курсором
        self._set_view(sub, sx - x * sub, sy - y * sub)
        self._draw_map()

    def _pan_move(self, ev):
        if self._pan:
            dx, dy = ev.x - self._pan[0], ev.y - self._pan[1]
            self._pan = (ev.x, ev.y)
            self._set_view(self.sub, self.vx - dx * self.sub, self.vy - dy * self.sub)
            self._draw_map()

    def _center_ac(self):
        if self.last:
            x, y = self.to_px(self.last["n"], self.last["e"])
            self._set_view(self.sub, self.vx + (x - self.MW / 2) * self.sub,
                           self.vy + (y - self.MH / 2) * self.sub)

    def _pick(self, x, y):
        best, k_best = PICK_PX, None
        for k, (n, e, _) in enumerate(self.wps):
            px, py = self.to_px(n, e)
            d = math.hypot(px - x, py - y)
            if d <= best:
                best, k_best = d, k
        return k_best

    # --- мышь ----------------------------------------------------------------
    def _button_at(self, x, y):
        x, y = x / UI, y / UI                                # кнопки — в номинальных пикселях
        for x0, y0, x1, y1, act in self.buttons:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return act
        return None

    def _press(self, ev):
        act = self._button_at(ev.x, ev.y)
        if act is not None:
            act()
            self._draw_map()
            return
        k = self._pick(ev.x, ev.y)
        if k is None:
            n, e = self.to_ne(ev.x, ev.y)
            self.wps.append([n, e, self.h_new])
            self.sel = len(self.wps) - 1
            self._send()
        else:
            self.drag = self.sel = k

    def _motion(self, ev):
        if self.drag is not None:
            x, y = min(max(ev.x, 0), self.MW), min(max(ev.y, 0), self.MH)
            self.wps[self.drag][:2] = self.to_ne(x, y)
            self._draw_map()

    def _release(self, _ev):
        if self.drag is not None:
            self.drag = None
            self._send()

    def _delete(self, ev):
        k = self._pick(ev.x, ev.y)
        if k is not None:
            del self.wps[k]
            self.sel = None
            self._send()

    def _wheel(self, ev):
        k = self._pick(ev.x, ev.y)
        if k is not None:
            self.wps[k][2] = min(max(self.wps[k][2] + (10 if ev.delta > 0 else -10), 20), 1000)
            self.sel = k
            self._send()
            return
        sub = self._next_sub(ev.delta)
        if sub != self.sub:
            self._zoom_at(ev.x, ev.y, sub)

    # --- кнопки ---------------------------------------------------------------
    def _clear(self):
        self.wps, self.sel = [], None
        self._send()

    def _toggle_loop(self):
        self.loop = not self.loop
        self._send()

    def _h_step(self, d):
        self.h_new = min(max(self.h_new + d, 20.0), 1000.0)

    def _zoom_btn(self, d):
        """Кнопки масштаба: вокруг ЛА, если он в окне, иначе вокруг центра карты."""
        sub = self._next_sub(d)
        if sub == self.sub:
            return
        x, y = self.MW / 2, self.MH / 2
        if self.last:
            ax, ay = self.to_px(self.last["n"], self.last["e"])
            if 0 <= ax <= self.MW and 0 <= ay <= self.MH:
                x, y = ax, ay
        self._zoom_at(x, y, sub)

    # --- связь -----------------------------------------------------------------
    def _put(self, msg):
        try:
            self.q_cmd.put_nowait(msg)
        except queue.Full:
            pass

    def _send(self):
        self._put(("route", {"wps": [tuple(w) for w in self.wps], "loop": self.loop}))
        self._draw_map()

    def _tick(self):
        parent = mp.parent_process()
        if parent is not None and not parent.is_alive():     # 3D-окно закрыто/упало
            self.root.destroy()
            return
        while True:
            try:
                m = self.q_tel.get_nowait()
            except queue.Empty:
                break
            if self.last and m["t"] < self.last["t"]:         # сброс полёта (R)
                self.trail, self.flown = [], 0.0
            if not self.trail or m["t"] - self._t_trail >= TRAIL_DT:
                if self.trail:
                    self.flown += math.hypot(m["n"] - self.trail[-1][0], m["e"] - self.trail[-1][1])
                self.trail.append((m["n"], m["e"]))
                del self.trail[:-TRAIL_MAX]
                self._t_trail = m["t"]
            self.last = m
        self._draw_all()
        self.root.after(TICK_MS, self._tick)

    # ===========================================================================
    # Рисование
    # ===========================================================================
    def _draw_all(self):
        """Строка статуса, PFD и правая колонка рисуются в номинальных пикселях и масштабируются ×UI."""
        for cv, draw in ((self.top, self._draw_top), (self.pfd, self._draw_pfd), (self.side, self._draw_side)):
            cv.delete("all")
            draw()
            cv.scale("all", 0, 0, UI, UI)
        self._draw_map()

    # --- строка статуса ----------------------------------------------------------
    def _draw_top(self):
        cv, m = self.top, self.last
        W = int(cv["width"]) / UI
        if m is None:
            cv.create_text(16, TOP_H / 2, text="WAITING FOR TELEMETRY…", anchor="w", fill=DIM, font=F(13, "bold"))
            return
        st = m["status"]
        cv.create_rectangle(0, 0, 210, TOP_H, fill={"CRASH": "#8a1f1f", "PAUSED": "#6b5a1a"}.get(st, "#2f6f3e"),
                            width=0)
        cv.create_text(16, TOP_H / 2, text=("▲  " if st == "FLYING" else "■  ") + st, anchor="w",
                       fill="white", font=F(14, "bold"))
        wind = math.hypot(m["wind_n"], m["wind_e"])
        frm = (math.degrees(math.atan2(-m["wind_e"], -m["wind_n"])) % 360) if wind > 0.2 else None
        chips = [("◆ " + m["mode"], GREEN if m["ap"] else TXT),
                 (f"WIND {wind:.1f} m/s" + (f" FROM {frm:03.0f}°" if frm is not None else ""), CYAN),
                 (f"Va {m['Va']:.1f} m/s", TXT), (f"h {m['h']:.0f} m", TXT),
                 (f"α {m['alpha']:.1f}°", [GREEN, AMBER, RED, RED][min(m['alert'], 3)]),
                 (f"THR {100 * m['thr']:.0f} %", TXT)]
        x = 232
        for s, c in chips:
            t = cv.create_text(x, TOP_H / 2, text=s, anchor="w", fill=c, font=F(12))
            b = cv.bbox(t)
            x += (b[2] - b[0]) / UI + 26                     # ширина текста — в пикселях окна
        cv.create_oval(W - 190, TOP_H / 2 - 7, W - 176, TOP_H / 2 + 7, fill=RED, width=0)
        cv.create_text(W - 168, TOP_H / 2, text=f"REC {fmt_time(m['t'])}", anchor="w", fill=TXT, font=FM(12))
        cv.create_text(W - 12, TOP_H / 2, text=self.info["name"].upper(), anchor="e", fill=DIM, font=F(10))

    # --- PFD -----------------------------------------------------------------------
    def _draw_pfd(self):
        cv, m, W, H = self.pfd, self.last, self.PW / UI, self.PH / UI
        if m is None:
            return
        self._fma(cv, m, W, H)
        ax0, ax1, ay0, ay1 = 0.20 * W, 0.79 * W, 0.16 * H, 0.78 * H
        cx, cy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        ppd = (ay1 - ay0) / 60.0                              # ±30° по вертикали
        self._attitude(cv, m, ax0, ax1, ay0, ay1, cx, cy, ppd)
        self._speed_tape(cv, m, 0.03 * W, 0.17 * W, ay0, ay1, cy)
        self._alt_tape(cv, m, 0.815 * W, 0.935 * W, ay0, ay1, cy)
        self._vario(cv, m, 0.945 * W, 0.99 * W, ay0, ay1, cy)
        self._heading_tape(cv, m, ax0, ax1, 0.82 * H, 0.885 * H, cx)
        gs = f"GS {m['Vg']:.1f} m/s     θ {m['theta']:+.1f}°     φ {m['phi']:+.1f}°"
        cv.create_text(W / 2, 0.95 * H, text=gs, fill=DIM, font=FM(11))

    def _fit(self, text, width, size):
        """Шрифт не крупнее size, в котором text укладывается в width номинальных пикс (узкие колонки FMA)."""
        for sz in range(size, 6, -1):
            key = F(sz)
            if key not in self._fonts:
                self._fonts[key] = tkfont.Font(family=key[0], size=key[1])
            if self._fonts[key].measure(text) <= width * UI:
                return key
        return F(7)

    def _fma(self, cv, m, W, H):
        fh, cw = 0.11 * H, W / 5
        for i, (a, b, ca, cb) in enumerate(m["fma"]):
            x = cw * (i + 0.5)
            cv.create_text(x, fh * 0.32, text=a, fill=COL[ca], font=self._fit(a, cw - 10, 13))
            cv.create_text(x, fh * 0.72, text=b, fill=COL[cb], font=self._fit(b, cw - 10, 12))
            if i:
                cv.create_line(cw * i, fh * 0.08, cw * i, fh * 0.92, fill="#555")
        cv.create_line(0, fh, W, fh, fill="#333")

    def _attitude(self, cv, m, ax0, ax1, ay0, ay1, cx, cy, ppd):
        ph, th = math.radians(m["phi"]), m["theta"]
        c, s = math.cos(ph), math.sin(ph)
        # горизонтная СК → экран: (x, y) ↦ C + R(x, y); R — поворот на φ (правый крен — правый конец вверх)
        R = lambda x, y: (cx + x * c + y * s, cy - x * s + y * c)
        rect = [(ax0, ay0), (ax1, ay0), (ax1, ay1), (ax0, ay1)]
        cv.create_rectangle(ax0, ay0, ax1, ay1, fill=SKY, width=0)
        p0 = R(0, th * ppd)
        gnd = clip_halfplane(rect, p0, (s, c))
        if len(gnd) > 2:
            cv.create_polygon(*[v for p in gnd for v in p], fill=GND, width=0)
        inside = lambda p: ax0 + 2 < p[0] < ax1 - 2 and ay0 + 2 < p[1] < ay1 - 2
        seg = clip_segment(R(-2000, th * ppd), R(2000, th * ppd), ax0, ay0, ax1, ay1)
        if seg:
            cv.create_line(*seg[0], *seg[1], fill="white", width=2)
        for p in range(-30, 35, 5):                           # линии тангажа
            if p == 0:
                continue
            hw = 0.11 * (ax1 - ax0) if p % 10 == 0 else 0.05 * (ax1 - ax0)
            y = (th - p) * ppd
            e0, e1 = R(-hw, y), R(hw, y)
            if inside(e0) and inside(e1):
                cv.create_line(*e0, *e1, fill="white", width=1)
                if p % 10 == 0:
                    for sx in (-1, 1):
                        q = R(sx * (hw + 14), y)
                        if inside(q):
                            cv.create_text(*q, text=str(abs(p)), fill="white", font=FM(9))
        # шкала крена (связана с горизонтом) и неподвижный указатель
        Rr = 0.42 * (ax1 - ax0)
        for a in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
            ar = math.radians(a)
            l = 12 if a % 30 == 0 else 7
            q0 = R(Rr * math.sin(ar), -Rr * math.cos(ar))
            q1 = R((Rr + l) * math.sin(ar), -(Rr + l) * math.cos(ar))
            if inside(q0) and inside(q1):
                cv.create_line(*q0, *q1, fill="white", width=2 if a == 0 else 1)
        cv.create_polygon(cx, cy - Rr + 2, cx - 8, cy - Rr + 16, cx + 8, cy - Rr + 16, fill=YEL, outline="")
        # символ ЛА (неподвижный) и маркер вектора скорости: вниз на α, вбок на β
        for sx in (-1, 1):
            cv.create_polygon(cx + sx * 26, cy - 3, cx + sx * 74, cy - 3, cx + sx * 74, cy + 3,
                              cx + sx * 32, cy + 3, cx + sx * 32, cy + 13, cx + sx * 26, cy + 13,
                              fill="black", outline=YEL, width=2)
        cv.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4, fill="black", outline=YEL, width=2)
        fx = min(max(cx + m["beta"] * ppd, ax0 + 30), ax1 - 30)
        fy = min(max(cy + m["alpha"] * ppd, ay0 + 20), ay1 - 20)
        r = 7
        cv.create_oval(fx - r, fy - r, fx + r, fy + r, outline=GREEN, width=2)
        cv.create_line(fx - r, fy, fx - 22, fy, fill=GREEN, width=2)
        cv.create_line(fx + r, fy, fx + 22, fy, fill=GREEN, width=2)
        cv.create_line(fx, fy - r, fx, fy - 16, fill=GREEN, width=2)
        cv.create_rectangle(ax0, ay0, ax1, ay1, outline="#222", width=2)

    def _speed_tape(self, cv, m, x0, x1, y0, y1, cy):
        Va, sc = m["Va"], (y1 - y0) / 16.0                    # ±8 м/с
        cv.create_rectangle(x0, y0, x1, y1, fill=TAPE, width=0)
        Y = lambda v: cy - (v - Va) * sc
        vs, vw = self.info["V_stall"], self.info["V_warn"]    # полосы: срыв, предупреждение по α
        for v0, v1, c in ((0.0, vs, RED), (vs, vw, AMBER)):
            ya, yb = max(min(Y(v1), y1), y0), max(min(Y(v0), y1), y0)
            if yb > ya:
                cv.create_rectangle(x1 - 7, ya, x1, yb, fill=c, width=0)
        for v in range(int(Va) - 9, int(Va) + 10):
            y = Y(v)
            if y0 < y < y1 and v >= 0:
                cv.create_line(x1 - 12, y, x1, y, fill="white")
                if v % 2 == 0:
                    cv.create_text((x0 + x1) / 2 - 6, y, text=str(v), fill="white", font=FM(11))
        if m["Va_ref"] is not None:
            y = min(max(Y(m["Va_ref"]), y0), y1)
            cv.create_polygon(x1, y, x1 + 14, y - 8, x1 + 14, y + 8, fill="", outline=MAG, width=2)
            cv.create_text((x0 + x1) / 2, y0 - 12, text=f"{m['Va_ref']:.1f}", fill=MAG, font=FM(11))
        cv.create_line(x0 - 4, cy, x1 + 4, cy, fill=YEL, width=3)
        cv.create_text((x0 + x1) / 2, y1 + 14, text=f"Va {Va:.1f}", fill=GREEN, font=FM(11, "bold"))

    def _alt_tape(self, cv, m, x0, x1, y0, y1, cy):
        h, sc = m["h"], (y1 - y0) / 60.0                      # ±30 м
        cv.create_rectangle(x0, y0, x1, y1, fill=TAPE, width=0)
        Y = lambda v: cy - (v - h) * sc
        for v in range(int(h) - 35, int(h) + 36):
            if v % 5:
                continue
            y = Y(v)
            if y0 < y < y1:
                cv.create_line(x0, y, x0 + (12 if v % 10 == 0 else 7), y, fill="white")
                if v % 20 == 0:
                    cv.create_text((x0 + x1) / 2 + 6, y, text=str(v), fill="white", font=FM(10))
        if m["h_ref"] is not None:
            y = min(max(Y(m["h_ref"]), y0), y1)
            cv.create_polygon(x0, y, x0 - 13, y - 8, x0 - 13, y + 8, fill=CYAN, outline="")
            cv.create_text((x0 + x1) / 2, y0 - 12, text=f"{m['h_ref']:.0f}", fill=CYAN, font=FM(11))
        cv.create_rectangle(x0 - 3, cy - 15, x1 + 3, cy + 15, fill="black", outline=YEL, width=2)
        cv.create_text((x0 + x1) / 2, cy, text=f"{h:.0f}", fill=GREEN, font=FM(14, "bold"))

    def _vario(self, cv, m, x0, x1, y0, y1, cy):
        hh = 0.42 * (y1 - y0)
        Y = lambda v: cy - math.copysign(hh * math.log1p(abs(v)) / math.log1p(6.0), v)
        cv.create_polygon(x0, cy - hh, x0 + 8, cy - hh - 12, x1, cy - hh - 12, x1, cy + hh + 12,
                          x0 + 8, cy + hh + 12, x0, cy + hh, fill=TAPE, outline="")
        for v in (-6, -2, -1, 1, 2, 6):
            cv.create_line(x0 + 2, Y(v), x0 + 9, Y(v), fill="white")
        vh = max(min(m["Vh"], 6.0), -6.0)
        cv.create_line(x1, cy, x0 + 3, Y(vh), fill=GREEN, width=3)
        cv.create_text((x0 + x1) / 2, cy - hh - 26, text=f"{m['Vh']:+.1f}", fill=GREEN, font=FM(10))

    def _heading_tape(self, cv, m, x0, x1, y0, y1, cx):
        psi, k = m["psi"], (x1 - x0) / 60.0                   # ±30°
        cv.create_rectangle(x0, y0, x1, y1, fill=TAPE, width=0)
        X = lambda a: cx + ((a - psi + 180) % 360 - 180) * k
        for a in range(int(psi) - 35, int(psi) + 36):
            if a % 5:
                continue
            x = X(a)
            if x0 < x < x1:
                cv.create_line(x, y0, x, y0 + (12 if a % 10 == 0 else 6), fill="white")
                if a % 10 == 0 and x0 + 14 < x < x1 - 14:      # подпись у края ленты не печатаем
                    a3 = a % 360
                    cv.create_text(x, y0 + 24, text={0: "N", 90: "E", 180: "S", 270: "W"}.get(a3, f"{a3 // 10:02d}"),
                                   fill="white", font=FM(11))
        cv.create_line(cx, y0 - 10, cx, y0 + 4, fill=YEL, width=3)
        xc = min(max(X(m["chi"]), x0), x1)
        cv.create_polygon(xc, y0 + 1, xc - 7, y0 + 9, xc, y0 + 17, xc + 7, y0 + 9, fill="", outline=GREEN, width=2)
        if m["chi_ref"] is not None:
            xr = min(max(X(m["chi_ref"]), x0), x1)
            cv.create_line(xr, y0 - 9, xr, y0 + 1, fill=MAG, width=4)
        cv.create_text(x0 - 8, (y0 + y1) / 2, text=f"ψ {psi:03.0f}", anchor="e", fill="white", font=FM(11))
        cv.create_text(x1 + 8, (y0 + y1) / 2, text=f"χ {m['chi']:03.0f}", anchor="w", fill=GREEN, font=FM(11))

    # --- карта -----------------------------------------------------------------------
    def _draw_map(self):
        cv, m = self.cv, self.last
        cv.delete("dyn")
        if len(self.trail) > 1:
            pts = [c for n, e in self.trail for c in self.to_px(n, e)]
            cv.create_line(*pts, fill=TRAIL, width=2, tags="dyn")
        nav = bool(m and m["nav"])
        if nav and m["circle"]:                             # дуга / окружность навигатора
            cn, ce, rho = m["circle"]
            x, y = self.to_px(cn, ce)
            r = rho * self._scale()
            cv.create_oval(x - r, y - r, x + r, y + r, outline=WP_CUR, dash=(4, 3), tags="dyn")
        P = [self.to_px(n, e) for n, e, _ in self.wps]
        if len(P) > 1:
            for w, c in ((5, "#3a2500"), (3, ROUTE)):
                cv.create_line(*[v for p in P for v in p], fill=c, width=w, tags="dyn")
            if self.loop:
                cv.create_line(*P[-1], *P[0], fill=ROUTE, width=3, dash=(6, 4), tags="dyn")
        if nav and self.wps and m["idx"] < len(self.wps) and m["mode_nav"] != "orbit":
            cv.create_line(*self.to_px(m["n"], m["e"]), *P[m["idx"]], fill="white", dash=(4, 3), tags="dyn")
        for k, ((x, y), (_, _, h)) in enumerate(zip(P, self.wps)):
            cur = nav and k == m["idx"]
            r = (11 if cur else 9) * UI
            cv.create_oval(x - r, y - r, x + r, y + r, fill=WP_CUR if cur else WP,
                           outline=ROUTE if cur else "#3a2500", width=2, tags="dyn")
            cv.create_text(x, y, text=str(k + 1), font=F(9, "bold"), fill="black", tags="dyn")
            for dx, dy, c in ((1, 1, "black"), (0, 0, "white")):
                cv.create_text(x + 13 * UI + dx, y - 12 * UI + dy, text=f"{h:.0f} m", font=F(9), fill=c,
                               anchor="w", tags="dyn")
        if m:                                               # ЛА: стрелка по курсу ψ
            x, y = self.to_px(m["n"], m["e"])
            s, c = math.sin(math.radians(m["psi"])), math.cos(math.radians(m["psi"]))
            tri = [(0, -16), (11, 11), (0, 5), (-11, 11)]
            pts = [(x + UI * (px * c - py * s), y + UI * (px * s + py * c)) for px, py in tri]
            cv.create_polygon(*[v for p in pts for v in p], fill=AC, outline="white", width=2, tags="dyn")
        self._map_overlay(cv, m)
        cv.scale("ovl", 0, 0, UI, UI)

    def _card(self, cv, x0, y0, x1, y1, tags=OV, fill=BAR):
        cv.create_rectangle(x0, y0, x1, y1, fill=fill, outline=EDGE, tags=tags)

    def _map_overlay(self, cv, m):
        """Кнопки, масштаб, карточки, компас — в номинальных пикселях (окно / UI), затем ×UI."""
        self.buttons = []
        MW, MH = self.MW / UI, self.MH / UI
        # панель кнопок слева
        items = [("✈", "FLY", lambda: self._put(("go", None)), GREEN),
                 ("■", "STOP", lambda: self._put(("stop", None)), TXT),
                 ("✕", "CLEAR", self._clear, TXT),
                 ("↻", "LOOP", self._toggle_loop, CYAN if self.loop else TXT),
                 ("⊙", "CENTER", self._center_ac, TXT)]
        bx0, by, bw, bh = 10, 10, 66, 58
        self._card(cv, bx0, by, bx0 + bw, by + bh * len(items) + 96)
        for i, (ic, lbl, act, c) in enumerate(items):
            y0 = by + bh * i
            if lbl == "LOOP" and self.loop:
                cv.create_rectangle(bx0 + 3, y0 + 3, bx0 + bw - 3, y0 + bh - 3, fill="#1d3a4a", outline="", tags=OV)
            cv.create_text(bx0 + bw / 2, y0 + 22, text=ic, fill=c, font=F(16), tags=OV)
            cv.create_text(bx0 + bw / 2, y0 + 45, text=lbl, fill=TXT, font=F(8), tags=OV)
            self.buttons.append((bx0, y0, bx0 + bw, y0 + bh, act))
        y0 = by + bh * len(items)
        cv.create_line(bx0 + 8, y0, bx0 + bw - 8, y0, fill=EDGE, tags=OV)
        cv.create_text(bx0 + bw / 2, y0 + 14, text="WP ALT", fill=DIM, font=F(8), tags=OV)
        cv.create_text(bx0 + bw / 2, y0 + 48, text=f"{self.h_new:.0f} m", fill=TXT, font=F(10, "bold"), tags=OV)
        cv.create_text(bx0 + bw / 2, y0 + 30, text="▲", fill=TXT, font=F(10), tags=OV)
        cv.create_text(bx0 + bw / 2, y0 + 68, text="▼", fill=TXT, font=F(10), tags=OV)
        self.buttons.append((bx0, y0 + 20, bx0 + bw, y0 + 40, lambda: self._h_step(10)))
        self.buttons.append((bx0, y0 + 58, bx0 + bw, y0 + 80, lambda: self._h_step(-10)))
        # масштаб: кнопки +/−, линейка
        zx = bx0 + bw + 12
        for i, (s, d) in enumerate((("+", 1), ("−", -1))):
            x0 = zx + 40 * i
            self._card(cv, x0, 10, x0 + 34, 44)
            cv.create_text(x0 + 17, 27, text=s, fill=TXT, font=F(14), tags=OV)
            self.buttons.append((x0, 10, x0 + 34, 44, lambda d=d: self._zoom_btn(d)))
        k = self._scale() / UI                              # номинальных пикс / м
        L = next((L for L in (50, 100, 200, 500, 1000, 2000) if L * k >= 70), 5000)
        sx0, sx1, sy = zx + 92, zx + 92 + L * k, 30
        for w, c in ((5, "black"), (2, "white")):
            cv.create_line(sx0, sy - 6, sx0, sy, sx1, sy, sx1, sy - 6, width=w, fill=c, tags=OV)
        cv.create_text((sx0 + sx1) / 2, sy - 13, text=f"{L} m  ×{self.levels[0] / self.sub:g}", font=F(9, "bold"), fill="white", tags=OV)
        # карточка точки / подсказка (справа вверху)
        x1 = MW - 10
        self._card(cv, x1 - 250, 10, x1, 58)
        if self.sel is not None and self.sel < len(self.wps):
            n, e, h = self.wps[self.sel]
            cv.create_text(x1 - 238, 26, text=f"WP {self.sel + 1}  ·  {h:.0f} m", anchor="w", fill=TXT,
                           font=F(11, "bold"), tags=OV)
            cv.create_text(x1 - 238, 46, text=f"N {n:.0f}  E {e:.0f} m", anchor="w", fill=DIM, font=F(9), tags=OV)
        else:
            cv.create_text(x1 - 238, 26, text="LMB add / drag   RMB delete", anchor="w", fill=TXT, font=F(9), tags=OV)
            cv.create_text(x1 - 238, 46, text="wheel: WP alt / zoom   MMB pan", anchor="w", fill=DIM, font=F(9), tags=OV)
        if m is None:
            return
        # полоса чисел (снизу)
        tgt = (m["dist"] / max(m["Vg"], 1.0)) if m["nav"] else None
        vals = [("ALT", f"{m['h']:.0f} m"), ("VS", f"{m['Vh']:+.1f} m/s"), ("GS", f"{m['Vg']:.1f} m/s"),
                ("TO WP", f"{m['dist']:.0f} m" if m["nav"] else "—"),
                ("ETE", fmt_time(tgt) if tgt is not None else "—"), ("DIST", f"{self.flown / 1000:.2f} km")]
        sx0, sy0, sw, sh = 92, MH - 66, 360, 56
        self._card(cv, sx0, sy0, sx0 + sw, sy0 + sh)
        for i, (k, v) in enumerate(vals):
            x, y = sx0 + 12 + 118 * (i % 3), sy0 + 16 + 26 * (i // 3)
            cv.create_text(x, y, text=k, anchor="w", fill=DIM, font=F(8), tags=OV)
            cv.create_text(x + 44, y, text=v, anchor="w", fill=TXT, font=FM(11), tags=OV)
        # компас: курс ЛА и ветер (север вверху)
        R, ccx, ccy = 50, MW - 72, MH - 92
        cv.create_oval(ccx - R - 10, ccy - R - 10, ccx + R + 10, ccy + R + 10, fill=BAR, outline=EDGE, tags=OV)
        for a in range(0, 360, 30):
            t = math.radians(a)
            l = 12 if a % 90 == 0 else 7
            cv.create_line(ccx + R * math.sin(t), ccy - R * math.cos(t), ccx + (R - l) * math.sin(t),
                           ccy - (R - l) * math.cos(t), fill="white", tags=OV)
        for a, s in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            t = math.radians(a)
            cv.create_text(ccx + (R - 22) * math.sin(t), ccy - (R - 22) * math.cos(t), text=s,
                           fill=RED if s == "N" else "white", font=F(9, "bold"), tags=OV)
        t = math.radians(m["psi"])
        f, rgt = (math.sin(t), -math.cos(t)), (math.cos(t), math.sin(t))
        tri = [(26, 0), (-14, 12), (-6, 0), (-14, -12)]
        cv.create_polygon(*[v for a, b in tri for v in (ccx + a * f[0] + b * rgt[0], ccy + a * f[1] + b * rgt[1])],
                          fill=AC, outline="", tags=OV)
        V = math.hypot(m["wind_n"], m["wind_e"])
        if V > 0.2:                                        # стрелка ветра снаружи круга — откуда дует
            ux, uy = m["wind_e"] / V, -m["wind_n"] / V      # куда дует, экранные оси
            cv.create_line(ccx - ux * (R + 44), ccy - uy * (R + 44), ccx - ux * (R + 12), ccy - uy * (R + 12),
                           fill=CYAN, width=3, arrow="last", arrowshape=(10, 12, 4), tags=OV)
        cv.create_text(ccx, ccy + R + 22, text=f"{m['psi']:03.0f}°  WIND {V:.1f}", fill="white",
                       font=F(9, "bold"), tags=OV)

    # --- правая колонка ----------------------------------------------------------------
    def _draw_side(self):
        cv, m, W, H = self.side, self.last, SIDE_W, self.PH / UI
        if m is None:
            return
        pad, y = 10, 8
        hs = [0.34, 0.15, 0.20]
        y = self._controls(cv, m, pad, y, W - pad, y + hs[0] * H) + 8
        y = self._aoa(cv, m, pad, y, W - pad, y + hs[1] * H) + 8
        y = self._energy(cv, m, pad, y, W - pad, y + hs[2] * H) + 8
        self._events(cv, m, pad, y, W - pad, H - 8)

    def _title(self, cv, x0, y0, x1, y1, title):
        cv.create_rectangle(x0, y0, x1, y1, fill=CARD, outline=EDGE)
        cv.create_text(x0 + 12, y0 + 14, text=title, anchor="w", fill=DIM, font=F(9, "bold"))

    def _bar(self, cv, x, y, w, val, cmd, lo, hi, zero=True):
        """Полоса: положение (заливка) и команда (риска)."""
        X = lambda v: x + w * (min(max(v, lo), hi) - lo) / (hi - lo)
        cv.create_rectangle(x, y - 6, x + w, y + 6, fill="#2b3036", outline="#454c55")
        x0 = X(0) if zero else x
        cv.create_rectangle(min(x0, X(val)), y - 6, max(x0, X(val)), y + 6, fill=GREEN, width=0)
        if zero:
            cv.create_line(x0, y - 9, x0, y + 9, fill="white")
        if cmd is not None:
            cv.create_line(X(cmd), y - 10, X(cmd), y + 10, fill=MAG, width=3)

    def _controls(self, cv, m, x0, y0, x1, y1):
        self._title(cv, x0, y0, x1, y1, "CONTROLS")
        info = self.info
        self._plane(cv, m, x0 + 92, (y0 + y1) / 2 + 6, min(0.14 * (y1 - y0), 34.0))
        da, de, dr = m["da"], m["de"], m["dr"]
        bx, bw = x0 + 190, x1 - x0 - 270
        rows = [("δe", de, m["de_cmd"], info["de_min"], info["de_max"], True),
                ("δa", da, m["da_cmd"], -info["da_max"], info["da_max"], True),
                ("δr", dr, m["dr_cmd"], -info["dr_max"], info["dr_max"], True),
                ("δt", m["thr"], m["thr_cmd"], 0.0, 1.0, False)]
        dy = (y1 - y0 - 50) / 4
        for i, (lbl, v, c, lo, hi, z) in enumerate(rows):
            y = y0 + 40 + dy * i
            cv.create_text(bx - 10, y, text=lbl, anchor="e", fill=TXT, font=FM(12))
            self._bar(cv, bx, y, bw, v, c, lo, hi, z)
            cv.create_text(bx + bw + 10, y, text=f"{v:+.1f}°" if z else f"{100 * v:.0f} %", anchor="w",
                           fill=TXT, font=FM(11))
        cv.create_text(bx, y1 - 12, text="▮ position   ", anchor="w", fill=GREEN, font=F(8))
        cv.create_text(bx + 80, y1 - 12, text="▎ command", anchor="w", fill=MAG, font=F(8))
        return y1

    def _plane(self, cv, m, cx, cy, u):
        """Вид сверху (нос вверх), u — пикс на условную единицу (полуразмах крыла 2.4).
        Рули — на задней кромке, цвет по отклонению: + оранжевый, − голубой; руль
        направления повёрнут на 2·δr (для наглядности)."""
        info = self.info
        P = lambda pts: [v for a, b in pts for v in (cx + u * a, cy - u * b)]
        body, line = "#7d8995", "#c9d2da"

        def col(v, vmax):
            k = max(-1.0, min(1.0, v / vmax))
            c0 = (0x5a, 0x66, 0x73)
            c1 = (0xff, 0x9a, 0x2e) if k > 0 else (0x3f, 0xc1, 0xff)
            a = min(1.0, abs(k) * 2.5)                    # заметно уже при малых отклонениях
            return "#%02x%02x%02x" % tuple(int(p + a * (q - p)) for p, q in zip(c0, c1))

        # крыло: трапеция со скруглёнными законцовками, элероны — внешние части задней кромки
        wing = [(-0.12, 0.78), (-2.25, 0.70), (-2.4, 0.62), (-2.4, 0.36), (-2.25, 0.30),
                (-0.12, 0.16), (0.12, 0.16), (2.25, 0.30), (2.4, 0.36), (2.4, 0.62), (2.25, 0.70), (0.12, 0.78)]
        cv.create_polygon(*P(wing), fill=body, outline=line, width=1)
        for sx, v in ((-1, -m["da"]), (1, m["da"])):     # δa > 0: правый элерон вверх, левый вниз
            a0, a1 = 1.15, 2.2
            te = lambda x: 0.16 + 0.14 * (x - 0.12) / 2.13          # задняя кромка
            ail = [(sx * a0, te(a0)), (sx * a1, te(a1)), (sx * a1, te(a1) + 0.12), (sx * a0, te(a0) + 0.12)]
            cv.create_polygon(*P(ail), fill=col(v, info["da_max"]), outline=line, width=1)
        # горизонтальное оперение и руль высоты
        stab = [(-0.06, -1.22), (-0.95, -1.30), (-1.0, -1.38), (-1.0, -1.58), (1.0, -1.58), (1.0, -1.38),
                (0.95, -1.30), (0.06, -1.22)]
        cv.create_polygon(*P(stab), fill=body, outline=line, width=1)
        cv.create_polygon(*P([(-0.92, -1.47), (0.92, -1.47), (0.92, -1.58), (-0.92, -1.58)]),
                          fill=col(m["de"], info["de_max"]), outline=line, width=1)
        # фюзеляж — обтекаемый (сглаженный многоугольник)
        fus = [(0, 2.05), (0.13, 1.95), (0.22, 1.6), (0.23, 0.9), (0.18, 0.0), (0.11, -1.0), (0.07, -1.6),
               (0.03, -1.75), (-0.03, -1.75), (-0.07, -1.6), (-0.11, -1.0), (-0.18, 0.0), (-0.23, 0.9),
               (-0.22, 1.6), (-0.13, 1.95)]
        cv.create_polygon(*P(fus), fill="#93a0ac", outline=line, width=1, smooth=True)
        cv.create_line(*P([(-0.42, 2.1), (0.42, 2.1)]), fill="#aab3bb", width=2)          # винт
        cv.create_oval(*P([(-0.09, 1.45), (0.09, 1.05)]), fill="#3b4a58", outline="")      # фонарь
        # киль (вид сверху — линия) и руль направления
        cv.create_line(*P([(0, -1.05), (0, -1.68)]), fill=line, width=3)
        ang = math.radians(2.0 * m["dr"])
        L = 0.32
        cv.create_line(*P([(0, -1.68), (-L * math.sin(ang), -1.68 - L * math.cos(ang))]),
                       fill=col(m["dr"], info["dr_max"]), width=4, capstyle="round")

    def _aoa(self, cv, m, x0, y0, x1, y1):
        self._title(cv, x0, y0, x1, y1, "ANGLE OF ATTACK / SIDESLIP")
        aw, ac = self.info["alpha_warn"], self.info["alpha_crit"]
        c = [GREEN, AMBER, RED, RED][min(m["alert"], 3)]
        yc = (y0 + y1) / 2 + 8
        cv.create_text(x0 + 14, yc, text=f"α {m['alpha']:.1f}°", anchor="w", fill=c, font=FM(17, "bold"))
        gx, gw, lo, hi = x0 + 140, 170, -5.0, 20.0
        X = lambda a: gx + gw * (min(max(a, lo), hi) - lo) / (hi - lo)
        for a0, a1, cc in ((lo, aw, GREEN), (aw, ac, AMBER), (ac, hi, RED)):
            cv.create_rectangle(X(a0), yc - 3, X(a1), yc + 5, fill=cc, width=0)
        xa = X(m["alpha"])
        cv.create_polygon(xa, yc - 5, xa - 7, yc - 17, xa + 7, yc - 17, fill="white", outline="")
        cv.create_text(X(aw), yc + 16, text=f"{aw:.0f}°", fill=AMBER, font=F(8))
        cv.create_text(X(ac), yc + 16, text=f"{ac:.1f}°", fill=RED, font=F(8))
        bx = x1 - 62                                         # «шарик» скольжения
        cv.create_rectangle(bx - 44, yc - 8, bx + 44, yc + 8, fill="#000", outline="#888")
        for s in (-1, 1):
            cv.create_line(bx + s * 9, yc - 11, bx + s * 9, yc + 11, fill="white")
        xb = bx + max(-1.0, min(1.0, m["beta"] / 10.0)) * 36
        cv.create_oval(xb - 7, yc - 7, xb + 7, yc + 7, fill="white", outline="")
        cv.create_text(bx, yc + 22, text=f"β {m['beta']:+.1f}°", fill=TXT, font=FM(10))
        return y1

    def _energy(self, cv, m, x0, y0, x1, y1):
        self._title(cv, x0, y0, x1, y1, "ENERGY (TECS)")
        e = m["tecs"]
        if e is None:
            cv.create_text((x0 + x1) / 2, (y0 + y1) / 2 + 8, text="TECS off (manual / split loops)", fill=DIM, font=F(10))
            return y1
        bx, bw = x0 + 70, x1 - x0 - 180
        rows = [("Ė", e[0], e[1]), ("Ḃ", e[2], e[3])]
        dy = (y1 - y0 - 50) / 2
        for i, (lbl, v, r) in enumerate(rows):
            y = y0 + 42 + dy * i
            cv.create_text(bx - 12, y, text=lbl, anchor="e", fill=TXT, font=F(14))
            self._bar(cv, bx, y, bw, v, r, -5.0, 5.0)
            cv.create_text(bx + bw + 10, y, text=f"{v:+.2f} m/s", anchor="w", fill=TXT, font=FM(11))
        cv.create_text(x0 + 12, y1 - 12, text="Ė → throttle,  Ḃ → θ_ref     (× Va: climb-rate equivalent)",
                       anchor="w", fill=DIM, font=F(8))
        return y1

    def _events(self, cv, m, x0, y0, x1, y1):
        self._title(cv, x0, y0, x1, y1, "EVENTS")
        y = y0 + 38
        for t, lbl, c in reversed(m["events"]):
            if y > y1 - 10:
                break
            cv.create_text(x0 + 14, y, text=fmt_time(t), anchor="w", fill=DIM, font=FM(10))
            cv.create_text(x0 + 74, y, text=lbl, anchor="w", fill=EV_COL.get(c, TXT), font=F(10))
            y += 22


def run(q_cmd, q_tel, bg_path, extent, info, geometry="+0+0", h_new=150.0):
    """Точка входа процесса окна станции."""
    root = tk.Tk()
    Station(root, q_cmd, q_tel, bg_path, extent, info, h_new, pos=geometry)
    root.mainloop()
