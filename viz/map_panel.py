# -*- coding: utf-8 -*-
"""
Окно карты: вид сверху, ЛА со следом и маршрут по точкам для навигации САУ.
Работает в ОТДЕЛЬНОМ процессе (tkinter, стандартная библиотека) рядом с
3D-окном scenarios/GameScenario3D.py и scenarios/Bench3D.py.

Мышь:
  ЛКМ по пустому месту — новая точка в конец маршрута (высота — из поля «h новой»)
  ЛКМ по точке и тянуть — перенести точку
  ПКМ по точке — удалить;   колесо над точкой — её высота ±10 м
  колесо над картой — масштаб ×1 / ×2 / ×4;  средняя кнопка и тянуть — сдвиг карты
Кнопки: «Лететь по маршруту» — САУ включается и ведёт по точкам с первой;
        «Стоп» — навигация выкл, САУ держит текущие курс и высоту;
        «Очистить»; флажок «замкнуть» — после последней точки снова к первой
        (без него — кружение вокруг последней);  «К ЛА» — карту на ЛА.

Связь — две очереди multiprocessing:
    q_cmd (окно → 3D):  ("route", {"wps": [(N, E, h), ...], "loop": bool}) — после правки
                        ("go", None) | ("stop", None)
    q_tel (3D → окно):  dict на конец кадра: t, n, e, h, psi, Va, ap, nav, idx, mode,
                        circle (c_N, c_E, ρ) | None, dist (до текущей точки, м)

Фон — PNG вида сверху (World.save_top_image(), viz/world3d.py) с охватом
extent = ((E_мин, E_макс), (N_мин, N_макс)), м, в ZOOM_MAX раз крупнее окна:
при ×1 показывается каждый ZOOM_MAX-й пиксель, при ×ZOOM_MAX — 1:1.
Физики и САУ модуль не знает.
"""

import math
import queue
import multiprocessing as mp
import tkinter as tk

TICK_MS = 40                   # период обновления, мс
TRAIL_DT = 0.5                 # шаг точек следа, с
TRAIL_MAX = 2400               # точек следа (20 мин)
PICK_PX = 10                   # радиус «попадания» в точку, пикс
ZOOM_MAX = 4                   # фон — в ZOOM_MAX раз крупнее окна
FONT = ("Segoe UI", 9)
FONT_B = ("Segoe UI", 9, "bold")
C_ROUTE, C_WP, C_TGT, C_AC, C_TRAIL = "#ffd400", "#ffffff", "#ff7f0e", "#d62728", "#ff4040"


class MapPanel:
    def __init__(self, root, q_cmd, q_tel, bg_path, extent, h_new=150.0):
        self.root, self.q_cmd, self.q_tel = root, q_cmd, q_tel
        (self.e0, self.e1), (self.n0, self.n1) = extent
        self.wps = []                     # [[N, E, h], ...]
        self.trail = []                   # [(N, E), ...]
        self.last = None
        self.drag = None                  # индекс перетаскиваемой точки
        self._t_trail = 0.0

        root.title("Карта — маршрут САУ")
        self.src = tk.PhotoImage(file=bg_path)
        self.SW, self.SH = self.src.width(), self.src.height()
        self.W, self.H = self.SW // ZOOM_MAX, self.SH // ZOOM_MAX
        self.cv = tk.Canvas(root, width=self.W, height=self.H, highlightthickness=0, bg="#888")
        self.cv.pack()
        self.bg = self.cv.create_image(0, 0, anchor="nw")
        self.view = None
        self.z, self.vx, self.vy = 1, 0, 0        # масштаб и левый верхний угол вида (пикс фона)
        self._pan = None

        bar = tk.Frame(root)
        bar.pack(fill="x")
        tk.Button(bar, text="Лететь по маршруту", font=FONT_B, bg="#d8f0d8",
                  command=lambda: self._put(("go", None))).pack(side="left", padx=4, pady=4)
        tk.Button(bar, text="Стоп", font=FONT,
                  command=lambda: self._put(("stop", None))).pack(side="left", padx=2)
        tk.Button(bar, text="Очистить", font=FONT, command=self._clear).pack(side="left", padx=2)
        self.loop = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="замкнуть", variable=self.loop, font=FONT,
                       command=self._send).pack(side="left", padx=6)
        tk.Label(bar, text="h новой, м:", font=FONT).pack(side="left")
        self.h_new = tk.DoubleVar(value=h_new)
        tk.Spinbox(bar, from_=20, to=1000, increment=10, width=5, textvariable=self.h_new,
                   font=FONT).pack(side="left")
        tk.Button(bar, text="К ЛА", font=FONT, command=self._center_ac).pack(side="right", padx=4)
        self.status = tk.Label(root, text="", font=("Consolas", 9), anchor="w", justify="left")
        self.status.pack(fill="x", padx=4)
        tk.Label(root, font=("Segoe UI", 8), fg="#666", anchor="w",
                 text="ЛКМ — точка / перенос   ПКМ — удалить   колесо: над точкой — её высота "
                      "±10 м, над картой — масштаб   средняя кнопка — сдвиг"
                 ).pack(fill="x", padx=4, pady=(0, 3))

        cv = self.cv
        cv.bind("<Button-1>", self._press)
        cv.bind("<B1-Motion>", self._motion)
        cv.bind("<ButtonRelease-1>", self._release)
        cv.bind("<Button-3>", self._delete)
        cv.bind("<MouseWheel>", self._wheel)
        cv.bind("<Button-2>", lambda ev: setattr(self, "_pan", (ev.x, ev.y)))
        cv.bind("<B2-Motion>", self._pan_move)
        self._set_view(1, 0, 0)
        root.after(TICK_MS, self._tick)

    # --- координаты и вид ---------------------------------------------------
    def _k(self):
        return self.z / ZOOM_MAX                             # пикс окна на пикс фона

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

    def _set_view(self, z, vx, vy):
        """Масштаб z и левый верхний угол (vx, vy) в пикселях фона; вид — в пределах карты."""
        self.z = z
        w, h = self.W / self._k(), self.H / self._k()
        self.vx = int(min(max(vx, 0), self.SW - w))
        self.vy = int(min(max(vy, 0), self.SH - h))
        self.view = tk.PhotoImage()
        sub = ZOOM_MAX // z
        self.view.tk.call(self.view, "copy", self.src, "-from", self.vx, self.vy,
                          self.vx + int(w), self.vy + int(h), "-subsample", sub, sub)
        self.cv.itemconfigure(self.bg, image=self.view)
        self._scale_bar()
        self._draw()

    def _zoom_at(self, x, y, z):
        sx, sy = self.vx + x / self._k(), self.vy + y / self._k()     # точка под курсором
        k = z / ZOOM_MAX
        self._set_view(z, sx - x / k, sy - y / k)

    def _pan_move(self, ev):
        if self._pan:
            dx, dy = ev.x - self._pan[0], ev.y - self._pan[1]
            self._pan = (ev.x, ev.y)
            self._set_view(self.z, self.vx - dx / self._k(), self.vy - dy / self._k())

    def _center_ac(self):
        if self.last:
            x, y = self.to_px(self.last["n"], self.last["e"])
            self._set_view(self.z, self.vx + (x - self.W / 2) / self._k(),
                           self.vy + (y - self.H / 2) / self._k())

    def _pick(self, x, y):
        best, k_best = PICK_PX, None
        for k, (n, e, _) in enumerate(self.wps):
            px, py = self.to_px(n, e)
            d = math.hypot(px - x, py - y)
            if d <= best:
                best, k_best = d, k
        return k_best

    # --- мышь ----------------------------------------------------------------
    def _press(self, ev):
        k = self._pick(ev.x, ev.y)
        if k is None:
            n, e = self.to_ne(ev.x, ev.y)
            self.wps.append([n, e, float(self.h_new.get())])
            self._send()
        else:
            self.drag = k

    def _motion(self, ev):
        if self.drag is not None:
            x, y = min(max(ev.x, 0), self.W), min(max(ev.y, 0), self.H)
            self.wps[self.drag][:2] = self.to_ne(x, y)
            self._draw()

    def _release(self, _ev):
        if self.drag is not None:
            self.drag = None
            self._send()

    def _delete(self, ev):
        k = self._pick(ev.x, ev.y)
        if k is not None:
            del self.wps[k]
            self._send()

    def _wheel(self, ev):
        k = self._pick(ev.x, ev.y)
        if k is not None:
            self.wps[k][2] = min(max(self.wps[k][2] + (10 if ev.delta > 0 else -10), 20), 1000)
            self._send()
            return
        z = min(self.z * 2, ZOOM_MAX) if ev.delta > 0 else max(self.z // 2, 1)
        if z != self.z:
            self._zoom_at(ev.x, ev.y, z)

    def _clear(self):
        self.wps = []
        self._send()

    # --- связь -----------------------------------------------------------------
    def _put(self, msg):
        try:
            self.q_cmd.put_nowait(msg)
        except queue.Full:
            pass

    def _send(self):
        self._put(("route", {"wps": [tuple(w) for w in self.wps], "loop": self.loop.get()}))
        self._draw()

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
                self.trail = []
            if not self.trail or m["t"] - self._t_trail >= TRAIL_DT:
                self.trail.append((m["n"], m["e"]))
                del self.trail[:-TRAIL_MAX]
                self._t_trail = m["t"]
            self.last = m
        self._draw()
        self.root.after(TICK_MS, self._tick)

    # --- рисование -----------------------------------------------------------------
    def _scale_bar(self):
        self.cv.delete("bar")
        L = next(L for L in (50, 100, 200, 500, 1000, 2000) if L * self._scale() >= 70)
        x0, y0, x1 = 15, self.H - 18, 15 + L * self._scale()
        for w, c in ((5, "black"), (3, "white")):
            self.cv.create_line(x0, y0, x1, y0, width=w, fill=c, tags="bar")
        self.cv.create_text((x0 + x1) / 2, y0 - 10, text=f"{L} м   ×{self.z}", font=FONT_B,
                            fill="white", tags="bar")
        self.cv.create_text(self.W - 18, 18, text="С\n↑", font=FONT_B, fill="white",
                            justify="center", tags="bar")

    def _draw(self):
        cv, m = self.cv, self.last
        cv.delete("dyn")
        if len(self.trail) > 1:
            pts = [c for n, e in self.trail for c in self.to_px(n, e)]
            cv.create_line(*pts, fill=C_TRAIL, width=2, tags="dyn")
        nav = bool(m and m["nav"])
        # текущая дуга / окружность навигатора
        if nav and m["circle"]:
            cn, ce, rho = m["circle"]
            cx, cy = self.to_px(cn, ce)
            r = rho * self._scale()
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline=C_TGT, dash=(4, 3), tags="dyn")
        # маршрут
        P = [self.to_px(n, e) for n, e, _ in self.wps]
        if len(P) > 1:
            cv.create_line(*[c for p in P for c in p], fill=C_ROUTE, width=2, tags="dyn")
            if self.loop.get():
                cv.create_line(*P[-1], *P[0], fill=C_ROUTE, width=2, dash=(6, 4), tags="dyn")
        if nav and self.wps and m["idx"] < len(self.wps) and m["mode"] != "orbit":
            cv.create_line(*self.to_px(m["n"], m["e"]), *P[m["idx"]], fill=C_TGT,
                           dash=(3, 3), tags="dyn")
        for k, ((x, y), (_, _, h)) in enumerate(zip(P, self.wps)):
            tgt = nav and k == m["idx"]
            cv.create_oval(x - 8, y - 8, x + 8, y + 8, fill=C_TGT if tgt else C_WP,
                           outline="black", width=1.5, tags="dyn")
            cv.create_text(x, y, text=str(k + 1), font=FONT_B, tags="dyn")
            cv.create_text(x + 11, y - 11, text=f"{h:.0f} м", font=FONT, fill="white",
                           anchor="w", tags="dyn")
        # ЛА: треугольник по курсу ψ
        if m:
            x, y = self.to_px(m["n"], m["e"])
            s, c = math.sin(m["psi"]), math.cos(m["psi"])
            tri = [(0, -13), (7, 8), (0, 4), (-7, 8)]                # нос вверх в своей СК
            pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in tri]
            cv.create_polygon(*[v for p in pts for v in p], fill=C_AC, outline="white",
                              width=1.5, tags="dyn")
            self._status_text(m)

    def _status_text(self, m):
        if not m["ap"]:
            mode = "САУ ВЫКЛ (ручное)"
        elif not m["nav"]:
            mode = "САУ: курс/высота вручную"
        elif m["mode"] == "orbit":
            mode = f"НАВ: кружение у точки {len(self.wps)}"
        else:
            arc = " (дуга)" if m["mode"] == "fillet" else ""
            mode = f"НАВ: на точку {m['idx'] + 1}/{len(self.wps)}{arc}, до неё {m['dist']:.0f} м"
        self.status.configure(text=f"t {m['t']:6.1f} с   N {m['n']:7.0f}  E {m['e']:7.0f}  "
                                   f"h {m['h']:5.0f} м   Va {m['Va']:4.1f}   {mode}")


def run(q_cmd, q_tel, bg_path, extent, geometry="+0+0", h_new=150.0):
    """Точка входа процесса окна карты."""
    root = tk.Tk()
    root.geometry(geometry)
    root.resizable(False, False)
    MapPanel(root, q_cmd, q_tel, bg_path, extent, h_new)
    root.mainloop()
