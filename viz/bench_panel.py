# -*- coding: utf-8 -*-
"""
Окно испытательного стенда: блок-схема боковой САУ (канал крена) с ползунками
параметров и осциллограф. Работает в ОТДЕЛЬНОМ процессе (tkinter, стандартная
библиотека) рядом с 3D-окном scenarios/Bench3D.py.

Связь — две очереди multiprocessing:
    q_par  (окно → 3D):  ("par", {ключ: значение})  — новые параметры САУ
                         ("chi", шаг_град)           — ступенька уставки курса
    q_tel  (3D → окно):  dict сигналов на конец кадра (см. Bench3D._telemetry)

Модуль физики и САУ не импортирует — только рисует то, что пришло.
Значения параметров — в «инженерных» единицах окна (углы в градусах);
перевод в радианы делает Bench3D.
"""

import queue
import multiprocessing as mp
import tkinter as tk
from collections import deque

W_SCHEME, H_SCHEME = 1060, 380
W_SCOPE, H_SCOPE = 1060, 400
SCOPE_WIN = 20.0              # окно осциллографа, с
TICK_MS = 40                  # период обновления окна, мс (~25 Гц)

FONT = ("Segoe UI", 9)
FONT_B = ("Segoe UI", 10, "bold")
FONT_V = ("Consolas", 9)
C_LINE, C_BOX, C_SAT = "#333", "#eef3fb", "#fff4e0"
C_REF, C_MEAS, C_CTRL = "#d62728", "#1f77b4", "#2ca02c"

# Ползунки на схеме: ключ → (x, y) верхнего центра, длина в пикс, цвет блока
SLIDER_POS = {
    "chi_Kp": (205, 78, 120, C_BOX), "chi_Ki": (205, 140, 120, C_BOX),
    "phi_ref_max": (370, 122, 90, C_SAT),
    "phi_Kp": (595, 78, 120, C_BOX), "phi_Ki": (595, 140, 120, C_BOX),
    "da_max": (800, 122, 90, C_SAT),
    "phi_Kd": (800, 245, 90, C_BOX),
}


class Panel:
    def __init__(self, root, q_par, q_tel, spec):
        """spec — [(ключ, подпись, мин, макс, шаг, по_умолчанию), ...]"""
        self.root, self.q_par, self.q_tel = root, q_par, q_tel
        self.spec = spec
        self.vars = {}
        self.data = deque()               # (t, chi_ref, chi, phi_ref, phi, da)
        self.last = None

        root.title("Стенд: боковая САУ — канал крена")
        self.cv = tk.Canvas(root, width=W_SCHEME, height=H_SCHEME, bg="white",
                            highlightthickness=0)
        self.cv.pack()
        self._scheme()
        self._buttons()
        self.sc = tk.Canvas(root, width=W_SCOPE, height=H_SCOPE, bg="white",
                            highlightthickness=0)
        self.sc.pack()
        self._send_params()
        root.after(TICK_MS, self._tick)

    # --- схема ------------------------------------------------------------
    def _box(self, x0, y0, x1, y1, title, fill=C_BOX):
        self.cv.create_rectangle(x0, y0, x1, y1, fill=fill, outline=C_LINE, width=1.5)
        self.cv.create_text((x0 + x1) / 2, y0 + 10, text=title, font=FONT_B)

    def _sum(self, x, y):
        r = 11
        self.cv.create_oval(x - r, y - r, x + r, y + r, outline=C_LINE, width=1.5, fill="white")
        self.cv.create_line(x - 7, y - 7, x + 7, y + 7, fill=C_LINE)
        self.cv.create_line(x - 7, y + 7, x + 7, y - 7, fill=C_LINE)
        self.cv.create_text(x - 15, y - 12, text="+", font=FONT_B)
        self.cv.create_text(x - 9, y + 20, text="−", font=FONT_B)

    def _arrow(self, *pts):
        self.cv.create_line(*pts, fill=C_LINE, width=1.5, arrow=tk.LAST, arrowshape=(8, 10, 4))

    def _val(self, x, y, key, color="#444", anchor="center"):
        self.cv.create_text(x, y, text="", font=FONT_V, fill=color, anchor=anchor, tags=key)

    def _sat_icon(self, xc, yc):
        self.cv.create_line(xc - 22, yc + 9, xc - 9, yc + 9, xc + 9, yc - 9, xc + 22, yc - 9,
                            fill=C_LINE, width=1.5)

    def _scheme(self):
        cv, Y = self.cv, 130
        cv.create_text(10, 16, anchor="w", font=("Segoe UI", 11, "bold"),
                       text="Боковая САУ: курс χ → крен φ → элероны δa")
        self._val(1050, 16, "status", anchor="e")

        # вход
        cv.create_text(30, 110, text="χ_ref", font=FONT_B, fill=C_REF)
        self._val(30, 150, "v_chi_ref", C_REF)
        self._arrow(50, Y, 69, Y)
        self._sum(80, Y)
        self._arrow(91, Y, 135, Y)
        self._val(113, 115, "v_e_chi")

        self._box(135, 55, 275, 205, "ПИ курса")
        self._arrow(275, Y, 315, Y)
        self._box(315, 75, 425, 185, "огр. φ_ref", C_SAT)
        self._sat_icon(370, 103)
        self._arrow(425, Y, 469, Y)
        self._val(447, 115, "v_phi_ref", C_REF)
        self._sum(480, Y)
        self._arrow(491, Y, 525, Y)
        self._val(508, 115, "v_e_phi")

        self._box(525, 55, 665, 205, "ПИ крена")
        self._arrow(665, Y, 689, Y)
        self._sum(700, Y)
        self._arrow(711, Y, 745, Y)
        self._box(745, 75, 855, 185, "огр. δa", C_SAT)
        self._sat_icon(800, 103)
        self._arrow(855, Y, 900, Y)
        self._val(877, 115, "v_da", C_CTRL)

        self._box(900, 85, 990, 175, "ЛА")
        cv.create_text(945, 130, text="6DOF\n(3D-окно)", font=FONT, justify="center")

        # выходы ЛА и обратные связи
        self._arrow(990, 100, 1045, 100, 1045, 355, 80, 355, 80, 141)        # χ (GPS)
        cv.create_text(1000, 92, text="χ", font=FONT_B, fill=C_MEAS, anchor="w")
        self._val(300, 343, "v_chi", C_MEAS)
        self._arrow(990, Y, 1025, Y, 1025, 320, 480, 320, 480, 141)          # φ (ИНС)
        cv.create_text(1000, 122, text="φ", font=FONT_B, fill=C_MEAS, anchor="w")
        self._val(600, 308, "v_phi", C_MEAS)
        self._arrow(990, 160, 1005, 160, 1005, 255, 855, 255)                # p (гироскоп)
        cv.create_text(995, 152, text="p", font=FONT_B, fill=C_MEAS, anchor="w")
        self._val(930, 243, "v_p", C_MEAS)
        self._box(745, 222, 855, 305, "демпф. Kd")
        self._arrow(745, 255, 700, 255, 700, 141)
        self._val(675, 270, "v_kdp")

        cv.create_text(10, H_SCHEME - 8, anchor="w", font=("Segoe UI", 8), fill="#777",
                       text="δa = Kp·e_φ + Ki·∫e_φ − Kd·p;   φ_ref = Kp·e_χ + Ki·∫e_χ.   "
                            "Руль направления — контур β (штатный, на схеме не показан).")

        for key, label, lo, hi, res, default in self.spec:
            v = tk.DoubleVar(value=default)
            self.vars[key] = v
            x, y, length, bg = SLIDER_POS[key]
            s = tk.Scale(cv, variable=v, from_=lo, to=hi, resolution=res, orient="horizontal",
                         length=length, label=label, font=FONT, bg=bg, bd=0,
                         highlightthickness=0, command=lambda _v: self._send_params())
            cv.create_window(x, y, window=s, anchor="n")

    def _buttons(self):
        f = tk.Frame(self.root, bg="white")
        f.pack(fill="x")
        tk.Label(f, text="  Ступенька χ_ref:", font=FONT, bg="white").pack(side="left")
        for d in (-90, -30, -10, 10, 30, 90):
            tk.Button(f, text=f"{d:+d}°", font=FONT, width=5,
                      command=lambda d=d: self._put(("chi", d))).pack(side="left", padx=2, pady=4)
        tk.Button(f, text="Параметры по умолчанию", font=FONT,
                  command=self._defaults).pack(side="right", padx=8)

    def _defaults(self):
        for key, *_, default in self.spec:
            self.vars[key].set(default)
        self._send_params()

    def _put(self, msg):
        try:
            self.q_par.put_nowait(msg)
        except queue.Full:
            pass

    def _send_params(self):
        self._put(("par", {k: v.get() for k, v in self.vars.items()}))

    # --- обновление ---------------------------------------------------------
    def _tick(self):
        parent = mp.parent_process()
        if parent is not None and not parent.is_alive():   # 3D-окно закрыто/упало
            self.root.destroy()
            return
        while True:
            try:
                m = self.q_tel.get_nowait()
            except queue.Empty:
                break
            if self.data and m["t"] < self.data[-1][0]:     # сброс полёта (R)
                self.data.clear()
            self.data.append((m["t"], m["chi_ref"], m["chi"], m["phi_ref"], m["phi"], m["da"]))
            self.last = m
        while self.data and self.data[-1][0] - self.data[0][0] > SCOPE_WIN:
            self.data.popleft()
        if self.last:
            self._values(self.last)
        self._scope()
        self.root.after(TICK_MS, self._tick)

    def _values(self, m):
        cv = self.cv
        txt = {
            "v_chi_ref": f"{m['chi_ref'] % 360:5.1f}°", "v_e_chi": f"{m['e_chi']:+.1f}°",
            "v_phi_ref": f"{m['phi_ref']:+.1f}°", "v_e_phi": f"{m['e_phi']:+.1f}°",
            "v_da": f"{m['da']:+.1f}°", "v_chi": f"χ = {m['chi'] % 360:5.1f}°",
            "v_phi": f"φ = {m['phi']:+.1f}°", "v_p": f"{m['p']:+.0f}°/с",
            "v_kdp": f"Kd·p = {m['kdp']:+.1f}°",
            "status": (f"t = {m['t']:6.1f} с   " +
                       ("САУ ВКЛ" if m["ap"] else "САУ ВЫКЛ (P в 3D-окне) — контур разомкнут")),
        }
        for k, s in txt.items():
            cv.itemconfigure(k, text=s)
        cv.itemconfigure("status", fill="#2a7" if m["ap"] else "#c33")

    def _scope(self):
        sc = self.sc
        sc.delete("all")
        strips = [("курс χ, °", [(1, C_REF, "χ_ref"), (2, C_MEAS, "χ")], None),
                  ("крен φ, °", [(3, C_REF, "φ_ref"), (4, C_MEAS, "φ")], 35.0),
                  ("элероны δa, °", [(5, C_CTRL, "δa")], 10.0)]
        x0, x1 = 70, W_SCOPE - 15
        h = (H_SCOPE - 30) / len(strips)
        d = list(self.data)
        t_end = d[-1][0] if d else 0.0
        t_beg = t_end - SCOPE_WIN if t_end > SCOPE_WIN else 0.0
        for i, (title, lines, sym) in enumerate(strips):
            y0, y1 = 10 + i * h, 10 + i * h + h - 22
            vals = [r[j] for r in d for j, *_ in lines]
            if sym is not None:                       # симметричная шкала
                a = max([sym] + [abs(v) * 1.1 for v in vals])
                lo, hi = -a, a
            else:
                lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
                mid, half = (lo + hi) / 2, max((hi - lo) / 2 * 1.15, 15.0)
                lo, hi = mid - half, mid + half
            ty = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
            tx = lambda t: x0 + (t - t_beg) / SCOPE_WIN * (x1 - x0)
            sc.create_rectangle(x0, y0, x1, y1, outline="#bbb")
            for k in range(5):
                v = lo + (hi - lo) * k / 4
                sc.create_line(x0, ty(v), x1, ty(v), fill="#eee")
                sc.create_text(x0 - 4, ty(v), text=f"{v:.0f}", anchor="e", font=("Segoe UI", 8))
            if lo < 0 < hi:
                sc.create_line(x0, ty(0), x1, ty(0), fill="#ccc")
            sc.create_text(x0 + 6, y0 + 8, text=title, anchor="w", font=FONT_B)
            lx = x1 - 10
            for j, color, name in reversed(lines):
                sc.create_text(lx, y0 + 8, text=name, anchor="e", font=FONT_B, fill=color)
                lx -= 50
                if len(d) > 1:
                    pts = []
                    for r in d:
                        pts += (tx(r[0]), ty(r[j]))
                    sc.create_line(*pts, fill=color, width=2 if j in (2, 4, 5) else 1.5,
                                   dash=() if j in (2, 4, 5) else (6, 3))
        for k in range(0, int(SCOPE_WIN) + 1, 5):          # подписи времени
            t = t_beg + k
            sc.create_text(x0 + k / SCOPE_WIN * (x1 - x0), H_SCOPE - 12,
                           text=f"{t:.0f} с", font=("Segoe UI", 8))


def run(q_par, q_tel, spec, geometry="-0+0"):
    """Точка входа процесса окна стенда."""
    root = tk.Tk()
    root.configure(bg="white")
    root.geometry(geometry)
    Panel(root, q_par, q_tel, spec)
    root.mainloop()
