# -*- coding: utf-8 -*-
"""
Окно испытательного стенда САУ: вкладки по контурам (крен, тангаж/высота,
скорость, рыскание, защита по α, навигация). На каждой вкладке — блок-схема контура с ползунками
параметров, живыми значениями сигналов, кнопками воздействий и осциллограф.
Работает в ОТДЕЛЬНОМ процессе (tkinter, стандартная библиотека) рядом с
3D-окном scenarios/Bench3D.py.

Связь — две очереди multiprocessing:
    q_par  (окно → 3D):  ("par", {ключ: значение})  — все параметры САУ
                         ("chi" | "h" | "va", шаг)  — ступенька уставки (°, м, м/с)
                         ("kick", "de" | "da" | "dr") — толчок рулём на 1 с
                         ("stall" | "stall_off", _)  — режим «Срыв» вкл / отмена
                         ("go" | "stop", _)  — навигация по маршруту (окно карты) вкл / выкл
    q_tel  (3D → окно):  dict сигналов на конец кадра (см. Bench3D._telemetry)

Модуль физики и САУ не импортирует — только рисует то, что пришло.
Значения параметров — в «инженерных» единицах окна (углы в градусах);
перевод в радианы делает Bench3D.apply_params().
"""

import queue
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk
from collections import deque

W, H_SCHEME, H_SCOPE = 1100, 380, 400
SCOPE_WIN = 20.0              # окно осциллографа, с
TICK_MS = 40                  # период обновления окна, мс (~25 Гц)

FONT = ("Segoe UI", 9)
FONT_B = ("Segoe UI", 10, "bold")
FONT_S = ("Segoe UI", 8)
FONT_V = ("Consolas", 9)
C_LINE, C_BOX, C_SAT = "#333", "#eef3fb", "#fff4e0"
C_REF, C_MEAS, C_CTRL = "#d62728", "#1f77b4", "#2ca02c"

# Ползунки/флажки на схемах: ключ → (x, y) верхнего центра, длина в пикс, цвет блока
SLIDER_POS = {
    # крен
    "chi_Kp": (205, 78, 120, C_BOX), "chi_Ki": (205, 140, 120, C_BOX),
    "phi_ref_max": (370, 122, 90, C_SAT),
    "phi_Kp": (595, 78, 120, C_BOX), "phi_Ki": (595, 140, 120, C_BOX),
    "da_max": (800, 122, 90, C_SAT),
    "phi_Kd": (800, 245, 90, C_BOX),
    # тангаж / высота
    "KH": (155, 98, 85, C_BOX),
    "theta_Kp": (350, 55, 100, C_BOX), "theta_Ki": (350, 112, 100, C_BOX),
    "theta_Kd": (350, 169, 100, C_BOX),
    "q_max": (477, 127, 70, C_SAT),
    "q_Kp": (660, 55, 100, C_BOX), "q_Ki": (660, 112, 100, C_BOX), "q_Kd": (660, 169, 100, C_BOX),
    "gs": (787, 128, 0, C_BOX),
    "de_max": (897, 127, 70, C_SAT),
    # скорость
    "Va_Kp": (195, 70, 110, C_BOX), "Va_Ki": (195, 127, 110, C_BOX), "Va_Kd": (195, 184, 110, C_BOX),
    # рыскание
    "beta_hold": (215, 80, 0, C_BOX),
    "beta_Kp": (215, 108, 110, C_BOX), "beta_Ki": (215, 165, 110, C_BOX),
    "dr_max": (480, 140, 80, C_SAT),
    # защита по α
    "prot_on": (1010, 60, 0, C_BOX),
    "a_warn": (967, 88, 70, C_BOX), "a_crit": (967, 152, 70, C_BOX), "a_exit": (967, 216, 70, C_BOX),
    "th_warn": (1053, 88, 70, C_BOX), "th_rec": (1053, 152, 70, C_BOX), "thr_rec": (1053, 216, 70, C_BOX),
    # навигация
    "R_fillet": (295, 62, 160, C_BOX), "R_orbit": (295, 120, 160, C_BOX), "orbit_cw": (295, 183, 0, C_BOX),
    "chi_inf": (600, 62, 160, C_BOX), "k_path": (600, 120, 160, C_BOX), "k_orbit": (600, 178, 160, C_BOX),
}

# Осциллограф вкладки: (заголовок, [(ключ, цвет, подпись, пунктир)], шкала)
#   шкала: ("sym", a) — ±max(a, |v|), ("auto", мин_размах), ("fix", lo, hi)
SCOPES = {
    "roll": [("курс χ, °", [("chi_ref", C_REF, "χ_ref", 1), ("chi", C_MEAS, "χ", 0)], ("auto", 30)),
             ("крен φ, °", [("phi_ref", C_REF, "φ_ref", 1), ("phi", C_MEAS, "φ", 0)], ("sym", 35)),
             ("элероны δa, °", [("da", C_CTRL, "δa", 0)], ("sym", 10))],
    "pitch": [("высота h, м", [("h_ref", C_REF, "h_ref", 1), ("h", C_MEAS, "h", 0)], ("auto", 20)),
              ("тангаж θ, °", [("theta_ref", C_REF, "θ_ref", 1), ("theta", C_MEAS, "θ", 0)], ("sym", 10)),
              ("угл. скорость q, °/с", [("q_ref", C_REF, "q_ref", 1), ("q", C_MEAS, "q", 0)], ("sym", 10)),
              ("руль высоты δe, °", [("de", C_CTRL, "δe", 0)], ("auto", 6))],
    "speed": [("скорость Va, м/с", [("Va_ref", C_REF, "Va_ref", 1), ("Va", C_MEAS, "Va", 0)], ("auto", 6)),
              ("тяга δt", [("thr", C_CTRL, "δt", 0)], ("fix", 0.0, 1.0)),
              ("высота h, м", [("h_ref", C_REF, "h_ref", 1), ("h", C_MEAS, "h", 0)], ("auto", 20))],
    "yaw": [("скольжение β, °", [("beta", C_MEAS, "β", 0)], ("sym", 2)),
            ("руль направления δr, °", [("dr", C_CTRL, "δr", 0)], ("sym", 2)),
            ("крен φ, °", [("phi_ref", C_REF, "φ_ref", 1), ("phi", C_MEAS, "φ", 0)], ("sym", 35))],
    "prot": [("угол атаки α, °", [("a_warn", "#e69500", "α_пред", 1), ("a_crit", "#b00000", "α_крит", 1),
                                  ("alpha", C_MEAS, "α", 0)], ("auto", 10)),
             ("скорость Va, м/с", [("Va_ref", C_REF, "Va_ref", 1), ("Va", C_MEAS, "Va", 0)], ("auto", 6)),
             ("тангаж θ, °", [("theta_cmd", "#999", "θ_ref (h)", 1), ("theta_ref", C_REF, "θ_ref*", 1),
                              ("theta", C_MEAS, "θ", 0)], ("sym", 10)),
             ("тяга δt", [("thr", C_CTRL, "δt", 0)], ("fix", 0.0, 1.0))],
    "nav": [("отклонение от линии / окружности e_py, м", [("e_py", C_MEAS, "e_py", 0)], ("sym", 20)),
            ("курс χ, °", [("chi_ref", C_REF, "χ_ref", 1), ("chi", C_MEAS, "χ", 0)], ("auto", 30)),
            ("высота h, м", [("h_ref", C_REF, "h_ref", 1), ("h", C_MEAS, "h", 0)], ("auto", 20)),
            ("крен φ, °", [("phi_ref", C_REF, "φ_ref", 1), ("phi", C_MEAS, "φ", 0)], ("sym", 35))],
}

TABS = [("roll", "Крен"), ("pitch", "Тангаж / высота"), ("speed", "Скорость"), ("yaw", "Рыскание (β)"),
        ("prot", "Защита по α"), ("nav", "Навигация")]


class Panel:
    def __init__(self, root, q_par, q_tel, spec):
        """spec — [(вкладка, ключ, подпись, мин, макс, шаг, по_умолчанию), ...];
        шаг None — флажок вкл/выкл."""
        self.root, self.q_par, self.q_tel = root, q_par, q_tel
        self.spec = spec
        self.vars = {}
        self.data = deque()               # dict сигналов за последние SCOPE_WIN с
        self.last = None
        self.tabs = {}                    # имя → (схема, осциллограф, [(item, fmt)])

        root.title("Стенд САУ")
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)
        draw = {"roll": self._draw_roll, "pitch": self._draw_pitch,
                "speed": self._draw_speed, "yaw": self._draw_yaw, "prot": self._draw_prot,
                "nav": self._draw_nav}
        for name, title in TABS:
            f = tk.Frame(self.nb, bg="white")
            self.nb.add(f, text=f"  {title}  ")
            cv = tk.Canvas(f, width=W, height=H_SCHEME, bg="white", highlightthickness=0)
            cv.pack()
            self.tabs[name] = (cv, None, [])
            draw[name](cv)
            self._val(cv, W - 10, 16, self._status, anchor="e")
            self._controls(name, cv)
            bar = tk.Frame(f, bg="white")
            bar.pack(fill="x")
            self._buttons(name, bar)
            sc = tk.Canvas(f, width=W, height=H_SCOPE, bg="white", highlightthickness=0)
            sc.pack()
            self.tabs[name] = (cv, sc, self.tabs[name][2])
        self._send_params()
        root.after(TICK_MS, self._tick)

    # --- примитивы схемы --------------------------------------------------
    @staticmethod
    def _box(cv, x0, y0, x1, y1, title, fill=C_BOX):
        cv.create_rectangle(x0, y0, x1, y1, fill=fill, outline=C_LINE, width=1.5)
        cv.create_text((x0 + x1) / 2, y0 + 10, text=title, font=FONT_B)

    @staticmethod
    def _sum(cv, x, y, marks=(("+", -15, -12), ("−", -9, 20))):
        r = 11
        cv.create_oval(x - r, y - r, x + r, y + r, outline=C_LINE, width=1.5, fill="white")
        cv.create_line(x - 7, y - 7, x + 7, y + 7, fill=C_LINE)
        cv.create_line(x - 7, y + 7, x + 7, y - 7, fill=C_LINE)
        for sign, dx, dy in marks:
            cv.create_text(x + dx, y + dy, text=sign, font=FONT_B)

    @staticmethod
    def _arrow(cv, *pts):
        cv.create_line(*pts, fill=C_LINE, width=1.5, arrow=tk.LAST, arrowshape=(8, 10, 4))

    @staticmethod
    def _sat_icon(cv, xc, yc):
        cv.create_line(xc - 22, yc + 9, xc - 9, yc + 9, xc + 9, yc - 9, xc + 22, yc - 9,
                       fill=C_LINE, width=1.5)

    def _val(self, cv, x, y, fmt, color="#444", anchor="center"):
        """Живое значение на схеме: fmt(сигналы) → текст."""
        item = cv.create_text(x, y, text="", font=FONT_V, fill=color, anchor=anchor)
        for name, (c, _, vals) in self.tabs.items():
            if c is cv:
                vals.append((item, fmt))

    def _dyn(self, cv, fn):
        """Динамический элемент схемы: fn(холст, сигналы) вызывается каждый тик."""
        for name, (c, _, vals) in self.tabs.items():
            if c is cv:
                vals.append((None, fn))

    @staticmethod
    def _title(cv, text):
        cv.create_text(10, 16, anchor="w", font=("Segoe UI", 11, "bold"), text=text)

    @staticmethod
    def _footer(cv, text):
        cv.create_text(10, H_SCHEME - 10, anchor="w", font=FONT_S, fill="#777", text=text)

    @staticmethod
    def _status(m):
        return f"t = {m['t']:6.1f} с   " + ("САУ ВКЛ" if m["ap"] else
                                           "САУ ВЫКЛ (P в 3D-окне) — контуры разомкнуты")

    # --- схемы вкладок ------------------------------------------------------
    def _draw_roll(self, cv):
        Y, A, S, V = 130, self._arrow, self._sum, self._val
        self._title(cv, "Боковая САУ: курс χ → крен φ → элероны δa")
        cv.create_text(30, 110, text="χ_ref", font=FONT_B, fill=C_REF)
        V(cv, 30, 150, lambda m: f"{m['chi_ref'] % 360:5.1f}°", C_REF)
        A(cv, 50, Y, 69, Y)
        S(cv, 80, Y)
        A(cv, 91, Y, 135, Y)
        V(cv, 113, 115, lambda m: f"{m['e_chi']:+.1f}°")
        self._box(cv, 135, 55, 275, 205, "ПИ курса")
        A(cv, 275, Y, 315, Y)
        self._box(cv, 315, 75, 425, 185, "огр. φ_ref", C_SAT)
        self._sat_icon(cv, 370, 103)
        A(cv, 425, Y, 469, Y)
        V(cv, 447, 115, lambda m: f"{m['phi_ref']:+.1f}°", C_REF)
        S(cv, 480, Y)
        A(cv, 491, Y, 525, Y)
        V(cv, 508, 115, lambda m: f"{m['e_phi']:+.1f}°")
        self._box(cv, 525, 55, 665, 205, "ПИ крена")
        A(cv, 665, Y, 689, Y)
        S(cv, 700, Y)
        A(cv, 711, Y, 745, Y)
        self._box(cv, 745, 75, 855, 185, "огр. δa", C_SAT)
        self._sat_icon(cv, 800, 103)
        A(cv, 855, Y, 900, Y)
        V(cv, 877, 115, lambda m: f"{m['da']:+.1f}°", C_CTRL)
        self._box(cv, 900, 85, 990, 175, "ЛА")
        cv.create_text(945, 130, text="6DOF\n(3D-окно)", font=FONT, justify="center")
        A(cv, 990, 100, 1045, 100, 1045, 355, 80, 355, 80, 141)          # χ (GPS)
        cv.create_text(1000, 92, text="χ", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 300, 343, lambda m: f"χ = {m['chi'] % 360:5.1f}°", C_MEAS)
        A(cv, 990, Y, 1025, Y, 1025, 320, 480, 320, 480, 141)            # φ (ИНС)
        cv.create_text(1000, 122, text="φ", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 600, 308, lambda m: f"φ = {m['phi']:+.1f}°", C_MEAS)
        A(cv, 990, 160, 1005, 160, 1005, 255, 855, 255)                  # p (гироскоп)
        cv.create_text(995, 152, text="p", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 930, 243, lambda m: f"{m['p']:+.0f}°/с", C_MEAS)
        self._box(cv, 745, 222, 855, 305, "демпф. Kd")
        A(cv, 745, 255, 700, 255, 700, 141)
        V(cv, 675, 270, lambda m: f"Kd·p = {m['kdp']:+.1f}°")
        self._footer(cv, "φ_ref = Kp·e_χ + Ki·∫e_χ;   δa = Kp·e_φ + Ki·∫e_φ − Kd·p.   "
                         "Руль направления — контур β (вкладка «Рыскание»).")

    def _draw_pitch(self, cv):
        Y, A, S, V = 135, self._arrow, self._sum, self._val
        self._title(cv, "Продольная САУ: высота h → тангаж θ → угл. скорость q → руль высоты δe")
        cv.create_text(28, 115, text="h_ref", font=FONT_B, fill=C_REF)
        V(cv, 28, 155, lambda m: f"{m['h_ref']:.0f} м", C_REF)
        A(cv, 45, Y, 59, Y)
        S(cv, 70, Y)
        V(cv, 70, Y - 30, lambda m: f"{m['e_h']:+.1f}")
        A(cv, 81, Y, 105, Y)
        self._box(cv, 105, 75, 205, 195, "П высоты")
        cv.create_text(155, 172, text="+ α_трим\nогр. ±20°", font=FONT_S, justify="center")
        A(cv, 205, Y, 245, Y)
        V(cv, 225, Y - 15, lambda m: f"{m['theta_ref']:+.1f}", C_REF)
        S(cv, 256, Y)
        V(cv, 256, Y - 30, lambda m: f"{m['e_theta']:+.1f}")
        A(cv, 267, Y, 290, Y)
        self._box(cv, 290, 35, 410, 235, "ПИД тангажа")
        A(cv, 410, Y, 435, Y)
        self._box(cv, 435, 85, 520, 185, "огр. q_ref", C_SAT)
        self._sat_icon(cv, 477, 114)
        A(cv, 520, Y, 555, Y)
        V(cv, 537, Y - 15, lambda m: f"{m['q_ref']:+.0f}", C_REF)
        S(cv, 566, Y)
        V(cv, 566, Y - 30, lambda m: f"{m['e_q']:+.1f}")
        A(cv, 577, Y, 600, Y)
        self._box(cv, 600, 35, 720, 235, "ПИД q")
        A(cv, 720, Y, 745, Y)
        self._box(cv, 745, 85, 830, 185, "× −k")
        cv.create_text(787, 112, text="k = (Va₀/Va)²", font=FONT_S)
        V(cv, 787, 170, lambda m: f"k = {m['gs']:.2f}")
        A(cv, 830, Y, 855, Y)
        self._box(cv, 855, 85, 940, 185, "огр. δe", C_SAT)
        self._sat_icon(cv, 897, 114)
        A(cv, 940, Y, 985, Y)
        V(cv, 962, Y - 15, lambda m: f"{m['de']:+.1f}", C_CTRL)
        self._box(cv, 985, 90, 1040, 180, "ЛА")
        cv.create_text(1012, 140, text="6DOF", font=FONT)
        A(cv, 1040, 165, 1053, 165, 1053, 262, 566, 262, 566, 146)       # q (гироскоп)
        cv.create_text(1043, 157, text="q", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 800, 250, lambda m: f"q = {m['q']:+.1f}°/с", C_MEAS)
        A(cv, 1040, Y, 1071, Y, 1071, 292, 256, 292, 256, 146)           # θ (ИНС)
        cv.create_text(1043, 127, text="θ", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 650, 280, lambda m: f"θ = {m['theta']:+.1f}°", C_MEAS)
        A(cv, 1040, 105, 1090, 105, 1090, 322, 70, 322, 70, 146)         # h (баровысотомер)
        cv.create_text(1043, 97, text="h", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 450, 310, lambda m: f"h = {m['h']:.1f} м", C_MEAS)
        self._footer(cv, "θ_ref = α_трим + KH·(h_ref − h);   q_ref = ПИД(θ_ref − θ);   "
                         "δe = −k·ПИД(q_ref − q),  k = (Va₀/Va)² при GS (Va₀ = 30 м/с), иначе k = 1.")

    def _draw_speed(self, cv):
        Y, A, S, V = 150, self._arrow, self._sum, self._val
        self._title(cv, "САУ скорости: воздушная скорость Va → тяга δt")
        cv.create_text(30, 130, text="Va_ref", font=FONT_B, fill=C_REF)
        V(cv, 30, 170, lambda m: f"{m['Va_ref']:.1f}", C_REF)
        A(cv, 55, Y, 69, Y)
        S(cv, 80, Y)
        V(cv, 80, Y - 30, lambda m: f"{m['e_Va']:+.2f}")
        A(cv, 91, Y, 130, Y)
        self._box(cv, 130, 50, 260, 250, "ПИД скорости")
        A(cv, 260, Y, 334, Y)
        V(cv, 297, Y - 15, lambda m: f"{m['dthr']:+.3f}")
        S(cv, 345, Y, marks=(("+", -15, -12), ("+", 15, -16)))
        cv.create_text(345, 62, text="δt_трим (балансировка)", font=FONT)
        V(cv, 345, 80, lambda m: f"{m['thr_trim']:.3f}")
        A(cv, 345, 92, 345, 139)
        A(cv, 356, Y, 420, Y)
        self._box(cv, 420, 100, 520, 200, "огр. 0…1", C_SAT)
        self._sat_icon(cv, 470, 150)
        A(cv, 520, Y, 600, Y)
        V(cv, 560, Y - 15, lambda m: f"{m['thr']:.3f}", C_CTRL)
        self._box(cv, 600, 110, 690, 190, "ЛА")
        cv.create_text(645, 155, text="6DOF", font=FONT)
        A(cv, 690, Y, 740, Y, 740, 300, 80, 300, 80, 161)                # Va (ПВД)
        cv.create_text(700, 142, text="Va", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 400, 288, lambda m: f"Va = {m['Va']:.2f} м/с", C_MEAS)
        cv.create_text(780, 110, anchor="nw", font=FONT, fill="#555", justify="left",
                       text="Высота держится рулём высоты\n(вкладка «Тангаж / высота»).\n\n"
                            "Тяга меняет и скорость, и высоту:\n"
                            "ступенька Va_ref видна на обоих\n"
                            "графиках внизу.")
        self._footer(cv, "δt = δt_трим + ПИД(Va_ref − Va),  ограничение 0…1.   "
                         "δt_трим — балансировочная тяга при Va = 30 м/с (упреждение).")

    def _draw_yaw(self, cv):
        Y, A, S, V = 150, self._arrow, self._sum, self._val
        self._title(cv, "САУ рыскания: скольжение β → руль направления δr (координация разворота)")
        cv.create_text(40, 130, text="β_ref = 0", font=FONT_B, fill=C_REF)
        A(cv, 65, Y, 84, Y)
        S(cv, 95, Y)
        A(cv, 106, Y, 150, Y)
        V(cv, 128, Y - 15, lambda m: f"{-m['beta']:+.2f}°")
        self._box(cv, 150, 55, 280, 230, "ПИ скольжения")
        A(cv, 280, Y, 330, Y)
        self._box(cv, 330, 125, 380, 175, "× −1")
        A(cv, 380, Y, 430, Y)
        self._box(cv, 430, 100, 530, 200, "огр. δr", C_SAT)
        self._sat_icon(cv, 480, 125)
        A(cv, 530, Y, 600, Y)
        V(cv, 565, Y - 15, lambda m: f"{m['dr']:+.2f}°", C_CTRL)
        self._box(cv, 600, 110, 690, 190, "ЛА")
        cv.create_text(645, 155, text="6DOF", font=FONT)
        A(cv, 690, Y, 740, Y, 740, 300, 95, 300, 95, 161)                # β (зонд УС)
        cv.create_text(700, 142, text="β", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 400, 288, lambda m: f"β = {m['beta']:+.2f}°", C_MEAS)
        cv.create_text(780, 110, anchor="nw", font=FONT, fill="#555", justify="left",
                       text="β > 0 — поток справа; δr > 0 — нос влево,\n"
                            "поэтому δr = −(Kp·β + Ki·∫β).\n\n"
                            "β рождается в развороте: дайте ступеньку\n"
                            "курса и сравните с выключенным контуром.\n"
                            "Толчок δr — возмущение по рысканию.")
        self._footer(cv, "δr = −(Kp·β + Ki·∫β);  флажок выкл — руль направления в нейтрали (δr = 0). "
                         "β измеряется истинный (без шума).")

    def _draw_prot(self, cv):
        A, V = self._arrow, self._val
        self._title(cv, "Автомат защиты от выхода на закритические углы атаки (отключаемый)")
        cv.create_text(50, 92, text="θ_ref", font=FONT_B, fill=C_REF)
        cv.create_text(50, 108, text="(контур h)", font=FONT_S)
        V(cv, 50, 126, lambda m: f"{m['theta_cmd']:+.1f}°", C_REF)
        A(cv, 85, 100, 130, 100)

        # автомат: диаграмма состояний
        self._box(cv, 130, 40, 560, 245, "Автомат защиты по α")
        nodes = {0: (195, 110, "НОРМ", "—"), 1: (345, 110, "ПРЕД", "θ_ref + Δθ"),
                 2: (495, 110, "КРИТ", "θ_восст, δt_восст"), 3: (495, 205, "ВОССТ", "θ_восст, δt_восст")}
        rects = {}
        for k, (x, y, name, act) in nodes.items():
            hw = 45 if k < 2 else 52                      # КРИТ/ВОССТ шире — длинная подпись
            rects[k] = cv.create_rectangle(x - hw, y - 20, x + hw, y + 20, outline=C_LINE, width=1.5,
                                           fill="white")
            cv.create_text(x, y - 7, text=name, font=FONT_B)
            cv.create_text(x, y + 9, text=act, font=FONT_S)
        sm = ("Segoe UI", 7)
        A(cv, 240, 103, 300, 103); cv.create_text(270, 92, text="α ≥ α_пред", font=sm)
        A(cv, 300, 117, 240, 117); cv.create_text(270, 128, text="α < α_пред", font=sm)
        A(cv, 390, 110, 443, 110); cv.create_text(420, 99, text="α ≥ α_крит", font=sm)
        A(cv, 485, 130, 485, 185); cv.create_text(482, 158, text="α < α_крит", font=sm, anchor="e")
        A(cv, 505, 185, 505, 130); cv.create_text(509, 158, text="рецидив", font=sm, anchor="w")
        A(cv, 443, 205, 195, 205, 195, 130); cv.create_text(320, 196, text="α < α_выход", font=sm)
        off = cv.create_text(345, 232, text="", font=FONT_B, fill="#c33")
        colors = {0: "#c8f0c8", 1: "#ffe9a8", 2: "#ffc58a", 3: "#c6d8ff"}

        def states(cv, m):
            for k, r in rects.items():
                on = m["prot_on"] and m["prot_state"] == k
                cv.itemconfigure(r, fill=colors[k] if on else ("white" if m["prot_on"] else "#e4e4e4"))
            cv.itemconfigure(off, text="" if m["prot_on"] else "ЗАЩИТА ВЫКЛЮЧЕНА")
        self._dyn(cv, states)

        # выходы автомата
        A(cv, 560, 85, 640, 85)
        cv.create_text(600, 74, text="θ_ref*", font=FONT_B, fill=C_REF)
        V(cv, 600, 98, lambda m: f"{m['theta_ref']:+.1f}°", C_REF)
        self._box(cv, 640, 55, 770, 120, "ПИД тангажа")
        cv.create_text(705, 100, text="(вкладка «Тангаж»)", font=FONT_S)
        A(cv, 770, 85, 820, 85)
        self._box(cv, 640, 140, 770, 195, "САУ скорости")
        cv.create_text(705, 177, text="(вкладка «Скорость»)", font=FONT_S)
        A(cv, 770, 165, 820, 165)
        A(cv, 560, 222, 820, 222)
        cv.create_text(690, 211, text="δt_восст — только КРИТ / ВОССТ", font=FONT_S)
        V(cv, 795, 152, lambda m: f"δt {m['thr']:.2f}", C_CTRL)
        self._box(cv, 820, 55, 890, 245, "ЛА")
        cv.create_text(855, 150, text="6DOF", font=FONT)
        A(cv, 890, 150, 912, 150, 912, 268, 345, 268, 345, 246)        # α (истинный)
        cv.create_text(895, 142, text="α", font=FONT_B, fill=C_MEAS, anchor="w")
        V(cv, 620, 280, lambda m: f"α = {m['alpha']:+.1f}° (истинный)", C_MEAS)

        # настройки
        self._box(cv, 925, 40, 1095, 300, "Настройки")

        # шкала α
        g0, g1, gy = 130, 890, 312
        a_lo, a_hi = -5.0, 35.0
        gx = lambda a: g0 + (min(max(a, a_lo), a_hi) - a_lo) / (a_hi - a_lo) * (g1 - g0)
        cv.create_text(70, gy + 8, text="шкала α", font=FONT_B)
        zones = [cv.create_rectangle(0, gy, 0, gy + 16, width=0) for _ in range(4)]
        cv.create_rectangle(g0, gy, g1, gy + 16, outline=C_LINE)
        for a in range(-5, 36, 5):
            cv.create_line(gx(a), gy + 16, gx(a), gy + 21, fill=C_LINE)
            cv.create_text(gx(a), gy + 29, text=f"{a}°", font=FONT_S)
        needle = cv.create_polygon(0, 0, 0, 0, 0, 0, fill="black")
        ny = cv.create_text(1010, gy + 10, text="", font=("Consolas", 11, "bold"))

        def gauge(cv, m):
            edges = [a_lo, m["a_warn"], m["a_crit"], m["a_stall"], a_hi]
            for z, c, a, b in zip(zones, ("#c8f0c8", "#ffe9a8", "#ffc58a", "#ff9c9c"), edges, edges[1:]):
                cv.coords(z, gx(a), gy, gx(b), gy + 16)
                cv.itemconfigure(z, fill=c)
            x = gx(m["alpha"])
            cv.coords(needle, x, gy + 15, x - 6, gy - 6, x + 6, gy - 6)
            cv.itemconfigure(ny, text=f"n_y = {m['ny']:+.2f}")
        self._dyn(cv, gauge)
        self._footer(cv, "Зоны шкалы: норма / предупреждение (α_пред) / критический (α_крит) / срыв "
                         "(α_срыв из параметров ЛА).  «Срыв» без защиты — сваливание; R в 3D-окне — сброс.")

    def _draw_nav(self, cv):
        A, V = self._arrow, self._val
        MODES = {"line": "прямая", "fillet": "дуга скругления", "orbit": "кружение", "—": "выкл"}
        self._title(cv, "Навигация по точкам: маршрут → менеджер маршрута → следование → χ_ref, h_ref")
        self._box(cv, 15, 95, 140, 185, "Маршрут")
        cv.create_text(77, 135, text="точки w₁…w_N\n(окно карты)", font=FONT, justify="center")
        V(cv, 77, 170, lambda m: f"N = {m['nav_n']}")
        A(cv, 140, 140, 175, 140)
        self._box(cv, 175, 35, 415, 275, "Менеджер маршрута")
        V(cv, 295, 228, lambda m: "режим: " + MODES.get(m["nav_mode"], m["nav_mode"]))
        V(cv, 295, 250, lambda m: (f"цель {m['nav_idx']}/{m['nav_n']}, до неё {m['nav_dist']:.0f} м"
                                    if m["nav_on"] and m["nav_mode"] != "orbit" else ""))
        A(cv, 415, 140, 465, 140)
        cv.create_text(440, 120, text="участок\nили круг", font=FONT_S, justify="center")
        self._box(cv, 465, 35, 735, 275, "Следование по прямой / окружности")
        V(cv, 600, 236, lambda m: f"e_py = {m['e_py']:+.1f} м", C_MEAS)
        V(cv, 600, 256, lambda m: f"χ_q = {m['chi_q']:5.1f}°")
        A(cv, 735, 100, 800, 100)
        cv.create_text(767, 88, text="χ_ref", font=FONT_B, fill=C_REF)
        V(cv, 767, 114, lambda m: f"{m['chi_ref'] % 360:5.1f}°", C_REF)
        A(cv, 735, 200, 800, 200)
        cv.create_text(767, 188, text="h_ref", font=FONT_B, fill=C_REF)
        V(cv, 767, 214, lambda m: f"{m['h_ref']:.0f} м", C_REF)
        self._box(cv, 800, 70, 945, 130, "Курс → крен → δa")
        cv.create_text(872, 112, text="(вкладка «Крен»)", font=FONT_S)
        self._box(cv, 800, 170, 945, 230, "Высота → θ → δe")
        cv.create_text(872, 212, text="(вкладка «Тангаж»)", font=FONT_S)
        A(cv, 945, 100, 985, 100)
        A(cv, 945, 200, 985, 200)
        self._box(cv, 985, 70, 1065, 230, "ЛА")
        cv.create_text(1025, 150, text="6DOF", font=FONT)
        A(cv, 1025, 230, 1025, 318, 295, 318, 295, 276)                  # положение (GPS)
        A(cv, 600, 318, 600, 276)
        cv.create_text(820, 306, text="p = (N, E) — GPS", font=FONT_B, fill=C_MEAS)
        off = cv.create_text(600, 340, text="", font=FONT_B, fill="#c33")
        self._dyn(cv, lambda c, m: c.itemconfigure(
            off, text="" if m["nav_on"] else "НАВИГАЦИЯ ВЫКЛ — поставьте точки на карте и «Лететь»"))
        self._footer(cv, "Прямая: χ_ref = χ_q − χ∞·(2/π)·atan(k_path·e_py).   Окружность: "
                         "χ_ref = ∠(p − c) + λ(π/2 + atan(k_orbit(d − ρ)/ρ)).   Углы — дугой R скругления.")

    # --- ползунки и кнопки ----------------------------------------------------
    def _controls(self, tab, cv):
        for t, key, label, lo, hi, res, default in self.spec:
            if t != tab:
                continue
            x, y, length, bg = SLIDER_POS[key]
            if res is None:                                   # флажок
                v = tk.BooleanVar(value=default)
                w = tk.Checkbutton(cv, text=label, variable=v, font=FONT, bg=bg,
                                   activebackground=bg, command=self._send_params)
            else:
                v = tk.DoubleVar(value=default)
                w = tk.Scale(cv, variable=v, from_=lo, to=hi, resolution=res, orient="horizontal",
                             length=length, label=label, font=FONT, bg=bg, bd=0,
                             highlightthickness=0, command=lambda _v: self._send_params())
            self.vars[key] = v
            cv.create_window(x, y, window=w, anchor="n")

    def _buttons(self, tab, bar):
        def group(title, kind, steps, unit):
            tk.Label(bar, text=f"  {title}:", font=FONT, bg="white").pack(side="left")
            for d in steps:
                tk.Button(bar, text=f"{d:+g}{unit}", font=FONT, width=5,
                          command=lambda d=d: self._put((kind, d))).pack(side="left", padx=2, pady=4)

        def kick(name, text, kind="kick"):
            tk.Button(bar, text=text, font=FONT,
                      command=lambda: self._put((kind, name))).pack(side="left", padx=(14, 2))

        if tab in ("roll", "yaw"):
            group("Ступенька χ_ref", "chi", (-90, -30, -10, 10, 30, 90), "°")
        if tab == "roll":
            kick("da", "Толчок δa 1 с")
        if tab == "pitch":
            group("Ступенька h_ref", "h", (-50, -10, 10, 50), " м")
            kick("de", "Толчок δe 1 с")
        if tab == "speed":
            group("Ступенька Va_ref", "va", (-5, -1, 1, 5), "")
        if tab == "yaw":
            kick("dr", "Толчок δr 1 с")
        if tab == "prot":
            kick(None, "Срыв: δt = 0, θ_ref = +15°", "stall")
            kick(None, "Отмена срыва", "stall_off")
        if tab == "nav":
            kick(None, "Лететь по маршруту", "go")
            kick(None, "Стоп (держать курс и высоту)", "stop")
        tk.Button(bar, text="Параметры по умолчанию", font=FONT,
                  command=lambda: self._defaults(tab)).pack(side="right", padx=8)

    def _defaults(self, tab):
        for t, key, *_, default in self.spec:
            if t == tab:
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
            if self.data and m["t"] < self.data[-1]["t"]:     # сброс полёта (R)
                self.data.clear()
            self.data.append(m)
            self.last = m
        while self.data and self.data[-1]["t"] - self.data[0]["t"] > SCOPE_WIN:
            self.data.popleft()
        tab = TABS[self.nb.index("current")][0]               # рисуем только видимую вкладку
        cv, sc, vals = self.tabs[tab]
        if self.last:
            for item, fmt in vals:
                if item is None:
                    fmt(cv, self.last)
                else:
                    cv.itemconfigure(item, text=fmt(self.last))
        self._scope(sc, SCOPES[tab])
        self.root.after(TICK_MS, self._tick)

    def _scope(self, sc, strips):
        sc.delete("all")
        x0, x1 = 70, W - 15
        h = (H_SCOPE - 30) / len(strips)
        d = list(self.data)
        t_end = d[-1]["t"] if d else 0.0
        t_beg = t_end - SCOPE_WIN if t_end > SCOPE_WIN else 0.0
        for i, (title, lines, scale) in enumerate(strips):
            y0, y1 = 10 + i * h, 10 + i * h + h - 22
            vals = [r[k] for r in d for k, *_ in lines]
            if scale[0] == "sym":                           # симметричная шкала
                a = max([scale[1]] + [abs(v) * 1.1 for v in vals])
                lo, hi = -a, a
            elif scale[0] == "fix":
                lo, hi = scale[1], scale[2]
            else:
                lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
                mid, half = (lo + hi) / 2, max((hi - lo) / 2 * 1.15, scale[1] / 2)
                lo, hi = mid - half, mid + half
            ty = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0 - 18)   # сверху место под заголовок
            tx = lambda t: x0 + (t - t_beg) / SCOPE_WIN * (x1 - x0)
            sc.create_rectangle(x0, y0, x1, y1, outline="#bbb")
            for k in range(5):
                v = lo + (hi - lo) * k / 4
                sc.create_line(x0, ty(v), x1, ty(v), fill="#eee")
                txt = f"{v:.2f}" if hi - lo < 4 else f"{v:.0f}"
                sc.create_text(x0 - 4, ty(v), text=txt, anchor="e", font=FONT_S)
            if lo < 0 < hi:
                sc.create_line(x0, ty(0), x1, ty(0), fill="#ccc")
            sc.create_text(x0 + 6, y0 + 8, text=title, anchor="w", font=FONT_B)
            lx = x1 - 10
            for key, color, name, dashed in reversed(lines):
                sc.create_text(lx, y0 + 8, text=name, anchor="e", font=FONT_B, fill=color)
                lx -= 60
                if len(d) > 1:
                    pts = []
                    for r in d:
                        pts += (tx(r["t"]), ty(r[key]))
                    sc.create_line(*pts, fill=color, width=1.5 if dashed else 2,
                                   dash=(6, 3) if dashed else ())
        for k in range(0, int(SCOPE_WIN) + 1, 5):          # подписи времени
            sc.create_text(x0 + k / SCOPE_WIN * (x1 - x0), H_SCOPE - 12,
                           text=f"{t_beg + k:.0f} с", font=FONT_S)


def run(q_par, q_tel, spec, geometry="-0+0"):
    """Точка входа процесса окна стенда."""
    root = tk.Tk()
    root.configure(bg="white")
    root.geometry(geometry)
    Panel(root, q_par, q_tel, spec)
    root.mainloop()
