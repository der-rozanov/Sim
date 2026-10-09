# -*- coding: utf-8 -*-
"""
Окно редактора структурных схем САУ (лабораторная «Конструктор САУ»,
scenarios/BlockLab3D.py). Работает в ОТДЕЛЬНОМ процессе (tkinter) рядом с 3D-окном.

    слева   — библиотека блоков: щелчок по блоку, затем щелчок на схеме — блок поставлен;
    схема   — блоки перетаскиваются мышью; провод тянется от выхода блока (●)
              к входу другого (▶) или наоборот; провод, взятый за вход, переносится;
              провода прокладываются сами (в обход блоков, от одного выхода — общим
              стволом с точками ветвления); вертикальный участок провода (у обратной
              связи — и нижний) перетаскивается мышью, двойной щелчок по проводу —
              снова автоматически;
              двойной щелчок по блоку — параметры; Delete — удалить выделенное;
              правая кнопка — меню блока; на проводах — живые значения сигналов;
    снизу   — осциллограф: по полосе на каждый блок «Осциллограф» (до 4).

Связь — две очереди multiprocessing (см. scenarios/BlockLab3D.py):
    q_cmd  (окно → 3D): ("scheme", схема) | ("on", bool) | уставки и толчки
    q_tel  (3D → окно): dict — состояние, выходы блоков, входы осциллографов

Физику и САУ не импортирует: библиотека блоков приходит описанием
(control.blocks.catalog()), схема уходит словарём, проверяет и считает её 3D-процесс.
"""

import copy
import json
import os
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import deque

W_SCHEME, H_SCHEME = 960, 460          # видимая часть схемы, пикс
SCROLL = (0, 0, 1900, 1200)             # размер холста схемы, пикс
W_SCOPE, H_SCOPE = 960, 290
SCOPE_WIN = 20.0                        # окно осциллографа, с
TICK_MS = 40                            # период обновления окна, мс (~25 Гц)
GRID = 10                               # шаг привязки блоков, пикс

BW = 112                                # ширина блока, пикс
BW_TYPE = {"Sum": 46, "Product": 56, "Gain": 80, "Input": 100, "Output": 100, "Const": 90,
           "AngleErr": 128, "VaScale": 116, "LeadLag": 128}
# Трассировка проводов
STUB = 6                                # мин. отступ излома от порта, пикс
CLEAR = 4                               # зазор провода до блока, пикс
LANE = 8                                # расстояние между параллельными участками, пикс
N_CAND = 60                             # вариантов положения излома при подборе
PORT = 22                               # шаг входов по высоте, пикс

FONT = ("Segoe UI", 9)
FONT_B = ("Segoe UI", 9, "bold")
FONT_S = ("Segoe UI", 8)
FONT_V = ("Consolas", 8)
C_LINE, C_SEL, C_WIRE = "#333", "#d62728", "#555"
C_GROUP = {"Источники": "#e3f1e3", "Регуляторы": "#e6eefb", "Математика": "#f4f4f4",
           "Нелинейности": "#fff4e0", "Динамические звенья": "#f3e8fb", "Приёмники": "#fde8e8"}
C_TRACE = ["#d62728", "#1f77b4", "#2ca02c"]


def _fmt(v):
    """Число для подписи: коротко, без лишних нулей."""
    if v is None:
        return "—"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{v:.2e}"
    return f"{v:.4g}"


class Editor:
    def __init__(self, root, q_cmd, q_tel, cat, templates, start):
        self.root, self.q_cmd, self.q_tel = root, q_cmd, q_tel
        self.cat, self.templates = cat, templates
        self.lib = cat["blocks"]
        self.blocks, self.wires = {}, []
        self.sel = None                     # ("blk", id) | ("wire", индекс)
        self.placing = None                 # тип блока, который ставим щелчком
        self.drag = None
        self.applied = None                 # принятая 3D-процессом схема (копия)
        self.sent, self.version = None, 0   # отправленная схема; номер принятой
        self.tel = {}
        self.data = deque()                 # (t, {id осциллографа: [входы]})
        self.val_items = {}                 # id блока → подпись значения на выходе
        self.file = None

        root.title("Конструктор САУ — редактор схемы")
        self._build()
        self._load(start)
        self.root.after(TICK_MS, self._tick)

    # =====================================================================
    # Окно
    # =====================================================================
    def _build(self):
        r = self.root
        bar = tk.Frame(r, bg="white")
        bar.pack(fill="x", padx=4, pady=(4, 0))
        tk.Button(bar, text="▶ Применить (Ctrl+Enter)", font=FONT_B, bg="#dff0d8",
                  command=self._apply).pack(side="left")
        self.btn_on = tk.Button(bar, text="Схема: ВЫКЛ", font=FONT_B, width=14, command=self._toggle_on)
        self.btn_on.pack(side="left", padx=6)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        for name in self.templates:
            tk.Button(bar, text=name, font=FONT, command=lambda n=name: self._template(n)).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        tk.Button(bar, text="Открыть…", font=FONT, command=self._open).pack(side="left")
        tk.Button(bar, text="Сохранить…", font=FONT, command=self._save).pack(side="left", padx=2)
        self.lbl_status = tk.Label(bar, text="", font=FONT_B, bg="white", anchor="w")
        self.lbl_status.pack(side="left", padx=10, fill="x", expand=True)

        bar2 = tk.Frame(r, bg="white")
        bar2.pack(fill="x", padx=4, pady=2)
        tk.Label(bar2, text="Уставки:", font=FONT, bg="white").pack(side="left")
        for text, kind, val in [("θ_ref −5°", "theta", -5.0), ("θ_ref +5°", "theta", 5.0),
                                ("h_ref −10 м", "h", -10.0), ("h_ref +10 м", "h", 10.0),
                                ("Va_ref −1", "va", -1.0), ("Va_ref +1", "va", 1.0),
                                ("χ_ref −30°", "chi", -30.0), ("χ_ref +30°", "chi", 30.0),
                                ("Толчок δe", "kick", "de")]:
            tk.Button(bar2, text=text, font=FONT, command=lambda k=kind, v=val: self._send(k, v)
                      ).pack(side="left", padx=1)
        ttk.Separator(bar2, orient="vertical").pack(side="left", fill="y", padx=6)
        tk.Label(bar2, text="Режим САУ:", font=FONT, bg="white").pack(side="left")
        tk.Button(bar2, text="высота", font=FONT, command=lambda: self._send("mode", "alt")).pack(side="left")
        tk.Button(bar2, text="тангаж", font=FONT, command=lambda: self._send("mode", "pitch")).pack(side="left")
        self.lbl_flight = tk.Label(r, text="", font=FONT_V, bg="white", fg="#444", anchor="w")
        self.lbl_flight.pack(fill="x", padx=6)

        mid = tk.Frame(r, bg="white")
        mid.pack(fill="both", expand=True)
        self._palette(mid)
        box = tk.Frame(mid, bg="white")
        box.pack(side="left", fill="both", expand=True)
        self.cv = tk.Canvas(box, width=W_SCHEME, height=H_SCHEME, bg="white",
                            scrollregion=SCROLL, highlightthickness=1, highlightbackground="#ccc")
        sy = tk.Scrollbar(box, orient="vertical", command=self.cv.yview)
        sx = tk.Scrollbar(box, orient="horizontal", command=self.cv.xview)
        self.cv.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.cv.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        tk.Label(box, font=FONT_S, bg="white", fg="#777", anchor="w", justify="left",
                 wraplength=W_SCHEME, text=(
            "Блок — перетащить мышью · провод — от выхода ● к входу ▶ (провод за вход — перенести) · "
            "участок провода — перетащить (пунктир — проложен вручную), двойной щелчок — снова авто · "
            "двойной щелчок по блоку — параметры · Delete — удалить · правая кнопка — меню · "
            "колесо / Shift+колесо — прокрутка")).grid(row=2, column=0, sticky="w")

        self.sc = tk.Canvas(r, width=W_SCOPE + 190, height=H_SCOPE, bg="white", highlightthickness=0)
        self.sc.pack(fill="x")

        cv = self.cv
        cv.bind("<ButtonPress-1>", self._press)
        cv.bind("<B1-Motion>", self._motion)
        cv.bind("<ButtonRelease-1>", self._release)
        cv.bind("<Double-Button-1>", self._dbl)
        cv.bind("<Button-3>", self._menu)
        cv.bind("<Motion>", self._hover)
        cv.bind("<MouseWheel>", lambda e: cv.yview_scroll(-e.delta // 120, "units"))
        cv.bind("<Shift-MouseWheel>", lambda e: cv.xview_scroll(-e.delta // 120, "units"))
        r.bind("<Delete>", lambda e: self._delete_sel())
        r.bind("<Escape>", lambda e: self._cancel_place())
        r.bind("<Control-Return>", lambda e: self._apply())
        r.bind("<Control-s>", lambda e: self._save())
        r.bind("<Control-o>", lambda e: self._open())

    def _palette(self, parent):
        f = tk.Frame(parent, bg="white")
        f.pack(side="left", fill="y", padx=(4, 2))
        groups = {}
        for name, d in self.lib.items():
            groups.setdefault(d["group"], []).append(name)
        row = 0
        for grp, names in groups.items():                # группа — заголовок и кнопки в 2 колонки
            tk.Label(f, text=grp, font=FONT_B, bg="white", anchor="w").grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
            row += 1
            for k, name in enumerate(names):
                tk.Button(f, text=self.lib[name]["title"], font=FONT_S, anchor="w", pady=0,
                          bg=C_GROUP.get(grp, "white"), relief="groove", width=13,
                          command=lambda n=name: self._start_place(n)
                          ).grid(row=row + k // 2, column=k % 2, sticky="ew")
            row += (len(names) + 1) // 2

    # =====================================================================
    # Модель схемы
    # =====================================================================
    def _n_in(self, b):
        d = self.lib[b["type"]]
        if b["type"] == "Sum":
            return len([c for c in str(b["p"].get("signs", "")) if c in "+-−"])
        if b["type"] == "Scope":
            try:
                return min(max(int(b["p"].get("n", 1)), 1), 3)
            except ValueError:
                return 1
        return len(d["inputs"])

    def _in_label(self, b, i):
        if b["type"] == "Sum":
            c = [c for c in str(b["p"]["signs"]) if c in "+-−"][i]
            return "+" if c == "+" else "−"
        ins = self.lib[b["type"]]["inputs"]
        return ins[i] if ins and i < len(ins) else ""

    def _has_out(self, b):
        return b["type"] not in ("Output", "Scope")

    def _size(self, b):
        n = self._n_in(b)
        return BW_TYPE.get(b["type"], BW), max(48, PORT * n + 20)

    def _in_pos(self, b, i):
        w, h = self._size(b)
        return b["x"], b["y"] + h * (i + 1) / (self._n_in(b) + 1)

    def _out_pos(self, b):
        w, h = self._size(b)
        return b["x"] + w, b["y"] + h / 2

    def _new_id(self, btype):
        base = btype.lower()
        k = 1
        while f"{base}{k}" in self.blocks:
            k += 1
        return f"{base}{k}"

    def _load(self, scheme):
        self.blocks = {b["id"]: {"type": b["type"], "x": b["x"], "y": b["y"],
                                 "p": dict(b.get("p", {}))} for b in scheme.get("blocks", [])
                       if b["type"] in self.lib}
        for b in self.blocks.values():          # недостающие параметры — по умолчанию
            for key, _, default, _ in self.lib[b["type"]]["params"]:
                b["p"].setdefault(key, default)
        self.wires = [dict(w) for w in scheme.get("wires", [])
                      if w["src"] in self.blocks and w["dst"] in self.blocks]
        self.sel = None
        self._changed()

    def _scheme(self):
        return {"blocks": [{"id": bid, **copy.deepcopy(b)} for bid, b in self.blocks.items()],
                "wires": [dict(w) for w in self.wires]}

    def _changed(self):
        self._redraw()

    @staticmethod
    def _content(scheme):
        """Содержание схемы без раскладки (места блоков и проводов на холсте)."""
        if scheme is None:
            return None
        return (sorted((b["id"], b["type"], json.dumps(b["p"], sort_keys=True))
                       for b in scheme["blocks"]),
                sorted((w["src"], w["dst"], w["port"]) for w in scheme["wires"]))

    @property
    def dirty(self):
        """Схема отличается от принятой 3D-процессом (раскладка не в счёт)."""
        return self._content(self._scheme()) != self._content(self.applied)

    def _prune(self, bid):
        """Провода к несуществующим входам блока (после смены числа входов)."""
        n = self._n_in(self.blocks[bid])
        self.wires = [w for w in self.wires if not (w["dst"] == bid and w["port"] >= n)]

    def _connect(self, src, dst, port):
        self.wires = [w for w in self.wires if not (w["dst"] == dst and w["port"] == port)]
        self.wires.append({"src": src, "dst": dst, "port": port})
        self._changed()

    # =====================================================================
    # Рисование схемы
    # =====================================================================
    def _summary(self, b):
        t, p, lib = b["type"], b["p"], self.lib[b["type"]]
        if t == "Input":
            return self.cat["inputs"].get(p["signal"], p["signal"])
        if t == "Output":
            return "→ " + self.cat["outputs"].get(p["act"], p["act"]).split(" — ")[0]
        if t == "Sat":
            return f"[{_fmt(float(p['min']))} … {_fmt(float(p['max']))}]"
        if t in ("Sum", "Product", "AngleErr"):
            return ""
        if t == "Scope":
            return str(p.get("title", ""))
        items = []
        for key, label, _, kind in lib["params"]:
            if kind not in ("num", "int"):
                continue
            try:
                v = float(p[key])
            except (TypeError, ValueError):
                v = None
            if key == "y_max" and not v:
                continue
            items.append(f"{key}={_fmt(v)}")
        lines = [" ".join(items[i:i + 2]) for i in range(0, len(items), 2)]
        return "\n".join(lines)

    # --- трассировка проводов ------------------------------------------------
    #
    # Провод из 3 участков («ступенька»): выход → излом xm → вход. Провода от одного
    #   выхода идут общим стволом (один xm) — на ветвлениях точки.
    # Провод из 5 участков («обход»): выход → xa → канал yb → xb → вход. Так идут
    #   обратные связи (вход левее выхода) и провода, которым ступенька режет блок;
    #   канал yb — над / под блоками или в промежутке между рядами блоков.
    # Изломы подбираются перебором вариантов; цена варианта: пересечение блока ×1000,
    #   наложение на чужой параллельный участок ×300, пересечение чужого провода ×40,
    #   длина / 20. Провода прокладываются по очереди — каждый учитывает уже проложенные.
    # Ручная правка — поля провода xm или xa, yb, xb (сохраняются в JSON схемы).

    def _rects(self):
        out = {}
        for bid, b in self.blocks.items():
            w, h = self._size(b)
            out[bid] = (b["x"], b["y"], b["x"] + w, b["y"] + h)
        return out

    @staticmethod
    def _hits(pts, rects, skip):
        """Сколько раз ломаная pts пересекает блоки (кроме skip)."""
        n = 0
        for k in range(0, len(pts) - 2, 2):
            x0, x1 = sorted((pts[k], pts[k + 2]))
            y0, y1 = sorted((pts[k + 1], pts[k + 3]))
            for bid, (rx0, ry0, rx1, ry1) in rects.items():
                if bid not in skip and rx0 - CLEAR < x1 and x0 < rx1 + CLEAR \
                        and ry0 - CLEAR < y1 and y0 < ry1 + CLEAR:
                    n += 1
        return n

    @staticmethod
    def _segs(pts):
        """Участки ломаной: [(x0, y0, x1, y1)], без вырожденных."""
        return [tuple(pts[k:k + 4]) for k in range(0, len(pts) - 2, 2)
                if (pts[k], pts[k + 1]) != (pts[k + 2], pts[k + 3])]

    def _wire_cost(self, pts, i, rects, placed):
        """Цена трассы pts провода i среди блоков и уже проложенных проводов."""
        w = self.wires[i]
        cost = 1000 * self._hits(pts, rects, (w["src"], w["dst"]))
        mine = self._segs(pts)
        length = 0
        for x0, y0, x1, y1 in mine:
            length += abs(x1 - x0) + abs(y1 - y0)
            hor = y0 == y1
            for other, owner in placed:
                if owner == w["src"]:                # свой выход — общий ствол, не штраф
                    continue
                for u0, v0, u1, v1 in other:
                    if (v0 == v1) == hor:            # параллельные: наложение
                        if hor and abs(v0 - y0) < LANE - 1 and \
                                min(max(x0, x1), max(u0, u1)) - max(min(x0, x1), min(u0, u1)) > 2:
                            cost += 300
                        elif not hor and abs(u0 - x0) < LANE - 1 and \
                                min(max(y0, y1), max(v0, v1)) - max(min(y0, y1), min(v0, v1)) > 2:
                            cost += 300
                    else:                            # перпендикулярные: пересечение
                        hx0, hx1, hy = (min(x0, x1), max(x0, x1), y0) if hor else (min(u0, u1), max(u0, u1), v0)
                        vy0, vy1, vx = (min(v0, v1), max(v0, v1), u0) if hor else (min(y0, y1), max(y0, y1), x0)
                        if hx0 < vx < hx1 and vy0 < hy < vy1:
                            cost += 40
        return cost + length / 20

    def _route_all(self):
        """Трассы всех проводов: {индекс: (вид "fwd" | "five", точки)} и точки ветвления."""
        rects = self._rects()
        ends = []
        for w in self.wires:
            x1, y1 = self._out_pos(self.blocks[w["src"]])
            x2, y2 = self._in_pos(self.blocks[w["dst"]], w["port"])
            ends.append((x1, y1, x2, y2))
        routes, placed = {}, []                      # placed: (участки, выход-владелец)

        def put(i, kind, pts):
            routes[i] = (kind, pts)
            placed.append((self._segs(pts), self.wires[i]["src"]))

        def fwd(i, xm):
            x1, y1, x2, y2 = ends[i]
            return [x1, y1, xm, y1, xm, y2, x2, y2]

        def five(i, xa_auto):
            """Обход: xa, xb — ручные или по умолчанию, канал yb — ручной или лучший."""
            w = self.wires[i]
            x1, y1, x2, y2 = ends[i]
            xa = max(w.get("xa", xa_auto), x1 + STUB)
            xb = min(w.get("xb", x2 - 2 * STUB), x2 - STUB)
            mk = lambda y: [x1, y1, xa, y1, xa, y, xb, y, xb, y2, x2, y2]
            if "yb" in w:
                return mk(w["yb"])
            lo_x, hi_x = min(xa, xb), max(xa, xb)
            ys = {y1, y2}
            for r in rects.values():                 # каналы над и под блоками на пути
                if r[0] - CLEAR < hi_x and lo_x < r[2] + CLEAR:
                    for k in range(4):
                        ys.update((r[1] - 2 * LANE - k * LANE, r[3] + 2 * LANE + k * LANE))
            best = min((self._wire_cost(mk(y), i, rects, placed), y) for y in ys if y >= 0)
            return mk(best[1])

        manual = lambda w: any(k in w for k in ("xa", "yb", "xb"))

        # 1. Ступеньки: группы по выходу; ручной xm — отдельно
        groups, rest = {}, []
        for i, w in enumerate(self.wires):
            x1, y1, x2, y2 = ends[i]
            if manual(w) or x2 - x1 < 2 * STUB:
                rest.append(i)
            elif "xm" in w:                          # ручной излом — в пределах между портами
                put(i, "fwd", fwd(i, min(max(w["xm"], x1 + STUB), x2 - STUB)))
            else:
                groups.setdefault(w["src"], []).append(i)
        trunk = {}                                   # выход → xm его ствола
        for src in sorted(groups, key=lambda s: ends[groups[s][0]][0]):
            idx = groups[src]
            x1 = ends[idx[0]][0]
            lo, hi = x1 + STUB, min(ends[i][2] for i in idx) - STUB
            step = max(2.0, (hi - lo) / N_CAND)
            cand = sorted({round(lo + k * step) for k in range(int((hi - lo) / step) + 1)})
            pref = (lo + hi) / 2
            # ствол — по проводам, которые ступенькой проходят свободно (остальные — в обход)
            best = None
            for xm in cand:
                parts = [(self._wire_cost(fwd(i, xm), i, rects, placed), i) for i in idx]
                free = [c for c, i in parts if c < 1000]
                cost = (sum(min(c, 2000) for c, _ in parts) + 5 * len(parts) * (len(parts) - len(free))
                        + abs(xm - pref) / 10)
                if best is None or cost < best[0]:
                    best = (cost, xm)
            xm = trunk[src] = best[1]
            for i in idx:
                if self._hits(fwd(i, xm), rects, (src, self.wires[i]["dst"])):
                    rest.append(i)
                else:
                    put(i, "fwd", fwd(i, xm))

        # 2. Обходы: обратные связи, провода, которым ступенька режет блок, ручные
        for i in sorted(rest, key=lambda i: ends[i][0]):
            x1 = ends[i][0]
            put(i, "five", five(i, trunk.get(self.wires[i]["src"], x1 + 2 * STUB)))

        # 3. Точки ветвления: где у проводов одного выхода сходятся ≥ 3 направления
        dots = []
        by_src = {}
        for i, w in enumerate(self.wires):
            by_src.setdefault(w["src"], []).append(i)
        for src, idx in by_src.items():
            if len(idx) < 2:
                continue
            segs = [sg for i in idx for sg in self._segs(routes[i][1])]
            pts = {(x, y) for x0, y0, x1, y1 in segs for x, y in ((x0, y0), (x1, y1))}
            for px, py in pts:
                dirs = set()
                for x0, y0, x1, y1 in segs:
                    on = (min(x0, x1) <= px <= max(x0, x1)) and (min(y0, y1) <= py <= max(y0, y1))
                    if not on:
                        continue
                    for ex, ey in ((x0, y0), (x1, y1)):
                        if (ex, ey) != (px, py):
                            dirs.add(((ex > px) - (ex < px), (ey > py) - (ey < py)))
                if len(dirs) >= 3:
                    dots.append((px, py))
        return routes, dots

    def _redraw(self):
        cv = self.cv
        cv.delete("all")
        self.val_items = {}
        self.routes, dots = self._route_all()
        for i, w in enumerate(self.wires):
            kind, pts = self.routes[i]
            sel = self.sel == ("wire", i)
            manual = any(k in w for k in ("xm", "xa", "yb", "xb"))
            cv.create_line(*pts, fill=C_SEL if sel else C_WIRE, width=3 if sel else 1.5,
                           dash=(6, 2) if manual and not sel else (), arrow=tk.LAST,
                           arrowshape=(8, 10, 4), tags=(f"wire:{i}",))
            # по участку — широкая невидимая линия: легче попасть мышью, участок тянется
            for k in range(0, len(pts) - 2, 2):
                cv.create_line(*pts[k:k + 4], fill="", width=9,
                               tags=(f"wire:{i}", f"seg:{i}:{k // 2}"))
        for x, y in dots:
            cv.create_oval(x - 3.5, y - 3.5, x + 3.5, y + 3.5, fill=C_WIRE, outline="")
        for bid, b in self.blocks.items():
            self._draw_block(bid, b)
        self._status()

    def _seg_key(self, i, k):
        """Какой параметр трассы меняет перетаскивание участка k провода i:
        (поле, "x" | "y") или None — участок не двигается (идёт от порта)."""
        kind = self.routes[i][0]
        if kind == "fwd":
            return ("xm", "x") if k == 1 else None
        return {1: ("xa", "x"), 2: ("yb", "y"), 3: ("xb", "x")}.get(k)

    def _draw_block(self, bid, b):
        cv, lib = self.cv, self.lib[b["type"]]
        w, h = self._size(b)
        x, y = b["x"], b["y"]
        sel = self.sel == ("blk", bid)
        tag = (f"blk:{bid}",)
        cv.create_rectangle(x, y, x + w, y + h, fill=C_GROUP.get(lib["group"], "white"),
                            outline=C_SEL if sel else C_LINE, width=2.5 if sel else 1.2, tags=tag)
        title = lib["title"]
        if b["type"] == "Sum":
            title = "Σ"
        elif b["type"] == "Product":
            title = "×"
        cv.create_text(x + w / 2, y + 11, text=title, font=FONT_B, tags=tag)
        cv.create_text(x + w / 2, y + 13 + (h - 13) / 2, text=self._summary(b), font=FONT_S,
                       fill="#333", width=w - 14, justify="center", tags=tag)
        cv.create_text(x + w - 3, y + h + 7, text=bid, font=FONT_S, fill="#aaa", anchor="e", tags=tag)
        if not lib["feedthrough"]:                  # разрывает петли — пометка
            cv.create_text(x + 4, y + h + 7, text="⟳", font=FONT_S, fill="#aaa", anchor="w", tags=tag)
        for i in range(self._n_in(b)):
            px, py = self._in_pos(b, i)
            cv.create_polygon(px - 7, py - 5, px, py, px - 7, py + 5, fill="#555",
                              outline="", tags=(f"in:{bid}:{i}",))
            lab = self._in_label(b, i)
            if lab:
                cv.create_text(px + 4, py, text=lab, font=FONT_S, fill="#555", anchor="w",
                               tags=(f"in:{bid}:{i}",))
        if self._has_out(b):
            px, py = self._out_pos(b)
            cv.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#555", outline="",
                           tags=(f"out:{bid}",))
            self.val_items[bid] = cv.create_text(px + 7, py - 4, text="", font=FONT_V,
                                                 fill="#0a6", anchor="sw")
        elif b["type"] == "Output":
            self.val_items[bid] = cv.create_text(x + w + 6, y + h / 2, text="", font=FONT_V,
                                                 fill="#0a6", anchor="w")

    # =====================================================================
    # Мышь и клавиши
    # =====================================================================
    def _xy(self, e):
        return self.cv.canvasx(e.x), self.cv.canvasy(e.y)

    def _hit(self, x, y, r=5):
        """Что под мышью: ("out", id) | ("in", id, порт) | ("blk", id) |
        ("wire", i, участок) | None. Порты важнее блока, блок важнее провода."""
        found, segs = {}, []
        for item in reversed(self.cv.find_overlapping(x - r, y - r, x + r, y + r)):
            for tag in self.cv.gettags(item):
                kind, _, rest = tag.partition(":")
                if kind in ("out", "in", "blk"):
                    found.setdefault(kind, rest)
                elif kind == "seg":
                    segs.append(rest)
        if segs:                                     # соседние провода — ближайший к мыши
            def dist(rest):
                i, k = map(int, rest.split(":"))
                x0, y0, x1, y1 = self.routes[i][1][2 * k:2 * k + 4]
                px = min(max(x, min(x0, x1)), max(x0, x1))
                py = min(max(y, min(y0, y1)), max(y0, y1))
                return abs(px - x) + abs(py - y)
            found["seg"] = min(segs, key=dist)
        for kind in ("out", "in", "blk", "seg"):
            if kind in found:
                rest = found[kind]
                if kind == "in":
                    bid, port = rest.rsplit(":", 1)
                    return ("in", bid, int(port))
                if kind == "seg":
                    i, k = rest.split(":")
                    return ("wire", int(i), int(k))
                return (kind, rest)
        return None

    def _start_place(self, btype):
        self.placing = btype
        self.cv.configure(cursor="crosshair")
        self.lbl_status.configure(text=f"щёлкните на схеме, куда поставить «{self.lib[btype]['title']}» "
                                       f"(Esc — отмена)", fg="#333")

    def _cancel_place(self):
        self.placing, self.drag = None, None
        self.cv.configure(cursor="")
        self.cv.delete("ghost")
        self._status()

    def _hover(self, e):
        if not self.placing:                         # над участком провода — курсор «двигать»
            hit = self._hit(*self._xy(e))
            key = self._seg_key(hit[1], hit[2]) if hit and hit[0] == "wire" else None
            cur = {"x": "sb_h_double_arrow", "y": "sb_v_double_arrow"}.get(key[1] if key else "", "")
            if self.cv.cget("cursor") != cur:
                self.cv.configure(cursor=cur)
            return
        x, y = self._xy(e)
        self.cv.delete("ghost")
        w = BW_TYPE.get(self.placing, BW)
        self.cv.create_rectangle(x, y, x + w, y + 48, outline="#888", dash=(4, 3), tags=("ghost",))

    def _press(self, e):
        x, y = self._xy(e)
        if self.placing:
            btype = self.placing
            bid = self._new_id(btype)
            p = {key: default for key, _, default, _ in self.lib[btype]["params"]}
            self.blocks[bid] = {"type": btype, "x": round(x / GRID) * GRID,
                                "y": round(y / GRID) * GRID, "p": p}
            self.sel = ("blk", bid)
            self._cancel_place()
            self._changed()
            return
        hit = self._hit(x, y)
        if hit is None:
            self.sel = None
            self._redraw()
        elif hit[0] == "out":
            self.drag = ("from_out", hit[1])
        elif hit[0] == "in":
            bid, port = hit[1], hit[2]
            old = [w for w in self.wires if w["dst"] == bid and w["port"] == port]
            if old:                                  # провод за вход — перенести
                self.wires.remove(old[0])
                self._changed()
                self.drag = ("from_out", old[0]["src"])
            else:
                self.drag = ("from_in", bid, port)
        elif hit[0] == "blk":
            b = self.blocks[hit[1]]
            self.sel = ("blk", hit[1])
            self.drag = ("move", hit[1], x - b["x"], y - b["y"], False)
            self._redraw()
        elif hit[0] == "wire":
            i, k = hit[1], hit[2]
            self.sel = ("wire", i)
            key = self._seg_key(i, k)
            if key:
                # ствол общий у проводов одного выхода — двигается целиком
                c = 2 * k if key[1] == "x" else 2 * k + 1
                pos = self.routes[i][1][c]
                same = [j for j, w in enumerate(self.wires)
                        if w["src"] == self.wires[i]["src"] and self.routes[j][0] == self.routes[i][0]
                        and abs(self.routes[j][1][c] - pos) < 1]
                self.drag = ("seg", same, key)
            self._redraw()

    def _motion(self, e):
        if not self.drag:
            return
        x, y = self._xy(e)
        if self.drag[0] == "seg":
            _, idx, (field, axis) = self.drag
            v = round((x if axis == "x" else y) / 2) * 2
            for j in idx:
                w, (kind, pts) = self.wires[j], self.routes[j]
                if kind == "five":                   # обход — фиксируем и остальные изломы
                    w.setdefault("xa", pts[2])
                    w.setdefault("yb", pts[5])
                    w.setdefault("xb", pts[6])
                w[field] = v
            self._redraw()
            return
        if self.drag[0] == "move":
            _, bid, dx, dy, _ = self.drag
            b = self.blocks[bid]
            nx, ny = round((x - dx) / GRID) * GRID, round((y - dy) / GRID) * GRID
            if (nx, ny) != (b["x"], b["y"]):
                b["x"], b["y"] = max(0, nx), max(0, ny)
                self.drag = ("move", bid, dx, dy, True)
                self._redraw()
        else:
            if self.drag[0] == "from_out":
                x0, y0 = self._out_pos(self.blocks[self.drag[1]])
            else:
                x0, y0 = self._in_pos(self.blocks[self.drag[1]], self.drag[2])
            self.cv.delete("rubber")
            self.cv.create_line(x0, y0, x, y, fill=C_SEL, dash=(4, 2), width=1.5, tags=("rubber",))

    def _release(self, e):
        if not self.drag:
            return
        x, y = self._xy(e)
        d, self.drag = self.drag, None
        self.cv.delete("rubber")
        if d[0] in ("move", "seg"):
            return
        hit = self._hit(x, y, r=9)
        if d[0] == "from_out" and hit and hit[0] == "in":
            self._connect(d[1], hit[1], hit[2])
        elif d[0] == "from_in" and hit and hit[0] in ("out", "blk") and self._has_out(self.blocks[hit[1]]):
            self._connect(hit[1], d[1], d[2])

    def _dbl(self, e):
        hit = self._hit(*self._xy(e))
        if hit and hit[0] in ("blk", "in", "out"):
            self._params(hit[1])
        elif hit and hit[0] == "wire":
            self._auto_route(hit[1])

    def _auto_route(self, i):
        """Провод — снова автоматически (сброс ручных изломов)."""
        for key in ("xm", "xa", "yb", "xb"):
            self.wires[i].pop(key, None)
        self._redraw()

    def _menu(self, e):
        hit = self._hit(*self._xy(e))
        if not hit or hit[0] not in ("blk", "in", "out"):
            if hit and hit[0] == "wire":
                self.sel = hit[:2]
                self._redraw()
                m = tk.Menu(self.root, tearoff=0)
                m.add_command(label="Проложить автоматически", command=lambda: self._auto_route(hit[1]))
                m.add_command(label="Все провода — автоматически", command=self._auto_all)
                m.add_separator()
                m.add_command(label="Удалить провод", command=self._delete_sel)
                m.tk_popup(e.x_root, e.y_root)
            return
        bid = hit[1]
        self.sel = ("blk", bid)
        self._redraw()
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label="Параметры…", command=lambda: self._params(bid))
        m.add_command(label="Копия", command=lambda: self._duplicate(bid))
        m.add_separator()
        m.add_command(label="Удалить", command=self._delete_sel)
        m.tk_popup(e.x_root, e.y_root)

    def _auto_all(self):
        for w in self.wires:
            for key in ("xm", "xa", "yb", "xb"):
                w.pop(key, None)
        self._redraw()

    def _duplicate(self, bid):
        b = copy.deepcopy(self.blocks[bid])
        b["x"] += 30
        b["y"] += 30
        nid = self._new_id(b["type"])
        self.blocks[nid] = b
        self.sel = ("blk", nid)
        self._changed()

    def _delete_sel(self):
        if self.sel is None:
            return
        if self.sel[0] == "blk":
            bid = self.sel[1]
            del self.blocks[bid]
            self.wires = [w for w in self.wires if bid not in (w["src"], w["dst"])]
        else:
            del self.wires[self.sel[1]]
        self.sel = None
        self._changed()

    # =====================================================================
    # Параметры блока
    # =====================================================================
    def _params(self, bid):
        b = self.blocks[bid]
        lib = self.lib[b["type"]]
        if not lib["params"]:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title(f"{lib['title']} ({bid})")
        dlg.transient(self.root)
        dlg.configure(padx=10, pady=8)
        names = {"in": self.cat["inputs"], "out": self.cat["outputs"]}
        widgets = {}
        for row, (key, label, _, kind) in enumerate(lib["params"]):
            tk.Label(dlg, text=label, font=FONT, anchor="w").grid(row=row, column=0, sticky="w", pady=2)
            if kind in names:
                opts = names[kind]
                var = tk.StringVar(value=opts.get(b["p"][key], b["p"][key]))
                ttk.Combobox(dlg, textvariable=var, values=list(opts.values()), state="readonly",
                             width=28, font=FONT).grid(row=row, column=1, sticky="w")
                widgets[key] = (var, kind)
            else:
                var = tk.StringVar(value=str(b["p"][key]))
                ent = tk.Entry(dlg, textvariable=var, width=16, font=FONT)
                ent.grid(row=row, column=1, sticky="w")
                if row == 0:
                    ent.focus_set()
                    ent.select_range(0, "end")
                widgets[key] = (var, kind)
        hint = {"Sum": "знаки входов: например «+−» или «++−» (число знаков = число входов)",
                "P": "ошибку e = уставка − измерение собирает сумматор перед регулятором",
                "PI": "анти-виндап: при упоре в ±y_max интеграл не копится",
                "PID": "Tf — постоянная фильтра производной (шум!)",
                "Output": "приводы, к которым схема не подключена, ведёт штатная САУ",
                "VaScale": "y = u·(V0/Va)², в пределах [k_min, k_max] — gain scheduling"}.get(b["type"])
        n = len(lib["params"])
        if hint:
            tk.Label(dlg, text=hint, font=FONT_S, fg="#777", wraplength=330, justify="left"
                     ).grid(row=n, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def ok(_=None):
            new = {}
            for key, (var, kind) in widgets.items():
                v = var.get().strip()
                if kind in names:
                    v = next(k for k, t in names[kind].items() if t == v)
                elif kind == "num":
                    try:
                        v = float(v.replace(",", "."))
                    except ValueError:
                        messagebox.showerror("Параметр", f"«{key}»: нужно число", parent=dlg)
                        return
                elif kind == "int":
                    try:
                        v = int(float(v.replace(",", ".")))
                    except ValueError:
                        messagebox.showerror("Параметр", f"«{key}»: нужно целое число", parent=dlg)
                        return
                new[key] = v
            if b["type"] == "Sum" and not [c for c in new["signs"] if c in "+-−"]:
                messagebox.showerror("Параметр", "знаки: хотя бы один «+» или «−»", parent=dlg)
                return
            b["p"].update(new)
            self._prune(bid)
            dlg.destroy()
            self._changed()

        btns = tk.Frame(dlg)
        btns.grid(row=n + 1, column=0, columnspan=2, pady=(8, 0))
        tk.Button(btns, text="OK", width=10, command=ok).pack(side="left", padx=4)
        tk.Button(btns, text="Отмена", width=10, command=dlg.destroy).pack(side="left")
        dlg.bind("<Return>", ok)
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        dlg.grab_set()

    # =====================================================================
    # Команды
    # =====================================================================
    def _send(self, kind, val):
        try:
            self.q_cmd.put_nowait((kind, val))
        except queue.Full:
            pass

    def _apply(self):
        self.sent = self._scheme()
        self._send("scheme", self.sent)

    def _toggle_on(self):
        self._send("on", not self.tel.get("on", False))

    def _template(self, name):
        if self.blocks and not messagebox.askyesno(
                "Шаблон", f"Заменить текущую схему шаблоном «{name}»?", parent=self.root):
            return
        self._load(copy.deepcopy(self.templates[name]))

    def _open(self):
        fn = filedialog.askopenfilename(parent=self.root, title="Открыть схему",
                                        initialdir=self._dir(), filetypes=[("Схема САУ", "*.json")])
        if not fn:
            return
        try:
            with open(fn, encoding="utf-8") as f:
                self._load(json.load(f))
            self.file = fn
        except (OSError, ValueError, KeyError) as ex:
            messagebox.showerror("Открыть", f"Не удалось прочитать схему:\n{ex}", parent=self.root)

    def _save(self):
        fn = filedialog.asksaveasfilename(parent=self.root, title="Сохранить схему",
                                          initialdir=self._dir(), defaultextension=".json",
                                          initialfile=os.path.basename(self.file or "схема.json"),
                                          filetypes=[("Схема САУ", "*.json")])
        if not fn:
            return
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(self._scheme(), f, ensure_ascii=False, indent=1)
        self.file = fn

    def _dir(self):
        if self.file:
            return os.path.dirname(self.file)
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
        return os.path.normpath(d) if os.path.isdir(d) else os.getcwd()

    # =====================================================================
    # Телеметрия
    # =====================================================================
    def _status(self):
        if self.placing:
            return
        tel = self.tel
        if tel.get("err"):
            text, fg = "✖ " + tel["err"], "#b00000"
        elif self.dirty:
            text, fg = "схема изменена — «Применить», чтобы передать в САУ", "#c06000"
        else:
            text, fg = "✔ схема принята", "#2a7a2a"
        self.lbl_status.configure(text=text, fg=fg)

    def _tick(self):
        last = None
        while True:
            try:
                msg = self.q_tel.get_nowait()
            except queue.Empty:
                break
            if self.data and msg["t"] < self.data[-1][0]:     # сброс полёта
                self.data.clear()
            if msg["scopes"]:
                self.data.append((msg["t"], msg["scopes"]))
            last = msg
        if last is not None:
            if last["version"] != self.version:               # 3D приняла новую схему
                self.version = last["version"]
                self.applied = self.sent
            self.tel = last
            while self.data and self.data[-1][0] - self.data[0][0] > SCOPE_WIN:
                self.data.popleft()
            self._update_live()
            self._draw_scopes()
        self.root.after(TICK_MS, self._tick)

    def _update_live(self):
        tel = self.tel
        on = tel.get("on", False)
        self.btn_on.configure(text="Схема: ВКЛ" if on else "Схема: ВЫКЛ",
                              bg="#8fd18f" if on else "#e0e0e0")
        state = " ПАУЗА" if tel.get("paused") else (" АВАРИЯ" if tel.get("crashed") else "")
        self.lbl_flight.configure(text=(
            f"t {tel['t']:6.1f} с  режим: {tel['mode']}{state}   θ_ref {tel['theta_ref']:5.1f}°  "
            f"h_ref {tel['h_ref']:5.0f} м  Va_ref {tel['Va_ref']:4.1f}   │  "
            f"θ {tel['theta']:5.1f}°  h {tel['h']:5.0f} м  Va {tel['Va']:4.1f}  α {tel['alpha']:4.1f}°"))
        vals = tel.get("values", {})
        for bid, item in self.val_items.items():
            self.cv.itemconfigure(item, text=_fmt(vals[bid]) if bid in vals else "")
        self._status()

    # =====================================================================
    # Осциллограф
    # =====================================================================
    def _scope_list(self):
        """Осциллографы принятой схемы — сверху вниз, слева направо (до 4)."""
        sch = self.applied or {"blocks": [], "wires": []}
        sc = [b for b in sch["blocks"] if b["type"] == "Scope"]
        sc.sort(key=lambda b: (b["y"], b["x"]))
        return sc[:4], sch

    def _trace_label(self, sch, scope_id, port):
        blocks = {b["id"]: b for b in sch["blocks"]}
        for w in sch["wires"]:
            if w["dst"] == scope_id and w["port"] == port:
                b = blocks[w["src"]]
                if b["type"] == "Input":
                    return self.cat["inputs"].get(b["p"]["signal"], "")
                return f"{self.lib[b['type']]['title']} ({w['src']})"
        return "—"

    def _draw_scopes(self):
        sc = self.sc
        sc.delete("all")
        scopes, sch = self._scope_list()
        if not scopes:
            sc.create_text(20, 30, anchor="w", font=FONT, fill="#888",
                           text="Добавьте в схему блок «Осциллограф» и нажмите «Применить» — "
                                "здесь появятся графики его входов.")
            return
        cols = 2 if len(scopes) > 1 else 1
        rows = (len(scopes) + 1) // 2 if cols == 2 else 1
        cw, ch = (W_SCOPE + 190) / cols, H_SCOPE / rows
        t_end = self.data[-1][0] if self.data else 0.0
        t0 = max(0.0, t_end - SCOPE_WIN)
        for k, b in enumerate(scopes):
            x0, y0 = (k % cols) * cw, (k // cols) * ch
            self._draw_strip(b, sch, x0 + 50, y0 + 20, cw - 70, ch - 42, t0)

    def _draw_strip(self, b, sch, x0, y0, w, h, t0):
        sc, bid = self.sc, b["id"]
        n = min(max(int(b["p"].get("n", 1)), 1), 3)
        series = [[] for _ in range(n)]
        pts = [(t, v[bid]) for t, v in self.data if bid in v]
        step = max(1, len(pts) // 500)
        for t, vals in pts[::step]:
            for i in range(min(n, len(vals))):
                if vals[i] is not None:
                    series[i].append((t, vals[i]))
        ys = [v for s in series for _, v in s]
        lo, hi = (min(ys), max(ys)) if ys else (-1.0, 1.0)
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        title = b["p"].get("title") or bid
        sc.create_text(x0, y0 - 11, anchor="w", font=FONT_B, text=title)
        sc.create_rectangle(x0, y0, x0 + w, y0 + h, outline="#bbb")
        if lo < 0 < hi:
            yz = y0 + h * (hi - 0) / (hi - lo)
            sc.create_line(x0, yz, x0 + w, yz, fill="#ddd")
        for v, anchor_y in ((hi, y0), (lo, y0 + h)):
            sc.create_text(x0 - 4, anchor_y, anchor="e", font=FONT_V, fill="#666", text=_fmt(v))
        sc.create_text(x0 + w, y0 + h + 9, anchor="e", font=FONT_V, fill="#666",
                       text=f"{SCOPE_WIN:.0f} с")
        lx = x0 + 6
        for i, s in enumerate(series):
            col = C_TRACE[i]
            if len(s) > 1:
                xy = []
                for t, v in s:
                    xy += [x0 + w * (t - t0) / SCOPE_WIN, y0 + h * (hi - v) / (hi - lo)]
                sc.create_line(*xy, fill=col, width=1.5)
            lab = self._trace_label(sch, bid, i)
            last = f" = {_fmt(s[-1][1])}" if s else ""
            item = sc.create_text(lx, y0 + h + 9, anchor="w", font=FONT_S, fill=col, text=lab + last)
            lx = sc.bbox(item)[2] + 12


def run(q_cmd, q_tel, cat, templates, start, geometry="-0+0"):
    """Точка входа процесса окна редактора."""
    root = tk.Tk()
    root.configure(bg="white")
    root.geometry(geometry)
    Editor(root, q_cmd, q_tel, cat, templates, start)
    root.mainloop()
