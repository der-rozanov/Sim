# -*- coding: utf-8 -*-
"""
Лаунчер симулятора: выбор режима, параметры, карта — и запуск.

Окно tkinter в стиле наземной станции (viz/station.py) собирает командную строку существующего скрипта (те же ключи argparse,
что в scenarios/*.py, viz/viewer*.py) и запускает его отдельным процессом. Лаунчер
не импортирует ursina и физику — только списки вариантов из sim/.

Запуск:  python launcher.py
         python launcher.py --selftest   — проверка сборки команд и списка карт
"""

import math
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from sim.config import AIRCRAFT_TYPES
from sim.turbulence import TURB_LEVELS
from viz.station import BG, BAR, CARD, EDGE, DIM, TXT, GREEN, AMBER, RED, CYAN   # палитра станции

# Имена — ключи viz/world3d.MAPS (world3d тянет ursina, поэтому список здесь;
# совпадение проверяет --selftest).
MAPS = {
    "default":          "10×10 км: ВПП, река, озеро, деревня, холм, леса",
    "realmap":           "[место] 4×4 км: спутниковый снимок на рельефе SRTM",
    "realmap_osm":       "[место] 4×4 км: контуры OpenStreetMap в стиле default",
    "realmap_large":     "[место] 10×10 км: спутниковый снимок",
    "realmap_large_osm": "[место] 10×10 км: раскраска по снимку + OSM",
}
MAP_NOTE = "Карты [место] при первом запуске скачивают снимок и рельеф в viz/map_cache/."

# Параметры: имя → (подпись, вид, варианты или умолчание, ключ командной строки).
# Вид: combo — список, num — число, check — флаг, file — путь к файлу.
PARAMS = {
    "aircraft": ("Тип ЛА",               "combo", list(AIRCRAFT_TYPES), "--aircraft"),
    "map":      ("Карта",                "combo", list(MAPS),           "--map"),
    "h0":       ("Начальная высота, м",  "num",   "100",                "--h0"),
    "cam":      ("Камера",               "combo", ["1", "2", "3", "4"], "--cam"),
    "wind_n":   ("Ветер на север, м/с",  "num",   "0",                  "--wind-n"),
    "wind_e":   ("Ветер на восток, м/с", "num",   "0",                  "--wind-e"),
    "turb":     ("Турбулентность",       "combo", list(TURB_LEVELS),    "--turb"),
    "turb_alt": ("Параметры турбул.",    "combo", ["low", "medium"],    "--turb-alt"),
    "sensors":  ("Датчики САУ",          "combo", ["real", "truth"],    "--sensors"),
    "est":      ("Оценка состояния",     "combo", ["ekf", "none"],      "--est"),
    "ap":       ("САУ включена при старте", "check", False,             "--ap"),
    "no_map":   ("Без окна станции",     "check", False,                "--no-map"),
    "scheme":   ("Схема (.json)",        "file",  "",                   "--scheme"),
    "scenario": ("Сценарий",             "combo", [],                   None),
    "log":      ("Файл .flightlog",      "file",  "",                   None),
    "speed":    ("Скорость проигрывания", "num",  "1",                  "--speed"),
    "static":   ("Графики без анимации", "check", False,                "--static"),
}
HINTS = {
    "turb":     "Драйден, табл. 4.1 B&M; в полёте — клавиша T",
    "turb_alt": "σ и L для высоты 50 м (low) или 600 м (medium)",
    "sensors":  "real — частоты, запаздывание, дрейф; truth — истина",
    "est":      "при real: EKF или «сырые» измерения",
    "log":      "пусто — последний полёт (2D — выбор в окне)",
}

SIM3D = ("aircraft", "map", "h0", "cam", "wind_n", "wind_e", "turb", "turb_alt",
         "sensors", "est")
MODES = {
    "Полёт 3D": dict(
        script="scenarios/GameScenario3D.py", params=SIM3D + ("ap", "no_map"), h0="100",
        help="Пилотирование в реальном времени: ручной режим или САУ, окно наземной станции "
             "с маршрутом. Полёт пишется в results/*.flightlog.\n\n"
             "P — САУ, H — высота, N — маршрут, Esc — выход."),
    "Стенд САУ": dict(
        script="scenarios/Bench3D.py", params=SIM3D + ("no_map",), h0="200",
        help="3D + окно блок-схем контуров (крен, TECS, тангаж, рыскание, защита по α, "
             "навигация) с ползунками коэффициентов на лету и осциллографом."),
    "Конструктор САУ": dict(
        script="scenarios/BlockLab3D.py", params=SIM3D + ("scheme",), h0="200",
        help="3D + редактор структурных схем: схема ведёт любые из δe, δt, δa, δr, "
             "остальные — штатная САУ. Меню «Шаблоны» — все каналы. "
             "Схемы лабораторной 6 — labs/lab6/schemes/."),
    "Сценарий s1–s10": dict(
        script=None, params=("scenario",),
        help="Расчётный прогон без реального времени: отчёт в консоли, "
             ".flightlog в results/, затем окно графиков."),
    "Графики полёта (2D)": dict(
        script="viz/viewer.py", params=("log", "speed", "static"), speed="2",
        help="Панели по логу: продольный канал; при манёвре по курсу — вид сверху, χ, φ, β, "
             "δa, δr. Анимация с заданной скоростью или статичные графики."),
    "3D-повтор полёта": dict(
        script="viz/viewer3d.py", params=("log", "map", "cam", "speed"), speed="1",
        help="Повтор .flightlog в 3D-сцене с моделью ЛА и рулями."),
}


def scenario_list():
    """[(файл, первая строка docstring)] для scenarios/s*.py, по номеру."""
    out = []
    folder = os.path.join(ROOT, "scenarios")
    for f in os.listdir(folder):
        if f.startswith("s") and f[1:2].isdigit() and f.endswith(".py"):
            with open(os.path.join(folder, f), encoding="utf-8") as fh:
                text = fh.read(2000)
            doc = text.split('"""')[1].strip().splitlines()[0] if '"""' in text else ""
            out.append((f, doc.rstrip(".")))
    return sorted(out, key=lambda x: int(x[0][1:].split("_")[0]))


def build_cmd(mode, values):
    """Командная строка режима по значениям параметров {имя: str | bool}."""
    m = MODES[mode]
    if m["script"] is None:
        cmd = [sys.executable, "scenarios/" + values["scenario"].split(" — ")[0]]
    else:
        cmd = [sys.executable, m["script"]]
    for name in m["params"]:
        _, kind, _, flag = PARAMS[name]
        v = values.get(name)
        if name == "log" and v:
            cmd.insert(2, v)                        # позиционный аргумент
        elif flag is None or v in ("", None, False):
            continue
        elif kind == "check":
            cmd.append(flag)
        else:
            if kind == "num":
                float(v)                            # ValueError — не число
            cmd += [flag, str(v)]
    return cmd


FIELD = "#2b3036"                               # фон полей ввода (как полосы станции)
UIF = lambda size, w="normal": ("Segoe UI", size, w)
MONO = lambda size, w="normal": ("Consolas", size, w)


def _btn(parent, text, cmd, fg=CYAN, bg=CARD, bold=False):
    """Плоская кнопка в стиле вкладок станции."""
    return tk.Button(parent, text=text, command=cmd, font=UIF(10, "bold" if bold else "normal"),
                     fg=fg, bg=bg, activebackground="#1d3a4a" if bg in (CARD, FIELD) else bg,
                     activeforeground=fg, relief="flat", bd=0, padx=14, pady=6,
                     highlightthickness=1, highlightbackground=EDGE, cursor="hand2")


def _card(parent, title):
    """Карточка станции: фон CARD, рамка EDGE, заголовок серым."""
    f = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=EDGE)
    tk.Label(f, text=title, bg=CARD, fg=DIM, font=UIF(9, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
    return f


class Launcher:
    def __init__(self, root):
        self.root = root
        self.procs = []
        root.title("Симулятор БПЛА — запуск")
        root.configure(bg=BG)
        root.minsize(1000, 640)
        root.geometry("1280x880")                   # размер не прыгает при смене режима
        self._style()
        self.scenarios = scenario_list()
        PARAMS["scenario"] = (PARAMS["scenario"][0], "combo",
                              [f"{f} — {d}" for f, d in self.scenarios], None)

        # Строка статуса (как у станции): состояние, режим, сводка параметров
        self.top = tk.Canvas(root, bg=BAR, height=44, highlightthickness=0)
        self.top.pack(fill="x")

        bottom = tk.Frame(root, bg=BAR)                 # команда и кнопки — всегда видны
        bottom.pack(side="bottom", fill="x")
        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=10)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Режим — кнопки как кнопки режима полёта станции (активная — зелёная)
        left = _card(body, "РЕЖИМ")
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        self.mode = tk.StringVar(value=next(iter(MODES)))
        self.mode_btns = {}
        for name in MODES:
            b = tk.Label(left, text=name, font=UIF(11, "bold"), anchor="w", padx=14, pady=7,
                         width=20, cursor="hand2", highlightthickness=1)
            b.pack(fill="x", padx=10, pady=3)
            b.bind("<Button-1>", lambda e, n=name: self.set_mode(n))
            self.mode_btns[name] = b
        tk.Label(left, text="О РЕЖИМЕ", bg=CARD, fg=DIM, font=UIF(9, "bold")).pack(
            anchor="w", padx=12, pady=(18, 4))
        self.about = tk.Label(left, bg=CARD, fg=TXT, font=UIF(10), justify="left", anchor="nw",
                              wraplength=280)
        self.about.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        # Параметры
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        card = _card(right, "ПАРАМЕТРЫ")
        card.grid(row=0, column=0, sticky="nsew")
        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="both", padx=12, pady=(0, 10))
        self.vars, self.rows = {}, {}
        for i, (name, (label, kind, choices, _)) in enumerate(PARAMS.items()):
            lab = tk.Label(grid, text=label if kind != "check" else "", bg=CARD, fg=TXT, font=UIF(10))
            if kind == "check":
                var = tk.BooleanVar(value=choices)
                w = tk.Checkbutton(grid, text=label, variable=var, command=self.refresh, bg=CARD,
                                   fg=TXT, selectcolor=FIELD, activebackground=CARD,
                                   activeforeground=TXT, font=UIF(10), bd=0, highlightthickness=0)
            elif kind == "combo":
                var = tk.StringVar(value=choices[0] if choices else "")
                w = ttk.Combobox(grid, textvariable=var, values=choices, state="readonly",
                                 font=MONO(10), width=62 if name == "scenario" else 18)
                w.bind("<<ComboboxSelected>>", lambda e: self.refresh())
            else:
                var = tk.StringVar(value=choices)
                w = tk.Frame(grid, bg=CARD)
                tk.Entry(w, textvariable=var, width=20 if kind == "num" else 30, font=MONO(10),
                         bg=FIELD, fg=TXT, insertbackground=TXT, relief="flat",
                         highlightthickness=1, highlightbackground=EDGE,
                         highlightcolor=CYAN).pack(side="left", ipady=3)
                if kind == "file":
                    _btn(w, "…", lambda n=name: self.pick_file(n), bg=FIELD).pack(side="left", padx=4)
                var.trace_add("write", lambda *a: self.refresh())
            hint = tk.Label(grid, text=HINTS.get(name, ""), bg=CARD, fg=DIM, font=UIF(9),
                            wraplength=300, justify="left")
            self.vars[name] = var
            self.rows[name] = (lab, w, hint)
            lab.grid(row=i, column=0, sticky="w", pady=3, padx=(0, 14))
            w.grid(row=i, column=1, sticky="w", pady=3)
            hint.grid(row=i, column=2, sticky="w", padx=10)
        self.vars["sensors"].trace_add("write", lambda *a: self.refresh())

        # Команда и кнопки
        self.cmd_text = tk.StringVar()
        tk.Entry(bottom, textvariable=self.cmd_text, state="readonly", font=MONO(10),
                 readonlybackground=BAR, fg=GREEN, relief="flat",
                 highlightthickness=0).pack(fill="x", padx=12, pady=(8, 2))
        row = tk.Frame(bottom, bg=BAR)
        row.pack(fill="x", padx=10, pady=(2, 10))
        self.status = tk.Label(row, bg=BAR, fg=DIM, font=UIF(10))
        self.status.pack(side="left")
        _btn(row, "▶  ЗАПУСК", self.launch, fg="black", bg=GREEN, bold=True).pack(side="right", padx=(8, 0))
        for text, fn in reversed([("СПРАВКА", self.open_help), ("ПАПКА RESULTS", self.open_results),
                                  ("КОПИРОВАТЬ КОМАНДУ", self.copy_cmd),
                                  ("ПО УМОЛЧАНИЮ", self.reset)]):
            _btn(row, text, fn).pack(side="right", padx=3)
        root.bind("<Return>", lambda e: self.launch())
        root.bind("<F1>", lambda e: self.open_help())

        self.on_mode()
        self._poll()

    def _style(self):
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure("TCombobox", fieldbackground=FIELD, background=CARD, foreground=TXT,
                     arrowcolor=TXT, bordercolor=EDGE, lightcolor=FIELD, darkcolor=FIELD,
                     selectbackground=FIELD, selectforeground=TXT, padding=3)
        st.map("TCombobox", fieldbackground=[("readonly", FIELD), ("disabled", CARD)],
               foreground=[("disabled", DIM)], arrowcolor=[("disabled", EDGE)],
               bordercolor=[("focus", CYAN)])
        for k, v in (("background", FIELD), ("foreground", TXT), ("selectBackground", "#1d3a4a"),
                     ("selectForeground", CYAN), ("font", MONO(10))):
            self.root.option_add(f"*TCombobox*Listbox.{k}", v)

    # --- значения ---------------------------------------------------------------
    def values(self):
        return {n: v.get() for n, v in self.vars.items()}

    def set_mode(self, name):
        self.mode.set(name)
        self.on_mode()

    def on_mode(self):
        m = MODES[self.mode.get()]
        for key in ("h0", "speed"):
            if key in m:
                self.vars[key].set(m[key])
        for name, b in self.mode_btns.items():
            on = name == self.mode.get()
            b.configure(bg=GREEN if on else CARD, fg="black" if on else TXT,
                        highlightbackground=GREEN if on else EDGE)
        self.refresh()

    def refresh(self):
        m = MODES[self.mode.get()]
        shown = set(m["params"])
        for name, widgets in self.rows.items():
            for w in widgets:
                w.grid() if name in shown else w.grid_remove()
        est = self.rows["est"][1]
        est.configure(state="readonly" if self.vars["sensors"].get() == "real" else "disabled")

        mp = self.vars["map"].get()
        self.rows["map"][2].configure(text=MAPS[mp] + (". " + MAP_NOTE if mp.startswith("realmap") else ""),
                                      fg=CYAN)
        self.about.configure(text=m["help"])
        try:
            cmd = build_cmd(self.mode.get(), self.values())
            self.cmd_text.set("> python " + " ".join(f'"{a}"' if " " in a else a for a in cmd[1:]))
            self.say("")
        except ValueError:
            self.cmd_text.set("")
            self.say("Числовое поле заполнено неверно", RED)
        self._top()

    def _top(self):
        """Строка статуса: плашка состояния, режим, сводка параметров (как у станции)."""
        cv, v = self.top, self.values()
        cv.delete("all")
        n = sum(p.poll() is None for p in self.procs)
        cv.create_rectangle(0, 0, 210, 44, fill="#2f6f3e" if n else "#2a2f36", width=0)
        cv.create_text(16, 22, text=f"▲  ЗАПУЩЕНО {n}" if n else "■  ГОТОВ", anchor="w",
                       fill="white", font=UIF(13, "bold"))
        x = 226
        t = cv.create_text(x, 22, text=self.mode.get().upper(), anchor="w", fill=CYAN,
                           font=UIF(11, "bold"))
        x = cv.bbox(t)[2] + 26
        shown = MODES[self.mode.get()]["params"]
        chips = []
        if "aircraft" in shown:
            chips.append((f"ЛА {v['aircraft'].upper()}", TXT))
        if "map" in shown:
            chips.append((f"КАРТА {v['map']}", TXT))
        if "h0" in shown:
            chips.append((f"h0 {v['h0']} m", TXT))
        if "wind_n" in shown:
            try:
                wn, we = float(v["wind_n"]), float(v["wind_e"])
                w = math.hypot(wn, we)
                frm = f" FROM {math.degrees(math.atan2(-we, -wn)) % 360:03.0f}°" if w > 0.2 else ""
                chips.append((f"WIND {w:.1f} m/s{frm}", CYAN))
            except ValueError:
                chips.append(("WIND ?", RED))
        if "turb" in shown and v["turb"] != "off":
            chips.append((f"TURB {v['turb']}", AMBER))
        if "sensors" in shown:
            chips.append(("SENS " + (f"real/{v['est']}" if v["sensors"] == "real" else "truth"), TXT))
        for s, c in chips:
            t = cv.create_text(x, 22, text=s, anchor="w", fill=c, font=UIF(11))
            x = cv.bbox(t)[2] + 24
        cv.create_text(int(cv.winfo_width()) - 12, 22, text="UAV SIM", anchor="e", fill=DIM,
                       font=UIF(10, "bold"))

    def _poll(self):
        """Раз в секунду — число запущенных процессов в плашке состояния."""
        self.procs = [p for p in self.procs if p.poll() is None]
        self._top()
        self.root.after(1000, self._poll)

    def say(self, text, color=DIM):
        self.status.configure(text=text, fg=color)

    def reset(self):
        for name, (_, kind, choices, _) in PARAMS.items():
            self.vars[name].set(choices[0] if kind == "combo" and choices else
                                choices if kind != "combo" else "")
        self.on_mode()

    # --- кнопки -----------------------------------------------------------------
    def pick_file(self, name):
        if name == "log":
            path = filedialog.askopenfilename(
                initialdir=os.path.join(ROOT, "results"),
                filetypes=[("Полётный лог", "*.flightlog *.npz"), ("Все файлы", "*.*")])
        else:
            path = filedialog.askopenfilename(
                initialdir=os.path.join(ROOT, "labs", "lab6", "schemes"),
                filetypes=[("Схема САУ", "*.json"), ("Все файлы", "*.*")])
        if path:
            self.vars[name].set(path)

    def launch(self):
        try:
            cmd = build_cmd(self.mode.get(), self.values())
        except ValueError:
            return
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env))
        self.say(f"Запущено: {self.mode.get()} — окно откроется через несколько секунд, "
                 "вывод в консоли лаунчера", GREEN)
        self._top()

    def open_help(self):
        os.startfile(os.path.join(ROOT, "docs", "USER_GUIDE.md"))

    def open_results(self):
        path = os.path.join(ROOT, "results")
        os.makedirs(path, exist_ok=True)
        os.startfile(path)

    def copy_cmd(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.cmd_text.get()[2:])
        self.say("Команда скопирована")


def selftest():
    from viz.world3d import MAPS as WORLD_MAPS
    assert list(MAPS) == list(WORLD_MAPS), "список карт разошёлся с viz/world3d.MAPS"
    sc = scenario_list()
    assert [f[:3] for f, _ in sc][:2] == ["s1_", "s2_"] and sc[-1][0].startswith("s10"), sc
    v = {"aircraft": "fpv", "map": "realmap", "h0": "150", "cam": "2", "wind_n": "0",
         "wind_e": "5", "turb": "off", "turb_alt": "low", "sensors": "real", "est": "ekf",
         "ap": True, "no_map": False, "log": "", "speed": "1", "scenario": sc[0][0] + " — x"}
    cmd = build_cmd("Полёт 3D", v)[1:]
    assert cmd[0] == "scenarios/GameScenario3D.py" and "--ap" in cmd and "--no-map" not in cmd
    assert cmd[cmd.index("--map") + 1] == "realmap" and cmd[cmd.index("--h0") + 1] == "150"
    assert build_cmd("Сценарий s1–s10", v)[1:] == ["scenarios/" + sc[0][0]]
    assert build_cmd("3D-повтор полёта", dict(v, log="a.flightlog"))[1:3] == \
        ["viz/viewer3d.py", "a.flightlog"]
    try:
        build_cmd("Стенд САУ", dict(v, h0="abc"))
        raise AssertionError("нечисловое h0 не отловлено")
    except ValueError:
        pass
    for mode in MODES:                              # каждый скрипт существует
        s = MODES[mode]["script"]
        assert s is None or os.path.exists(os.path.join(ROOT, s)), s
    print("LAUNCHER SELFTEST PASSED")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--selftest" in sys.argv:
        selftest()
    else:
        try:                                        # чёткий шрифт при масштабе Windows > 100 %
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
        root = tk.Tk()
        Launcher(root)
        root.mainloop()
