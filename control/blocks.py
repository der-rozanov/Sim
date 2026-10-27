# -*- coding: utf-8 -*-
"""
Структурные схемы САУ из блоков (лабораторная «Конструктор САУ», scenarios/BlockLab3D.py).

Схема — набор блоков и проводов (как в Simulink), описывается словарём (JSON):

    {"blocks": [{"id": "b1", "type": "PID", "x": 100, "y": 80,
                 "p": {"Kp": 16.2, "Ki": 0.0, ...}}, ...],
     "wires":  [{"src": "b1", "dst": "b2", "port": 0}, ...]}

    src — блок-источник (у каждого блока один выход), dst/port — вход приёмника.
    x, y — место на холсте редактора (здесь не используются).

Diagram(scheme) проверяет схему и задаёт порядок счёта; Diagram.step(sig, t, dt)
— один такт САУ: входы схемы (сигналы ЛА) → выходы на приводы рулей.

Единицы на схеме — «инженерные»: углы в градусах, угловые скорости в °/с,
h — м, Va — м/с, газ δt — 0…1. Коэффициенты «угол → угол» от этого не зависят
(Kq = 0.0381 с одинаков в рад/(рад/с) и °/(°/с)).

Порядок счёта на такт:
    1. блоки без прямого прохода (запаздывание z⁻ⁿ, интегратор, апериодическое
       звено) выдают выход из своего состояния — входы им на этом такте не нужны;
    2. остальные блоки — по топологическому порядку проводов;
    3. блоки без прямого прохода обновляют состояние по входам такта.
Петля только из блоков с прямым проходом — алгебраическая: схема не принимается
(разорвать её — звеном запаздывания, интегратором или апериодическим звеном).

Шаблоны (заготовки и штатные каналы) — все каналы САУ: тангаж, высота, скорость,
крен, курс + скольжение; штатные совпадают с регуляторами control/controllers.py
(проверка — checks/check_blocks.py).

Модуль — только numpy; редактор (viz/block_editor.py) получает описание
библиотеки через catalog() — словарь из простых типов.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Сигналы ЛА: входы и выходы схемы
# ---------------------------------------------------------------------------

# Входы: ключ → подпись (с единицей). Значения даёт sau_signals(): уставки —
# от штатной САУ, измерения — её датчики. Ключи — часть формата схемы (JSON).
INPUT_SIGNALS = {
    "theta_ref": "θ_ref — тангаж заданный, °",     "theta": "θ — тангаж, °",
    "q": "q — угл. скорость тангажа, °/с",
    "h_ref": "h_ref — высота заданная, м",         "h": "h — высота, м",
    "hdot": "ḣ — вертикальная скорость, м/с",
    "Va_ref": "Va_ref — возд. скорость заданная, м/с",
    "Va": "Va — воздушная скорость, м/с",
    "alpha": "α — угол атаки, °",                  "beta": "β — угол скольжения, °",
    "phi_ref": "φ_ref — крен заданный, °",         "phi": "φ — крен, °",
    "p": "p — угл. скорость крена, °/с",           "r": "r — угл. скорость рыскания, °/с",
    "psi": "ψ — угол рыскания, °",
    "chi_ref": "χ_ref — путевой угол заданный, °", "chi": "χ — путевой угол, °",
    "Vg": "Vg — путевая скорость, м/с",
    "chidot_ff": "χ̇_уп — упреждение скорости разворота, °/с",
    "alpha_trim": "α_бал — балансировочный угол атаки, °",
    "de_trim": "δe_бал — балансировочный δe, °",
    "dt_trim": "δt_бал — балансировочный газ",
}

# Выходы: ключ → (подпись, индекс в векторе управления [δe, δt, δa, δr], в радианы?)
OUTPUT_SIGNALS = {
    "de": ("δe — руль высоты, °", 0, True),
    "dt": ("δt — газ (тяга), 0…1", 1, False),
    "da": ("δa — элероны, °", 2, True),
    "dr": ("δr — руль направления, °", 3, True),
}


def sau_signals(sau, state, alpha, beta):
    """
    Входы схемы в единицах схемы (углы — градусы) после шага штатной САУ
    (control.sau.SAU.step): измерения — оценка САУ (sau.est), уставки — то, что
    штатная САУ подаёт в свои контуры (θ_ref — после TECS, φ_ref — после контура курса).
    alpha, beta — истинные (как у TruthSensors).
    """
    from sim.state import R, euler
    from .controllers import wrap_angle
    from .sau import Mode
    e, deg = sau.est, math.degrees
    return {
        "theta_ref": deg(sau.theta_ref), "theta": deg(e["theta"]), "q": deg(e["q"]),
        "h_ref": sau.refs.h if sau.mode is not Mode.PITCH else e["h"],
        "h": e["h"], "hdot": e["hdot"],
        "Va_ref": sau.refs.Va, "Va": e["Va"],
        "alpha": deg(alpha), "beta": deg(beta),
        "phi_ref": deg(sau.lat.phi_ref), "phi": deg(wrap_angle(e["phi"])), "p": deg(e["p"]),
        "r": deg(state[R]), "psi": deg(euler(state)[2]) % 360.0,
        "chi_ref": deg(sau.refs.chi), "chi": deg(e["chi"]),
        "Vg": e["Vg"], "chidot_ff": deg(e["Vg"] * sau.refs.kappa),
        "alpha_trim": deg(sau.alpha_trim),
        "de_trim": deg(sau.de_trim), "dt_trim": sau.lon.thr_trim,
    }


def _num(x):
    return float(x)


def _lim(x):
    """Предел: 0 или пусто — без ограничения."""
    x = float(x)
    return math.inf if x <= 0.0 else x


# ---------------------------------------------------------------------------
# Блоки
# ---------------------------------------------------------------------------

class Block:
    """
    Блок схемы: один выход, n_in входов.
    feedthrough=True  — y = step(u, ctx): выход зависит от входа того же такта;
    feedthrough=False — y = output(ctx) из состояния, затем update(u, ctx).
    ctx: {"t": время, с; "dt": шаг, с; "sig": сигналы ЛА}.
    """
    feedthrough = True
    n_in = 1

    def __init__(self, p):
        self.p = p
        self.reset()

    def reset(self):
        pass


class Input(Block):
    n_in = 0

    def step(self, u, ctx):
        return float(ctx["sig"][self.p["signal"]])


class Const(Block):
    n_in = 0

    def step(self, u, ctx):
        return _num(self.p["value"])


class Sine(Block):
    """Синусоида y = b + A·sin(2π f t) — для частотных испытаний контура."""
    n_in = 0

    def step(self, u, ctx):
        p = self.p
        return _num(p["bias"]) + _num(p["A"]) * math.sin(2 * math.pi * _num(p["f"]) * ctx["t"])


class Sum(Block):
    """Сумматор: знаки входов — строка из «+» и «−», например «+−»."""

    def __init__(self, p):
        self.signs = [-1.0 if c in "-−" else 1.0 for c in p["signs"] if c in "+-−"]
        self.n_in = len(self.signs)
        super().__init__(p)

    def step(self, u, ctx):
        return sum(s * x for s, x in zip(self.signs, u))


class Gain(Block):
    """Усилительное (безынерционное) звено y = K·u."""

    def step(self, u, ctx):
        return _num(self.p["K"]) * u[0]


class Product(Block):
    n_in = 2

    def step(self, u, ctx):
        return u[0] * u[1]


class Saturation(Block):
    """Насыщение: y = u в пределах [min, max], за пределами — предел."""

    def step(self, u, ctx):
        return min(max(u[0], _num(self.p["min"])), _num(self.p["max"]))


class DeadZone(Block):
    """Зона нечувствительности ±w: внутри — 0, снаружи — сдвиг на w."""

    def step(self, u, ctx):
        w = abs(_num(self.p["w"]))
        return u[0] - w if u[0] > w else (u[0] + w if u[0] < -w else 0.0)


class RateLimit(Block):
    """
    Ограничение производной: выход следует за входом, но |dy/dt| ≤ R (ед. сигнала / с).
    Дискретно: y_k = y_(k−1) + clip(u_k − y_(k−1), −R·Δt, R·Δt); старт — y = u.
    (В Simulink — Rate Limiter.)
    """

    def reset(self):
        self.y = None

    def step(self, u, ctx):
        if self.y is None:
            self.y = u[0]
        d = _num(self.p["rate"]) * ctx["dt"]
        self.y += min(max(u[0] - self.y, -d), d)
        return self.y


class AngleError(Block):
    """Разность углов a − b, °, приведённая к [−180, 180): ошибка по курсу кратчайшим путём."""
    n_in = 2

    def step(self, u, ctx):
        return (u[0] - u[1] + 180.0) % 360.0 - 180.0


class VaScale(Block):
    """
    Перестройка коэффициента по воздушной скорости (gain scheduling):
    y = u·clip((V0/Va)ⁿ, k_min, k_max). Вход 0 — сигнал, вход 1 — Va, м/с.
    n = 2: эффективность руля ∝ Va² — контур сохраняет свойства на любой скорости
    (PitchController, ПИ скорости крена); n = 1 — упреждение крена (RollController).
    """
    n_in = 2

    def step(self, u, ctx):
        p = self.p
        k = (_num(p["V0"]) / max(u[1], 1.0)) ** _num(p["n"])
        return u[0] * min(max(k, _num(p["k_min"])), _num(p["k_max"]))


class TurnBank(Block):
    """
    Крен координированного разворота: φ = arctg(V·ω/g).
    Вход 0 — заданная скорость разворота ω = χ̇, °/с; вход 1 — путевая скорость V, м/с;
    выход — φ, °. Из равновесия сил в развороте без скольжения: tg φ = V·ω/g.
    """
    n_in = 2

    def step(self, u, ctx):
        return math.degrees(math.atan(u[1] * math.radians(u[0]) / _num(self.p["g"])))


class PIDBlock(Block):
    """
    П / ПИ / ПД / ПИД по ошибке на входе (тот же алгоритм, что control.controllers.PID):
        y = Kp·e + Ki·∫e dt + Kd·ė_ф,   ė_ф — производная через фильтр 1-го порядка Tf.
    И-составляющая Ki·∫e ограничена ±I_max (0 — без ограничения) — как integral_limit
    у PID: I_max = Ki·integral_limit.
    Выход ограничен ±y_max (0 — без ограничения); при упоре интеграл не копится
    дальше (анти-виндап — условное интегрирование).
    Набор коэффициентов задаёт тип: у П нет Ki, Kd; у ПИ — Kd; у ПД — Ki.
    """
    KEYS = ("Kp", "Ki", "Kd")

    def reset(self):
        self.integral = 0.0
        self.d_filt = 0.0
        self.prev = 0.0

    def step(self, u, ctx):
        p, dt, e = self.p, ctx["dt"], u[0]
        Kp, Ki, Kd = (_num(p.get(k, 0.0)) for k in self.KEYS)
        y_max = _lim(p.get("y_max", 0.0))
        i_lim = _lim(p.get("i_max", 0.0)) / abs(Ki) if Ki != 0.0 else math.inf
        tf = _num(p.get("Tf", 0.0))

        if Kd != 0.0 and dt > 0:
            a = dt / (tf + dt) if tf > 0 else 1.0
            self.d_filt += ((e - self.prev) / dt - self.d_filt) * a
        self.prev = e

        i_new = min(max(self.integral + e * dt, -i_lim), i_lim)
        y = Kp * e + Ki * i_new + Kd * self.d_filt
        if abs(y) > y_max and Ki * e * y > 0:          # упор, и интеграл тянет дальше
            y = Kp * e + Ki * self.integral + Kd * self.d_filt
        else:
            self.integral = i_new
        return min(max(y, -y_max), y_max)


class Derivative(Block):
    """Реальное дифференцирующее звено Kd·s/(Tf·s + 1) (идеальное Kd·s с фильтром Tf)."""

    def reset(self):
        self.y, self.prev = 0.0, None

    def step(self, u, ctx):
        dt = ctx["dt"]
        if self.prev is None:
            self.prev = u[0]
        tf = _num(self.p["Tf"])
        a = dt / (tf + dt) if tf > 0 else 1.0
        self.y += ((u[0] - self.prev) / dt - self.y) * a
        self.prev = u[0]
        return _num(self.p["Kd"]) * self.y


class Integrator(Block):
    """Интегратор Ki/s: y = Ki·∫u dt, ограничение ±y_max (0 — без). Без прямого прохода."""
    feedthrough = False

    def reset(self):
        self.x = 0.0

    def output(self, ctx):
        return self.x

    def update(self, u, ctx):
        y_max = _lim(self.p["y_max"])
        self.x = min(max(self.x + _num(self.p["Ki"]) * u[0] * ctx["dt"], -y_max), y_max)


class Aperiodic(Block):
    """Апериодическое звено K/(T·s + 1); точная дискретизация. Без прямого прохода."""
    feedthrough = False

    def reset(self):
        self.x = 0.0

    def output(self, ctx):
        return self.x

    def update(self, u, ctx):
        T = _num(self.p["T"])
        a = math.exp(-ctx["dt"] / T) if T > 0 else 0.0
        self.x = a * self.x + (1.0 - a) * _num(self.p["K"]) * u[0]


class Washout(Block):
    """
    Фильтр высоких частот 1-го порядка T·s/(T·s + 1) (washout): пропускает изменения,
    срезает постоянную составляющую. Не путать с изодромным звеном k·(1 + 1/(T·s)).
    """

    def reset(self):
        self.lp = None

    def step(self, u, ctx):
        if self.lp is None:
            self.lp = u[0]
        T = _num(self.p["T"])
        a = ctx["dt"] / (T + ctx["dt"]) if T > 0 else 1.0
        self.lp += (u[0] - self.lp) * a
        return u[0] - self.lp


class LeadLag(Block):
    """
    Корректирующее звено (T1·s + 1)/(T2·s + 1); дискретизация методом Тастина.
    T1 > T2 — опережение по фазе, T1 < T2 — отставание по фазе.
    """

    def reset(self):
        self.u1, self.y1 = None, 0.0

    def step(self, u, ctx):
        dt, T1, T2 = ctx["dt"], _num(self.p["T1"]), _num(self.p["T2"])
        if self.u1 is None:                             # старт из установившегося режима
            self.u1, self.y1 = u[0], u[0]
        a, b = 2 * T1 / dt, 2 * T2 / dt
        y = ((a + 1) * u[0] + (1 - a) * self.u1 - (1 - b) * self.y1) / (b + 1)
        self.u1, self.y1 = u[0], y
        return y


class Delay(Block):
    """Звено запаздывания на n тактов САУ: z⁻ⁿ (τ = n·Δt). Без прямого прохода."""
    feedthrough = False

    def reset(self):
        self.buf = [0.0] * max(1, int(self.p["n"]))

    def output(self, ctx):
        return self.buf[0]

    def update(self, u, ctx):
        self.buf = self.buf[1:] + [u[0]]


class Output(Block):
    """Выход на привод руля / газ; значение = вход."""

    def step(self, u, ctx):
        return u[0]


class Scope(Block):
    """Осциллограф: 1–3 входа, рисуется в окне редактора."""

    def __init__(self, p):
        self.n_in = min(max(int(p["n"]), 1), 3)
        super().__init__(p)

    def step(self, u, ctx):
        return u[0] if u else 0.0


# ---------------------------------------------------------------------------
# Библиотека: тип → (класс, группа, название, параметры, подписи входов, формула)
#   параметр: (ключ, подпись, по умолчанию, вид); вид: "num" | "str" | "in" | "out" | "int"
#   название — технический термин (ТАУ), формула — передаточная функция / закон блока
#   Ключи типов и параметров — часть формата схемы (JSON), не менять.
# ---------------------------------------------------------------------------

_Y_MAX = ("y_max", "y_max — предел выхода ± (0 — без предела)", 0.0, "num")
_I_MAX = ("i_max", "I_max — предел И-составляющей ± (0 — без предела)", 0.0, "num")
_TF = ("Tf", "Tf — постоянная фильтра Д-составляющей, с", 0.05, "num")
LIBRARY = {
    "Input":   (Input, "Источники сигналов", "Вход", [("signal", "сигнал ЛА", "theta", "in")], [], ""),
    "Const":   (Const, "Источники сигналов", "Константа", [("value", "значение", 0.0, "num")], [], ""),
    "Sine":    (Sine, "Источники сигналов", "Синусоида",
                [("A", "A — амплитуда", 2.0, "num"), ("f", "f — частота, Гц", 0.5, "num"),
                 ("bias", "b — постоянная составляющая", 0.0, "num")], [], "b + A·sin 2πft"),

    "P":   (PIDBlock, "Регуляторы", "П-регулятор", [("Kp", "Kp", 1.0, "num"), _Y_MAX], ["e"], ""),
    "PI":  (PIDBlock, "Регуляторы", "ПИ-регулятор",
            [("Kp", "Kp", 1.0, "num"), ("Ki", "Ki", 0.1, "num"), _I_MAX, _Y_MAX], ["e"], ""),
    "PD":  (PIDBlock, "Регуляторы", "ПД-регулятор",
            [("Kp", "Kp", 1.0, "num"), ("Kd", "Kd", 0.1, "num"), _TF, _Y_MAX], ["e"], ""),
    "PID": (PIDBlock, "Регуляторы", "ПИД-регулятор",
            [("Kp", "Kp", 1.0, "num"), ("Ki", "Ki", 0.1, "num"), ("Kd", "Kd", 0.1, "num"),
             _TF, _I_MAX, _Y_MAX], ["e"], ""),
    "I":   (Integrator, "Регуляторы", "Интегратор",
            [("Ki", "Ki", 1.0, "num"), _Y_MAX], [""], "Ki/s"),
    "D":   (Derivative, "Регуляторы", "Дифференциатор",
            [("Kd", "Kd", 1.0, "num"), ("Tf", "Tf — постоянная фильтра, с", 0.05, "num")],
            [""], "Kd·s/(Tf·s+1)"),

    "Sum":      (Sum, "Математические операции", "Сумматор",
                 [("signs", "знаки входов", "+-", "str")], None, ""),
    "Gain":     (Gain, "Математические операции", "Усилительное звено",
                 [("K", "K", 1.0, "num")], [""], ""),
    "Product":  (Product, "Математические операции", "Умножение", [], ["", ""], ""),
    "AngleErr": (AngleError, "Математические операции", "Разность углов", [], ["a", "b"],
                 "a − b → [−180°, 180°)"),
    "VaScale":  (VaScale, "Математические операции", "Перестройка по Va",
                 [("V0", "V0 — скорость настройки, м/с", 16.0, "num"),
                  ("n", "n — показатель степени", 2.0, "num"),
                  ("k_min", "k_min — нижний предел множителя", 0.25, "num"),
                  ("k_max", "k_max — верхний предел множителя", 9.0, "num")], ["u", "Va"],
                 "u·(V0/Va)ⁿ"),
    "TurnBank": (TurnBank, "Математические операции", "Крен разворота",
                 [("g", "g — ускорение свободного падения, м/с²", 9.81, "num")], ["ω", "V"],
                 "φ = arctg(V·ω/g)"),

    "Sat":      (Saturation, "Нелинейные звенья", "Насыщение",
                 [("min", "нижний предел", -25.0, "num"), ("max", "верхний предел", 25.0, "num")],
                 [""], ""),
    "DeadZone": (DeadZone, "Нелинейные звенья", "Зона нечувствительности",
                 [("w", "w — полуширина зоны", 0.5, "num")], [""], ""),
    "RateLim":  (RateLimit, "Нелинейные звенья", "Ограничение производной",
                 [("rate", "R — предел |dy/dt|, ед./с", 60.0, "num")], [""], "|dy/dt| ≤ R"),

    "Aperiodic": (Aperiodic, "Линейные звенья", "Апериодическое звено",
                  [("K", "K", 1.0, "num"), ("T", "T, с", 0.1, "num")], [""], "K/(T·s+1)"),
    "Washout":   (Washout, "Линейные звенья", "Фильтр высоких частот",
                  [("T", "T, с", 1.0, "num")], [""], "T·s/(T·s+1)"),
    "LeadLag":   (LeadLag, "Линейные звенья", "Корректирующее звено",
                  [("T1", "T1, с", 0.2, "num"), ("T2", "T2, с", 0.05, "num")], [""],
                  "(T1·s+1)/(T2·s+1)"),
    "Delay":     (Delay, "Линейные звенья", "Звено запаздывания",
                  [("n", "n — число тактов (такт 0.01 с)", 1, "int")], [""], "z⁻ⁿ"),

    "Output": (Output, "Приёмники", "Выход", [("act", "привод", "de", "out")], [""], ""),
    "Scope":  (Scope, "Приёмники", "Осциллограф",
               [("title", "заголовок", "", "str"), ("n", "число входов 1–3", 2, "int")], None, ""),
}


def catalog():
    """Описание библиотеки для редактора (только простые типы — уходит в другой процесс)."""
    return {
        "blocks": {name: {"group": grp, "title": title, "formula": formula,
                          "params": [list(x) for x in params],
                          "inputs": ins, "feedthrough": cls.feedthrough}
                   for name, (cls, grp, title, params, ins, formula) in LIBRARY.items()},
        "inputs": dict(INPUT_SIGNALS),
        "outputs": {k: v[0] for k, v in OUTPUT_SIGNALS.items()},
    }


def default_params(btype):
    return {k: d for k, _, d, _ in LIBRARY[btype][3]}


# ---------------------------------------------------------------------------
# Схема
# ---------------------------------------------------------------------------

class SchemeError(ValueError):
    """Схема не принимается: текст — для студента."""


def _title(b):
    return f"«{LIBRARY[b['type']][2]}» ({b['id']})"


class Diagram:
    """
    Схема, готовая к счёту.

    Атрибуты после step():
        values  — {id блока: выход на этом такте} (подписи проводов в редакторе)
        scopes  — {id осциллографа: [входы]}
    """

    def __init__(self, scheme):
        self.blocks, self.desc = {}, {}
        for b in scheme.get("blocks", []):
            if b["type"] not in LIBRARY:
                raise SchemeError(f"неизвестный блок {b['type']!r}")
            p = default_params(b["type"])
            p.update(b.get("p", {}))
            try:
                for key, label, _, kind in LIBRARY[b["type"]][3]:
                    if kind == "num":
                        p[key] = float(p[key])
                    elif kind == "int":
                        p[key] = int(p[key])
                blk = LIBRARY[b["type"]][0](p)
            except (ValueError, TypeError):
                raise SchemeError(f"{_title(b)}: параметр не число")
            self.blocks[b["id"]] = blk
            self.desc[b["id"]] = b
            if b["type"] == "Input" and p["signal"] not in INPUT_SIGNALS:
                raise SchemeError(f"{_title(b)}: неизвестный сигнал {p['signal']!r}")
            if b["type"] == "Output" and p["act"] not in OUTPUT_SIGNALS:
                raise SchemeError(f"{_title(b)}: неизвестный привод {p['act']!r}")

        # Входы: (dst, port) → src; у входа не больше одного провода
        self.src = {bid: [None] * blk.n_in for bid, blk in self.blocks.items()}
        for w in scheme.get("wires", []):
            if w["src"] not in self.blocks or w["dst"] not in self.blocks:
                continue                                  # провод к удалённому блоку
            ports = self.src[w["dst"]]
            if not 0 <= w["port"] < len(ports):
                continue
            if ports[w["port"]] is not None:
                raise SchemeError(f"{_title(self.desc[w['dst']])}: к входу {w['port'] + 1} "
                                  f"подведено два провода")
            ports[w["port"]] = w["src"]

        # Выходы на приводы: не больше одного блока на привод, вход подключён
        self.outputs = {}
        for bid, blk in self.blocks.items():
            if isinstance(blk, Output):
                act = blk.p["act"]
                if act in self.outputs:
                    raise SchemeError(f"на привод {OUTPUT_SIGNALS[act][0]} два блока «Выход»")
                if self.src[bid][0] is None:
                    raise SchemeError(f"{_title(self.desc[bid])}: вход не подключён")
                self.outputs[act] = bid

        self.unconnected = [bid for bid, s in self.src.items()
                            if any(x is None for x in s) and not isinstance(self.blocks[bid], Scope)]
        self.order = self._sort()
        self.values = {bid: 0.0 for bid in self.blocks}
        self.scopes = {}

    def _sort(self):
        """Топологический порядок блоков с прямым проходом (провода от блоков
        без прямого прохода порядка не задают — их выход известен в начале такта)."""
        ft = [bid for bid, b in self.blocks.items() if b.feedthrough]
        deps = {bid: {s for s in self.src[bid]
                      if s is not None and self.blocks[s].feedthrough} for bid in ft}
        order, done = [], set()
        while len(order) < len(ft):
            ready = [bid for bid in ft if bid not in done and deps[bid] <= done]
            if not ready:
                rest = {b for b in ft if b not in done}
                while True:                       # отбросить блоки после петли (их никто не ждёт)
                    tail = {b for b in rest if not any(b in deps[x] for x in rest)}
                    if not tail:
                        break
                    rest -= tail
                loop = [_title(self.desc[b]) for b in ft if b in rest]
                raise SchemeError("алгебраическая петля (контур без звена запаздывания, "
                                  "интегратора или апериодического звена): " + ", ".join(loop))
            order += ready
            done.update(ready)
        return order

    def reset(self):
        for blk in self.blocks.values():
            blk.reset()

    def step(self, sig, t, dt):
        """
        Такт схемы. sig — {ключ INPUT_SIGNALS: значение в единицах схемы}.
        Возвращает {привод: значение в единицах схемы} — только подключённые приводы.
        """
        ctx = {"t": t, "dt": dt, "sig": sig}
        v = self.values
        delayed = [(bid, b) for bid, b in self.blocks.items() if not b.feedthrough]
        for bid, b in delayed:
            v[bid] = b.output(ctx)
        inp = lambda bid: [0.0 if s is None else v[s] for s in self.src[bid]]
        for bid in self.order:
            u = inp(bid)
            v[bid] = self.blocks[bid].step(u, ctx)
            if isinstance(self.blocks[bid], Scope):
                self.scopes[bid] = [None if s is None else v[s] for s in self.src[bid]]
        for bid, b in delayed:
            b.update(inp(bid), ctx)
        return {act: v[bid] for act, bid in self.outputs.items()}

    def commands(self, out):
        """Выходы схемы → {индекс в [δe, δt, δa, δr]: значение в СИ (рад, о.е.)}."""
        res = {}
        for act, val in out.items():
            _, idx, is_angle = OUTPUT_SIGNALS[act]
            res[idx] = math.radians(val) if is_angle else val
        return res


# ---------------------------------------------------------------------------
# Шаблоны схем: заготовки (входы и выходы канала, без регулятора) и штатные
# каналы (регуляторы control/controllers.py, собранные из блоков)
# ---------------------------------------------------------------------------

def _blk(bid, btype, x, y, **p):
    return {"id": bid, "type": btype, "x": x, "y": y, "p": {**default_params(btype), **p}}


def _w(src, dst, port=0):
    return {"src": src, "dst": dst, "port": port}


def _inputs(x, y0, step, *signals):
    """Столбец блоков «Вход»: id in_<сигнал>."""
    return [_blk(f"in_{s}", "Input", x, y0 + k * step, signal=s) for k, s in enumerate(signals)]


def _blank(signals, acts):
    return {"blocks": _inputs(20, 30, 80, *signals)
            + [_blk(f"out_{a}", "Output", 900, 30 + 120 * k, act=a) for k, a in enumerate(acts)],
            "wires": []}


def pitch_blank():
    """Заготовка: канал тангажа θ_ref → δe."""
    return _blank(["theta_ref", "theta", "q", "Va", "de_trim"], ["de"])


def altitude_blank():
    """Заготовка: высота h_ref → (θ_ref) → δe."""
    return _blank(["h_ref", "h", "hdot", "alpha_trim", "theta", "q", "Va", "de_trim"], ["de"])


def speed_blank():
    """Заготовка: воздушная скорость Va_ref → δt."""
    return _blank(["Va_ref", "Va", "dt_trim"], ["dt"])


def roll_blank():
    """Заготовка: крен φ_ref → δa."""
    return _blank(["phi_ref", "phi", "p", "Va"], ["da"])


def lateral_blank():
    """Заготовка: курс χ_ref → (φ_ref) → δa; скольжение β → δr."""
    return _blank(["chi_ref", "chi", "Vg", "chidot_ff", "phi", "p", "Va", "beta", "r"],
                  ["da", "dr"])


def _pid_block(bid, x, y, Kp, Ki, Kd, Tf, **extra):
    """Регулятор того типа (П / ПИ / ПД / ПИД), какие коэффициенты не нулевые."""
    btype = {(False, False): "P", (True, False): "PI",
             (False, True): "PD", (True, True): "PID"}[(Ki != 0.0, Kd != 0.0)]
    p = {"Kp": Kp, "Ki": Ki, "Kd": Kd, "Tf": Tf, **extra}
    return _blk(bid, btype, x, y, **{k: v for k, v in p.items() if k in default_params(btype)})


def _pitch_part(aircraft, pit, dx, theta_ref_src):
    """
    Каскад тангажа (PitchController) от сигнала theta_ref_src:
        θ_ref − θ → [ПИД θ] → [насыщение ±q_max] → q_ref − q → [ПИД q] → [×(−1)]
        → [перестройка по Va, n = 2] → + δe_бал → [насыщение δe] → δe
    """
    deg = math.degrees
    q_max = deg(pit.q_max)
    blocks = [
        _blk("in_theta", "Input", dx + 20, 140, signal="theta"),
        _blk("in_q", "Input", dx + 20, 270, signal="q"),
        _blk("in_Va", "Input", dx + 20, 360, signal="Va"),
        _blk("in_de_trim", "Input", dx + 20, 450, signal="de_trim"),
        _blk("s_th", "Sum", dx + 170, 50, signs="+-"),
        _pid_block("k_th", dx + 250, 58, pit.theta_Kp, pit.theta_Ki, pit.theta_Kd, pit.theta_tau),
        _blk("sat_q", "Sat", dx + 420, 58, min=-q_max, max=q_max),
        _blk("s_q", "Sum", dx + 170, 220, signs="+-"),
        _pid_block("k_q", dx + 250, 228, pit.q_Kp, pit.q_Ki, pit.q_Kd, pit.q_tau),
        _blk("neg_q", "Gain", dx + 420, 228, K=-1.0),
        _blk("gs_q", "VaScale", dx + 580, 220, V0=pit.Va_ref, n=2.0,
             k_min=pit.gs_scale_min, k_max=pit.gs_scale_max),
        _blk("s_de", "Sum", dx + 760, 230, signs="++"),
        _blk("sat_de", "Sat", dx + 840, 238,
             min=deg(aircraft.delta_e_min), max=deg(aircraft.delta_e_max)),
        _blk("out_de", "Output", dx + 1000, 238, act="de"),
        _blk("sc_q", "Scope", dx + 600, 110, title="угл. скорость тангажа q, °/с"),
        _blk("sc_de", "Scope", dx + 1000, 340, title="руль высоты δe, °", n=1),
    ]
    wires = [
        _w(theta_ref_src, "s_th", 0), _w("in_theta", "s_th", 1), _w("s_th", "k_th"),
        _w("k_th", "sat_q"), _w("sat_q", "s_q", 0), _w("in_q", "s_q", 1), _w("s_q", "k_q"),
        _w("k_q", "neg_q"), _w("neg_q", "gs_q", 0), _w("in_Va", "gs_q", 1),
        _w("gs_q", "s_de", 0), _w("in_de_trim", "s_de", 1), _w("s_de", "sat_de"),
        _w("sat_de", "out_de"),
        _w("sat_q", "sc_q", 0), _w("in_q", "sc_q", 1), _w("sat_de", "sc_de", 0),
    ]
    return blocks, wires


def pitch_standard(aircraft, pit=None):
    """
    Штатный канал тангажа (control.controllers.PitchController) из блоков.
    Знак: δe > 0 — нос вниз, поэтому выход q-контура инвертирован.
    pit — PitchControlParams (по умолчанию — расчёт control/tuning.py).
    """
    from .controllers import PitchControlParams
    pit = pit or PitchControlParams()
    blocks, wires = _pitch_part(aircraft, pit, 0, "in_theta_ref")
    blocks += [_blk("in_theta_ref", "Input", 20, 50, signal="theta_ref"),
               _blk("sc_th", "Scope", 250, 150, title="тангаж θ, °")]
    wires += [_w("in_theta_ref", "sc_th", 0), _w("in_theta", "sc_th", 1)]
    return {"blocks": blocks, "wires": wires}


def altitude_standard(aircraft, pit=None, alt=None):
    """
    Высота + тангаж (AltitudeHold → PitchController, раздельные контуры B&M гл. 6):
        h_ref − h → [ПИ h, предел И] → + α_бал → [насыщение ±θ_max] → θ_ref → каскад тангажа
    Отличие от AltitudeHold: анти-виндап ПИ — по его выходу (±θ_max), а не по θ_ref
    после прибавки α_бал (расхождение — в полосе шириной α_бал у упора).
    alt — AltitudeHoldParams (по умолчанию — расчёт control/tuning.py).
    """
    from .controllers import PitchControlParams, AltitudeHoldParams
    pit = pit or PitchControlParams()
    alt = alt or AltitudeHoldParams()
    deg = math.degrees
    th_max = deg(alt.theta_max)
    blocks, wires = _pitch_part(aircraft, pit, 620, "sat_th")
    blocks += [
        _blk("in_h_ref", "Input", 20, 50, signal="h_ref"),
        _blk("in_h", "Input", 20, 140, signal="h"),
        _blk("in_alpha_trim", "Input", 170, 170, signal="alpha_trim"),
        _blk("s_h", "Sum", 170, 50, signs="+-"),
        _blk("k_h", "PI", 250, 58, Kp=deg(alt.Kp), Ki=deg(alt.Ki),
             i_max=deg(alt.Ki * alt.integral_limit), y_max=th_max),
        _blk("s_a", "Sum", 400, 60, signs="++"),
        _blk("sat_th", "Sat", 470, 68, min=-th_max, max=th_max),
        _blk("sc_h", "Scope", 250, 270, title="высота h, м"),
    ]
    wires += [_w("in_h_ref", "s_h", 0), _w("in_h", "s_h", 1), _w("s_h", "k_h"),
              _w("k_h", "s_a", 0), _w("in_alpha_trim", "s_a", 1), _w("s_a", "sat_th"),
              _w("in_h_ref", "sc_h", 0), _w("in_h", "sc_h", 1)]
    return {"blocks": blocks, "wires": wires}


def speed_standard(aircraft, spd=None):
    """
    Скорость → газ (control.controllers.SpeedController):
        Va_ref − Va → [ПИ, предел И] → + δt_бал → [насыщение 0…1] → δt
    spd — SpeedControlParams (по умолчанию — расчёт control/tuning.py).
    """
    from .controllers import SpeedControlParams
    spd = spd or SpeedControlParams()
    blocks = [
        _blk("in_Va_ref", "Input", 20, 50, signal="Va_ref"),
        _blk("in_Va", "Input", 20, 140, signal="Va"),
        _blk("in_dt_trim", "Input", 250, 150, signal="dt_trim"),
        _blk("s_V", "Sum", 170, 50, signs="+-"),
        _pid_block("k_V", 250, 58, spd.Va_Kp, spd.Va_Ki, spd.Va_Kd, spd.Va_tau,
                   i_max=spd.Va_Ki * spd.Va_integral_limit),
        _blk("s_dt", "Sum", 420, 60, signs="++"),
        _blk("sat_dt", "Sat", 500, 68, min=aircraft.throttle_min, max=aircraft.throttle_max),
        _blk("out_dt", "Output", 660, 68, act="dt"),
        _blk("sc_V", "Scope", 420, 180, title="воздушная скорость Va, м/с"),
    ]
    wires = [_w("in_Va_ref", "s_V", 0), _w("in_Va", "s_V", 1), _w("s_V", "k_V"),
             _w("k_V", "s_dt", 0), _w("in_dt_trim", "s_dt", 1), _w("s_dt", "sat_dt"),
             _w("sat_dt", "out_dt"), _w("in_Va_ref", "sc_V", 0), _w("in_Va", "sc_V", 1)]
    return {"blocks": blocks, "wires": wires}


def _roll_part(aircraft, rp, dx, dy, phi_ref_src):
    """
    Крен (RollController, схема PX4): φ_ref − φ → [1/τ_φ] → [насыщение ±p_max] = p_ref;
        δa = [FF] → [перестройка по Va, n = 1] + (p_ref − p) → [ПИ] → [перестройка, n = 2]
        → [насыщение ±δa_max]
    Отличие от RollController: тот при упоре элеронов не копит интеграл (анти-виндап
    по δa); здесь ограничена только И-составляющая.
    """
    deg = math.degrees
    p_max = deg(rp.p_max)
    da = deg(aircraft.delta_a_max)
    s2_min, s2_max = rp.scale_min ** 2, rp.scale_max ** 2
    blocks = [
        _blk("in_phi", "Input", dx + 20, dy + 90, signal="phi"),
        _blk("in_p", "Input", dx + 400, dy + 190, signal="p"),
        _blk("in_Va", "Input", dx + 560, dy + 280, signal="Va"),
        _blk("s_phi", "Sum", dx + 170, dy, signs="+-"),
        _blk("k_tau", "Gain", dx + 240, dy + 8, K=1.0 / rp.tau_phi),
        _blk("sat_p", "Sat", dx + 400, dy + 8, min=-p_max, max=p_max),
        _blk("k_ff", "Gain", dx + 560, dy + 8, K=rp.FF),
        _blk("gs_ff", "VaScale", dx + 720, dy, V0=rp.Va_ref, n=1.0,
             k_min=rp.scale_min, k_max=rp.scale_max),
        _blk("s_p", "Sum", dx + 560, dy + 130, signs="+-"),
        _blk("k_p", "PI", dx + 640, dy + 138, Kp=rp.Kp, Ki=rp.Ki,
             i_max=deg(rp.Ki * rp.integral_limit)),
        _blk("gs_p", "VaScale", dx + 800, dy + 130, V0=rp.Va_ref, n=2.0,
             k_min=s2_min, k_max=s2_max),
        _blk("s_da", "Sum", dx + 960, dy + 60, signs="++"),
        _blk("sat_da", "Sat", dx + 1040, dy + 68, min=-da, max=da),
        _blk("out_da", "Output", dx + 1200, dy + 68, act="da"),
        _blk("sc_p", "Scope", dx + 960, dy + 220, title="угл. скорость крена p, °/с"),
    ]
    wires = [
        _w(phi_ref_src, "s_phi", 0), _w("in_phi", "s_phi", 1), _w("s_phi", "k_tau"),
        _w("k_tau", "sat_p"), _w("sat_p", "k_ff"), _w("k_ff", "gs_ff", 0), _w("in_Va", "gs_ff", 1),
        _w("sat_p", "s_p", 0), _w("in_p", "s_p", 1), _w("s_p", "k_p"), _w("k_p", "gs_p", 0),
        _w("in_Va", "gs_p", 1), _w("gs_ff", "s_da", 0), _w("gs_p", "s_da", 1),
        _w("s_da", "sat_da"), _w("sat_da", "out_da"),
        _w("sat_p", "sc_p", 0), _w("in_p", "sc_p", 1),
    ]
    return blocks, wires


def roll_standard(aircraft, rp=None):
    """Штатный канал крена (RollController) от уставки φ_ref."""
    from .controllers import RollControlParams
    rp = rp or RollControlParams()
    blocks, wires = _roll_part(aircraft, rp, 0, 50, "in_phi_ref")
    blocks += [_blk("in_phi_ref", "Input", 20, 50, signal="phi_ref"),
               _blk("sc_phi", "Scope", 240, 230, title="крен φ, °")]
    wires += [_w("in_phi_ref", "sc_phi", 0), _w("in_phi", "sc_phi", 1)]
    return {"blocks": blocks, "wires": wires}


def lateral_standard(aircraft, lat=None):
    """
    Боковой канал (LateralController):
      курс:  χ_ref − χ (кратчайшим путём) → [K_χ] + χ̇_уп = ω → [крен разворота arctg(Vg·ω/g)]
             → [насыщение ±φ_max] → [ограничение производной] = φ_ref → крен → δa
      скольжение: β → [апериодическое τ_β] → [ПИ, предел И] → [×(−1)] + [K_mix]·δa → [насыщение ±δr_max] → δr
    Ограничитель уставки крена стартует с первой команды (LateralController — с текущего φ).
    """
    from .controllers import LateralControlParams
    lat = lat or LateralControlParams()
    deg = math.degrees
    phi_max, dr = deg(lat.phi_ref_max), deg(aircraft.delta_r_max)
    y_r = 360                                             # ряд крена
    y_b = 720                                             # ряд скольжения
    blocks, wires = _roll_part(aircraft, lat.roll, 0, y_r, "rl_phi")
    blocks += [
        _blk("in_chi_ref", "Input", 20, 50, signal="chi_ref"),
        _blk("in_chi", "Input", 20, 140, signal="chi"),
        _blk("in_chidot_ff", "Input", 330, 140, signal="chidot_ff"),
        _blk("in_Vg", "Input", 500, 160, signal="Vg"),
        _blk("e_chi", "AngleErr", 170, 50),
        _blk("k_chi", "Gain", 330, 58, K=lat.chi_K),
        _blk("s_w", "Sum", 500, 60, signs="++"),
        _blk("tb", "TurnBank", 580, 60, g=aircraft.g),
        _blk("sat_phi", "Sat", 760, 68, min=-phi_max, max=phi_max),
        _blk("rl_phi", "RateLim", 920, 68, rate=deg(lat.phi_ref_rate)),
        _blk("sc_chi", "Scope", 760, 200, title="путевой угол χ, °"),
        _blk("sc_phi", "Scope", 240, y_r + 230, title="крен φ, °"),

        _blk("in_beta", "Input", 20, y_b, signal="beta"),
        _blk("f_b", "Aperiodic", 20, y_b + 80, K=1.0, T=lat.beta_tau),
        _blk("k_b", "PI", 170, y_b + 8, Kp=lat.beta_Kp, Ki=lat.beta_Ki,
             i_max=deg(lat.beta_Ki * lat.beta_integral_limit)),
        _blk("neg_b", "Gain", 330, y_b + 8, K=-1.0),
        _blk("k_mix", "Gain", 330, y_b + 100, K=lat.da_dr_mix),
        _blk("s_dr", "Sum", 500, y_b + 10, signs="++"),
        _blk("sat_dr", "Sat", 580, y_b + 18, min=-dr, max=dr),
        _blk("out_dr", "Output", 740, y_b + 18, act="dr"),
        _blk("sc_b", "Scope", 740, y_b + 110, title="скольжение β, °", n=1),
    ]
    wires += [
        _w("in_chi_ref", "e_chi", 0), _w("in_chi", "e_chi", 1), _w("e_chi", "k_chi"),
        _w("k_chi", "s_w", 0), _w("in_chidot_ff", "s_w", 1), _w("s_w", "tb", 0),
        _w("in_Vg", "tb", 1), _w("tb", "sat_phi"), _w("sat_phi", "rl_phi"),
        _w("in_chi_ref", "sc_chi", 0), _w("in_chi", "sc_chi", 1),
        _w("rl_phi", "sc_phi", 0), _w("in_phi", "sc_phi", 1),
        _w("in_beta", "f_b"), _w("f_b", "k_b"), _w("k_b", "neg_b"), _w("neg_b", "s_dr", 0),
        _w("sat_da", "k_mix"), _w("k_mix", "s_dr", 1), _w("s_dr", "sat_dr"),
        _w("sat_dr", "out_dr"), _w("in_beta", "sc_b", 0),
    ]
    if not lat.beta_hold:                    # РН в нейтрали: канал скольжения не нужен
        keep = {b["id"] for b in blocks if b["id"] not in
                ("in_beta", "f_b", "k_b", "neg_b", "k_mix", "s_dr", "sat_dr", "out_dr", "sc_b")}
        blocks = [b for b in blocks if b["id"] in keep]
        wires = [w for w in wires if w["src"] in keep and w["dst"] in keep]
    return {"blocks": blocks, "wires": wires}


def _merge(*parts):
    """Несколько схем в одну: (префикс id, сдвиг по y, схема)."""
    blocks, wires = [], []
    for prefix, dy, sch in parts:
        for b in sch["blocks"]:
            blocks.append({**b, "id": prefix + b["id"], "y": b["y"] + dy})
        wires += [{**w, "src": prefix + w["src"], "dst": prefix + w["dst"]} for w in sch["wires"]]
    return {"blocks": blocks, "wires": wires}


def full_standard(aircraft, pit=None, alt=None, spd=None, lat=None):
    """
    Вся САУ из блоков (раздельные контуры B&M гл. 6, SAUParams(lon="split")):
    высота + тангаж → δe, скорость → δt, курс + крен → δa, скольжение → δr.
    Осциллографы — h, Va, χ, φ (редактор показывает до 4).
    """
    lon = altitude_standard(aircraft, pit, alt)
    spd_s = speed_standard(aircraft, spd)
    lat_s = lateral_standard(aircraft, lat)
    sch = _merge(("lon_", 0, lon), ("spd_", 560, spd_s), ("lat_", 860, lat_s))
    keep = {"lon_sc_h", "spd_sc_V", "lat_sc_chi", "lat_sc_phi"}
    drop = {b["id"] for b in sch["blocks"] if b["type"] == "Scope" and b["id"] not in keep}
    sch["blocks"] = [b for b in sch["blocks"] if b["id"] not in drop]
    sch["wires"] = [w for w in sch["wires"] if w["dst"] not in drop]
    return sch


def templates(aircraft, Va0):
    """Шаблоны для редактора: «Заготовка: …» и «Штатная САУ: …» (точка настройки Va0)."""
    from .controllers import (PitchControlParams, RollControlParams, LateralControlParams)
    pit = PitchControlParams(Va_ref=Va0)
    lat = LateralControlParams(roll=RollControlParams(Va_ref=Va0))
    return {
        "Заготовка: тангаж → δe": pitch_blank(),
        "Заготовка: высота → δe": altitude_blank(),
        "Заготовка: скорость → δt": speed_blank(),
        "Заготовка: крен → δa": roll_blank(),
        "Заготовка: курс и скольжение → δa, δr": lateral_blank(),
        "Штатная САУ: тангаж": pitch_standard(aircraft, pit),
        "Штатная САУ: высота + тангаж": altitude_standard(aircraft, pit),
        "Штатная САУ: скорость": speed_standard(aircraft),
        "Штатная САУ: крен": roll_standard(aircraft, lat.roll),
        "Штатная САУ: курс + крен + скольжение": lateral_standard(aircraft, lat),
        "Штатная САУ: все каналы": full_standard(aircraft, pit, lat=lat),
    }
