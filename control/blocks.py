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
    1. блоки без прямого прохода (z⁻¹, интегратор I, апериодическое звено) выдают
       выход из своего состояния — входы им на этом такте не нужны;
    2. остальные блоки — по топологическому порядку проводов;
    3. блоки без прямого прохода обновляют состояние по входам такта.
Петля только из блоков с прямым проходом — алгебраическая: схема не принимается
(разорвать её — звеном z⁻¹, I или апериодическим).

Модуль — только numpy; редактор (viz/block_editor.py) получает описание
библиотеки через catalog() — словарь из простых типов.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Сигналы ЛА: входы и выходы схемы
# ---------------------------------------------------------------------------

# Входы: ключ → подпись (с единицей). Значения даёт сценарий (BlockLab3D._signals).
INPUT_SIGNALS = {
    "theta_ref": "θ_ref, °",      "theta": "θ, °",          "q": "q, °/с",
    "h_ref": "h_ref, м",          "h": "h, м",              "hdot": "ḣ, м/с",
    "Va_ref": "Va_ref, м/с",      "Va": "Va, м/с",
    "alpha": "α, °",              "beta": "β, °",
    "phi_ref": "φ_ref, °",        "phi": "φ, °",            "p": "p, °/с",
    "r": "r, °/с",                "psi": "ψ, °",
    "chi_ref": "χ_ref, °",        "chi": "χ, °",
    "de_trim": "δe_трим, °",      "dt_trim": "δt_трим",
}

# Выходы: ключ → (подпись, индекс в векторе управления [δe, δt, δa, δr], в радианы?)
OUTPUT_SIGNALS = {
    "de": ("δe — руль высоты, °", 0, True),
    "dt": ("δt — газ, 0…1", 1, False),
    "da": ("δa — элероны, °", 2, True),
    "dr": ("δr — руль направления, °", 3, True),
}


def _num(x):
    return float(x)


def _lim(x):
    """Предел выхода: 0 или пусто — без ограничения."""
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
    """y = смещение + A·sin(2π f t) — для проверки отклика на гармонику."""
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
    def step(self, u, ctx):
        return _num(self.p["K"]) * u[0]


class Product(Block):
    n_in = 2

    def step(self, u, ctx):
        return u[0] * u[1]


class Saturation(Block):
    def step(self, u, ctx):
        return min(max(u[0], _num(self.p["min"])), _num(self.p["max"]))


class DeadZone(Block):
    """Зона нечувствительности ±w: внутри — 0, снаружи — сдвиг на w."""

    def step(self, u, ctx):
        w = abs(_num(self.p["w"]))
        return u[0] - w if u[0] > w else (u[0] + w if u[0] < -w else 0.0)


class RateLimit(Block):
    """Ограничение скорости изменения сигнала, ед./с."""

    def reset(self):
        self.y = None

    def step(self, u, ctx):
        if self.y is None:
            self.y = u[0]
        d = _num(self.p["rate"]) * ctx["dt"]
        self.y += min(max(u[0] - self.y, -d), d)
        return self.y


class AngleError(Block):
    """Ошибка угла ref − meas, °, приведённая к [−180, 180) — для курса."""
    n_in = 2

    def step(self, u, ctx):
        return (u[0] - u[1] + 180.0) % 360.0 - 180.0


class VaScale(Block):
    """
    Масштаб по скорости (gain scheduling, как в PitchController):
    y = u·clip((V0/Va)², k_min, k_max). Вход 0 — сигнал, вход 1 — Va, м/с.
    Эффективность руля ∝ Va² — коэффициент контура сохраняется на любой скорости.
    """
    n_in = 2

    def step(self, u, ctx):
        p = self.p
        k = (_num(p["V0"]) / max(u[1], 1.0)) ** 2
        return u[0] * min(max(k, _num(p["k_min"])), _num(p["k_max"]))


class PIDBlock(Block):
    """
    П / ПИ / ПД / ПИД по ошибке на входе (тот же алгоритм, что control.controllers.PID):
        y = Kp·e + Ki·∫e dt + Kd·ė_ф,   ė_ф — производная через фильтр 1-го порядка Tf.
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
        tf = _num(p.get("Tf", 0.0))

        if Kd != 0.0 and dt > 0:
            a = dt / (tf + dt) if tf > 0 else 1.0
            self.d_filt += ((e - self.prev) / dt - self.d_filt) * a
        self.prev = e

        i_new = self.integral + e * dt
        y = Kp * e + Ki * i_new + Kd * self.d_filt
        if abs(y) > y_max and Ki * e * y > 0:          # упор, и интеграл тянет дальше
            y = Kp * e + Ki * self.integral + Kd * self.d_filt
        else:
            self.integral = i_new
        return min(max(y, -y_max), y_max)


class Derivative(Block):
    """Дифференцирующее звено с фильтром: y ≈ Kd·s/(Tf·s + 1)·u."""

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
    """y = Ki·∫u dt, ограничение ±y_max (0 — без). Без прямого прохода — разрывает петли."""
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
    """Изодромное (фильтр высоких частот) T·s/(T·s + 1): пропускает изменения, срезает постоянную."""

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
    """Форсирующее / запаздывающее звено (T1·s + 1)/(T2·s + 1); метод Тастина."""

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
    """Задержка на n тактов САУ (z⁻ⁿ). Без прямого прохода."""
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
# Библиотека: тип → (класс, группа, заголовок, параметры, подписи входов)
#   параметр: (ключ, подпись, по умолчанию, вид); вид: "num" | "str" | "in" | "out" | "int"
# ---------------------------------------------------------------------------

_PID_COMMON = [("y_max", "огр. выхода ± (0 — нет)", 0.0, "num")]
LIBRARY = {
    "Input":   (Input, "Источники", "Вход", [("signal", "сигнал ЛА", "theta", "in")], []),
    "Const":   (Const, "Источники", "Константа", [("value", "значение", 0.0, "num")], []),
    "Sine":    (Sine, "Источники", "Синус",
                [("A", "амплитуда", 2.0, "num"), ("f", "частота, Гц", 0.5, "num"),
                 ("bias", "смещение", 0.0, "num")], []),

    "P":   (PIDBlock, "Регуляторы", "П", [("Kp", "Kp", 1.0, "num")] + _PID_COMMON, ["e"]),
    "PI":  (PIDBlock, "Регуляторы", "ПИ",
            [("Kp", "Kp", 1.0, "num"), ("Ki", "Ki", 0.1, "num")] + _PID_COMMON, ["e"]),
    "PD":  (PIDBlock, "Регуляторы", "ПД",
            [("Kp", "Kp", 1.0, "num"), ("Kd", "Kd", 0.1, "num"),
             ("Tf", "Tf фильтра D, с", 0.05, "num")] + _PID_COMMON, ["e"]),
    "PID": (PIDBlock, "Регуляторы", "ПИД",
            [("Kp", "Kp", 1.0, "num"), ("Ki", "Ki", 0.1, "num"), ("Kd", "Kd", 0.1, "num"),
             ("Tf", "Tf фильтра D, с", 0.05, "num")] + _PID_COMMON, ["e"]),
    "I":   (Integrator, "Регуляторы", "И  Ki/s",
            [("Ki", "Ki", 1.0, "num"), ("y_max", "огр. ± (0 — нет)", 0.0, "num")], [""]),
    "D":   (Derivative, "Регуляторы", "Д  Kd·s",
            [("Kd", "Kd", 1.0, "num"), ("Tf", "Tf фильтра, с", 0.05, "num")], [""]),

    "Sum":      (Sum, "Математика", "Сумматор", [("signs", "знаки входов", "+-", "str")], None),
    "Gain":     (Gain, "Математика", "Усиление K", [("K", "K", 1.0, "num")], [""]),
    "Product":  (Product, "Математика", "Произведение ×", [], ["", ""]),
    "AngleErr": (AngleError, "Математика", "Ошибка угла ±180°", [], ["ref", "изм"]),
    "VaScale":  (VaScale, "Математика", "Масштаб по Va",
                 [("V0", "V0 — скорость настройки, м/с", 16.0, "num"),
                  ("k_min", "k_min", 0.25, "num"), ("k_max", "k_max", 9.0, "num")], ["u", "Va"]),

    "Sat":      (Saturation, "Нелинейности", "Ограничение",
                 [("min", "min", -25.0, "num"), ("max", "max", 25.0, "num")], [""]),
    "DeadZone": (DeadZone, "Нелинейности", "Зона нечувств.", [("w", "±w", 0.5, "num")], [""]),
    "RateLim":  (RateLimit, "Нелинейности", "Огр. скорости",
                 [("rate", "макс. скорость, ед./с", 60.0, "num")], [""]),

    "Aperiodic": (Aperiodic, "Динамические звенья", "K/(Ts+1)",
                  [("K", "K", 1.0, "num"), ("T", "T, с", 0.1, "num")], [""]),
    "Washout":   (Washout, "Динамические звенья", "Ts/(Ts+1)", [("T", "T, с", 1.0, "num")], [""]),
    "LeadLag":   (LeadLag, "Динамические звенья", "(T1s+1)/(T2s+1)",
                  [("T1", "T1, с", 0.2, "num"), ("T2", "T2, с", 0.05, "num")], [""]),
    "Delay":     (Delay, "Динамические звенья", "Задержка z⁻ⁿ",
                  [("n", "n тактов (по 0.01 с)", 1, "int")], [""]),

    "Output": (Output, "Приёмники", "Выход", [("act", "привод", "de", "out")], [""]),
    "Scope":  (Scope, "Приёмники", "Осциллограф",
               [("title", "заголовок", "", "str"), ("n", "число входов 1–3", 2, "int")], None),
}


def catalog():
    """Описание библиотеки для редактора (только простые типы — уходит в другой процесс)."""
    return {
        "blocks": {name: {"group": grp, "title": title,
                          "params": [list(x) for x in params],
                          "inputs": ins, "feedthrough": cls.feedthrough}
                   for name, (cls, grp, title, params, ins) in LIBRARY.items()},
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
                raise SchemeError("алгебраическая петля (контур без z⁻¹, И или K/(Ts+1)): "
                                  + ", ".join(loop))
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
# Заготовки схем
# ---------------------------------------------------------------------------

def _blk(bid, btype, x, y, **p):
    return {"id": bid, "type": btype, "x": x, "y": y, "p": {**default_params(btype), **p}}


def _w(src, dst, port=0):
    return {"src": src, "dst": dst, "port": port}


def pitch_blank():
    """Заготовка лабораторной: входы и выход канала тангажа, без регулятора."""
    return {"blocks": [
        _blk("in_tr", "Input", 20, 30, signal="theta_ref"),
        _blk("in_t", "Input", 20, 110, signal="theta"),
        _blk("in_q", "Input", 20, 190, signal="q"),
        _blk("in_Va", "Input", 20, 270, signal="Va"),
        _blk("in_de0", "Input", 20, 350, signal="de_trim"),
        _blk("out_de", "Output", 900, 190, act="de"),
    ], "wires": []}


def _pid_block(bid, x, y, Kp, Ki, Kd, Tf):
    """Регулятор того типа (П / ПИ / ПД / ПИД), какие коэффициенты не нулевые."""
    btype = {(False, False): "P", (True, False): "PI",
             (False, True): "PD", (True, True): "PID"}[(Ki != 0.0, Kd != 0.0)]
    p = {"Kp": Kp, "Ki": Ki, "Kd": Kd, "Tf": Tf}
    return _blk(bid, btype, x, y, **{k: v for k, v in p.items() if k in default_params(btype)})


def pitch_standard(aircraft, pit=None):
    """
    Штатный канал тангажа (control.controllers.PitchController) из блоков:
        θ_ref − θ → [ПИД θ] → [огр. ±q_max] → q_ref − q → [ПИД q] → [×(−1)]
        → [масштаб по Va] → + δe_трим → [огр. δe] → δe
    Знак: δe > 0 — нос вниз, поэтому выход q-контура инвертирован.
    pit — PitchControlParams (по умолчанию — расчёт control/tuning.py).
    """
    from .controllers import PitchControlParams
    pit = pit or PitchControlParams()
    deg = math.degrees
    q_max, Va0 = deg(pit.q_max), pit.Va_ref
    de_min, de_max = deg(aircraft.delta_e_min), deg(aircraft.delta_e_max)
    return {"blocks": [
        _blk("in_tr", "Input", 20, 50, signal="theta_ref"),
        _blk("in_t", "Input", 20, 140, signal="theta"),
        _blk("in_q", "Input", 20, 270, signal="q"),
        _blk("in_Va", "Input", 20, 340, signal="Va"),
        _blk("in_de0", "Input", 20, 410, signal="de_trim"),
        _blk("s1", "Sum", 170, 50, signs="+-"),
        _pid_block("k_t", 250, 58, pit.theta_Kp, pit.theta_Ki, pit.theta_Kd, pit.theta_tau),
        _blk("sat_q", "Sat", 400, 58, min=-q_max, max=q_max),
        _blk("s2", "Sum", 170, 220, signs="+-"),
        _pid_block("k_q", 250, 228, pit.q_Kp, pit.q_Ki, pit.q_Kd, pit.q_tau),
        _blk("neg", "Gain", 390, 228, K=-1.0),
        _blk("gs", "VaScale", 500, 220, V0=Va0, k_min=pit.gs_scale_min, k_max=pit.gs_scale_max),
        _blk("s3", "Sum", 650, 230, signs="++"),
        _blk("sat_de", "Sat", 720, 238, min=de_min, max=de_max),
        _blk("out_de", "Output", 860, 238, act="de"),
        _blk("sc_t", "Scope", 250, 130, title="тангаж θ, °"),
        _blk("sc_q", "Scope", 560, 130, title="угл. скорость q, °/с"),
        _blk("sc_de", "Scope", 860, 330, title="руль высоты δe, °", n=1),
    ], "wires": [
        _w("in_tr", "s1", 0), _w("in_t", "s1", 1), _w("s1", "k_t"), _w("k_t", "sat_q"),
        _w("sat_q", "s2", 0), _w("in_q", "s2", 1), _w("s2", "k_q"), _w("k_q", "neg"),
        _w("neg", "gs", 0), _w("in_Va", "gs", 1), _w("gs", "s3", 0), _w("in_de0", "s3", 1),
        _w("s3", "sat_de"), _w("sat_de", "out_de"),
        _w("in_tr", "sc_t", 0), _w("in_t", "sc_t", 1),
        _w("sat_q", "sc_q", 0), _w("in_q", "sc_q", 1),
        _w("sat_de", "sc_de", 0),
    ]}
