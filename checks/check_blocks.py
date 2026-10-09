# -*- coding: utf-8 -*-
"""
Проверка структурных схем control/blocks.py (лабораторная «Конструктор САУ»):

  1. шаблон pitch_standard ≡ PitchController: на одной последовательности входов
     (θ_ref, θ, q, Va — случайные, с упором руля) δe совпадает до округления;
     то же с ПИД (Ki, Kd ≠ 0) в обоих контурах;
  2. блок ПИД ≡ control.controllers.PID (без ограничения выхода);
  3. звенья: апериодическое — переходная функция 1 − e^(−t/T);
     изодромное — постоянный вход срезает (старт из установившегося); z⁻ⁿ — сдвиг на n;
     LeadLag в установившемся режиме — коэффициент 1;
  4. ошибки схемы: алгебраическая петля, два провода на вход, два выхода на один
     привод; петля с z⁻¹ — принимается.
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams
from control.controllers import PitchController, PitchControlParams, PID, PIDParams
from control.blocks import (Diagram, SchemeError, pitch_standard, pitch_blank,
                            _blk, _w, default_params)

DT = 0.01
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {info}")


def run_pitch(pit, n=3000, seed=1):
    """Шаблон и PitchController на одних входах; возвращает max |Δδe|, °, и долю упора."""
    ac = AircraftParams()
    de_trim = math.radians(-0.7)
    ctrl = PitchController(ac, pit)
    ctrl.set_trim_elevator(de_trim)
    ctrl.reset({"theta": 0.0, "q": 0.0, "h": 100.0})
    dia = Diagram(pitch_standard(ac, pit))
    rng = np.random.default_rng(seed)
    err, sat = 0.0, 0
    for k in range(n):
        th_ref = math.radians(rng.uniform(-25, 25)) if k % 200 == 0 or k == 0 else th_ref
        th = th_ref + math.radians(rng.normal(0, 25))
        q = math.radians(rng.normal(0, 30))
        Va = rng.uniform(9, 30)
        ctrl.set_pitch_setpoint(th_ref)
        de_ref = ctrl.step(k * DT, {"theta": th, "q": q, "h": 100.0, "Va": Va}, DT)[0]
        sig = {"theta_ref": math.degrees(th_ref), "theta": math.degrees(th),
               "q": math.degrees(q), "Va": Va, "de_trim": math.degrees(de_trim)}
        de = dia.commands(dia.step(sig, k * DT, DT))[0]
        err = max(err, abs(math.degrees(de - de_ref)))
        sat += abs(de_ref) >= ac.delta_e_max - 1e-9
    return err, sat / n


print("1. Шаблон тангажа против PitchController")
e, s = run_pitch(PitchControlParams())
check("штатные коэффициенты (П θ, П q)", e < 1e-9, f"max|Δδe| = {e:.1e}°, упор {100 * s:.0f} %")
e, s = run_pitch(PitchControlParams(theta_Ki=0.8, theta_Kd=0.3, q_Ki=0.02, q_Kd=0.002))
check("ПИД в обоих контурах", e < 1e-9, f"max|Δδe| = {e:.1e}°, упор {100 * s:.0f} %")

print("2. Блок ПИД против control.controllers.PID")
pid = PID(PIDParams(Kp=1.3, Ki=0.7, Kd=0.2, tau=0.05))
dia = Diagram({"blocks": [_blk("c", "Input", 0, 0, signal="theta"),
                          _blk("r", "PID", 0, 0, Kp=1.3, Ki=0.7, Kd=0.2, Tf=0.05),
                          _blk("o", "Output", 0, 0, act="de")],
               "wires": [_w("c", "r"), _w("r", "o")]})
rng = np.random.default_rng(2)
err = 0.0
for k in range(2000):
    e_k = rng.normal()
    err = max(err, abs(pid.step(e_k, DT) - dia.step({"theta": e_k}, k * DT, DT)["de"]))
check("совпадение выхода", err < 1e-12, f"max|Δ| = {err:.1e}")


def chain(btype, steps=500, u=1.0, **p):
    """Ступенька u на вход звена → выход по тактам."""
    d = Diagram({"blocks": [_blk("c", "Const", 0, 0, value=u), _blk("x", btype, 0, 0, **p),
                            _blk("o", "Output", 0, 0, act="dt")],
                 "wires": [_w("c", "x"), _w("x", "o")]})
    return np.array([d.step({}, k * DT, DT)["dt"] for k in range(steps)])


print("3. Звенья")
t = np.arange(500) * DT
y = chain("Aperiodic", K=2.0, T=0.5)
check("K/(Ts+1): 2·(1 − e^(−t/T))", np.max(np.abs(y - 2 * (1 - np.exp(-t / 0.5)))) < 1e-12)
y = chain("Washout", T=0.5)
check("Ts/(Ts+1): постоянный вход срезан", np.all(y == 0.0),
      "(старт из установившегося входа: y = 0)")
y = chain("Delay", steps=10, u=3.0, n=5)
check("z⁻⁵: 5 нулей, затем вход", list(y[:5]) == [0.0] * 5 and y[5] == 3.0)
y = chain("LeadLag", T1=0.3, T2=0.05)
check("(T1s+1)/(T2s+1): установившееся = 1", abs(y[-1] - 1.0) < 1e-12)
y = chain("I", Ki=2.0, y_max=0.0)
check("И: 2·t", abs(y[100] - 2.0) < 1e-12)

print("4. Ошибки схемы")


def err_of(scheme):
    try:
        Diagram(scheme)
        return None
    except SchemeError as ex:
        return str(ex)


loop = {"blocks": [_blk("s", "Sum", 0, 0, signs="+-"), _blk("g", "Gain", 0, 0, K=0.5),
                   _blk("c", "Const", 0, 0, value=1.0), _blk("o", "Output", 0, 0, act="de")],
        "wires": [_w("c", "s", 0), _w("s", "g"), _w("g", "s", 1), _w("g", "o")]}
m = err_of(loop)
check("алгебраическая петля — отказ", m and "петля" in m, f"«{m}»")
loop["blocks"].append(_blk("z", "Delay", 0, 0))
loop["wires"] = [_w("c", "s", 0), _w("s", "g"), _w("g", "z"), _w("z", "s", 1), _w("g", "o")]
check("петля через z⁻¹ — принимается", err_of(loop) is None)
d = Diagram(loop)
y = [d.step({}, k * DT, DT)["de"] for k in range(200)]
check("и сходится к K/(1+K) = 1/3", abs(y[-1] - 1 / 3) < 1e-9, f"y = {y[-1]:.6f}")

two = pitch_blank()
two["blocks"].append(_blk("o2", "Output", 0, 0, act="de"))
two["wires"] += [_w("in_t", "out_de"), _w("in_q", "o2")]
m = err_of(two)
check("два выхода на δe — отказ", m and "два блока" in m, f"«{m}»")
two = pitch_blank()
two["wires"] += [_w("in_t", "out_de"), _w("in_q", "out_de")]
m = err_of(two)
check("два провода на вход — отказ", m and "два провода" in m, f"«{m}»")
m = err_of(pitch_blank())
check("заготовка без проводов — «вход не подключён»", m and "не подключён" in m, f"«{m}»")

print("\nALL BLOCK CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
