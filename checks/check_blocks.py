# -*- coding: utf-8 -*-
"""
Проверка структурных схем control/blocks.py (лабораторная «Конструктор САУ»):

  1. шаблоны штатных каналов ≡ регуляторы control/controllers.py на одной
     последовательности случайных входов:
       тангаж ≡ PitchController (П и ПИД, с упором руля),
       скорость ≡ SpeedController, высота + тангаж ≡ AltitudeHold + PitchController,
       крен ≡ RollController, курс + крен + скольжение ≡ LateralController
     (у высоты, крена — без упора: анти-виндап по выходу канала блоками не
     повторяется, см. docstring шаблонов);
  2. блок ПИД ≡ control.controllers.PID (без ограничения выхода; с пределом И);
  3. звенья: апериодическое — переходная функция 1 − e^(−t/T);
     ФВЧ — постоянный вход срезает (старт из установившегося); z⁻ⁿ — сдвиг на n;
     корректирующее в установившемся режиме — коэффициент 1; крен разворота;
  4. ошибки схемы: алгебраическая петля, два провода на вход, два выхода на один
     привод; петля с z⁻¹ — принимается;
  5. полёт (замкнутый контур, полная 6DOF-модель): «все каналы» из блоков против
     штатной SAU(lon="split") — ступеньки h, Va, χ.
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig
from sim.state import H, air_data, earth_velocity
from control.controllers import (PitchController, PitchControlParams, PID, PIDParams,
                                 SpeedController, SpeedControlParams, AltitudeHold,
                                 AltitudeHoldParams, RollController, RollControlParams,
                                 LateralController, LateralControlParams)
from control.blocks import (Diagram, SchemeError, pitch_standard, pitch_blank, speed_standard,
                            altitude_standard, roll_standard, lateral_standard, full_standard,
                            sau_signals, templates, _blk, _w)
from control.sau import SAU, SAUParams, TruthSensors, Mode
from runner import run, compute_trim, trim_state

DT = 0.01
deg, rad = math.degrees, math.radians
ok = True
AC = AircraftParams()


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {info}")


def run_pitch(pit, n=3000, seed=1):
    """Шаблон и PitchController на одних входах; возвращает max |Δδe|, °, и долю упора."""
    de_trim = rad(-0.7)
    ctrl = PitchController(AC, pit)
    ctrl.set_trim_elevator(de_trim)
    ctrl.reset({"theta": 0.0, "q": 0.0, "h": 100.0})
    dia = Diagram(pitch_standard(AC, pit))
    rng = np.random.default_rng(seed)
    err, sat = 0.0, 0
    for k in range(n):
        th_ref = rad(rng.uniform(-25, 25)) if k % 200 == 0 or k == 0 else th_ref
        th = th_ref + rad(rng.normal(0, 25))
        q = rad(rng.normal(0, 30))
        Va = rng.uniform(9, 30)
        ctrl.set_pitch_setpoint(th_ref)
        de_ref = ctrl.step(k * DT, {"theta": th, "q": q, "h": 100.0, "Va": Va}, DT)[0]
        sig = {"theta_ref": deg(th_ref), "theta": deg(th), "q": deg(q), "Va": Va,
               "de_trim": deg(de_trim)}
        de = dia.commands(dia.step(sig, k * DT, DT))[0]
        err = max(err, abs(deg(de - de_ref)))
        sat += abs(de_ref) >= AC.delta_e_max - 1e-9
    return err, sat / n


print("1. Шаблоны штатных каналов против регуляторов")
e, s = run_pitch(PitchControlParams())
check("тангаж: штатные коэффициенты (П θ, П q)", e < 1e-9, f"max|Δδe| = {e:.1e}°, упор {100 * s:.0f} %")
e, s = run_pitch(PitchControlParams(theta_Ki=0.8, theta_Kd=0.3, q_Ki=0.02, q_Kd=0.002))
check("тангаж: ПИД в обоих контурах", e < 1e-9, f"max|Δδe| = {e:.1e}°, упор {100 * s:.0f} %")

# Скорость: ступеньки Va_ref, интеграл доходит до предела (газ в упоре)
spd = SpeedController(AC, SpeedControlParams())
spd.set_trim_throttle(0.48)
dia = Diagram(speed_standard(AC))
rng = np.random.default_rng(3)
err, sat = 0.0, 0
for k in range(6000):
    Vr = rng.uniform(10, 25) if k % 1000 == 0 else Vr
    Va = Vr + rng.normal(0, 2.0) + (6.0 if (k // 1000) % 2 else 0.0)  # смещение — интеграл копится
    spd.set_Va_ref(Vr)
    ref = spd.step(Va, DT)
    out = dia.step({"Va_ref": Vr, "Va": Va, "dt_trim": 0.48}, k * DT, DT)["dt"]
    err = max(err, abs(out - ref))
    sat += ref in (AC.throttle_min, AC.throttle_max)
check("скорость ≡ SpeedController", err < 1e-12, f"max|Δδt| = {err:.1e}, упор {100 * sat / 6000:.0f} %")

# Высота + тангаж: малые ошибки высоты — θ_ref не в упоре
pit, alt = PitchControlParams(), AltitudeHoldParams()
a_trim, de_trim = rad(0.05), rad(-0.3)
ah = AltitudeHold(alt, a_trim)
pc = PitchController(AC, pit)
pc.set_trim_elevator(de_trim)
dia = Diagram(altitude_standard(AC, pit, alt))
rng = np.random.default_rng(4)
err = 0.0
for k in range(4000):
    h = 100.0 + rng.normal(0, 0.3)
    th, q, Va = rad(rng.normal(0, 3)), rad(rng.normal(0, 10)), rng.uniform(12, 22)
    th_ref = ah.step(100.0, h, DT)
    pc.set_pitch_setpoint(th_ref)
    ref = pc.step(0.0, {"theta": th, "q": q, "h": h, "Va": Va}, DT)[0]
    sig = {"h_ref": 100.0, "h": h, "alpha_trim": deg(a_trim), "theta": deg(th), "q": deg(q),
           "Va": Va, "de_trim": deg(de_trim)}
    out = dia.commands(dia.step(sig, k * DT, DT))[0]
    err = max(err, abs(deg(out - ref)))
check("высота + тангаж ≡ AltitudeHold + PitchController", err < 1e-9, f"max|Δδe| = {err:.1e}°")

# Крен: уставки и измерения без упора элеронов
rc = RollController(AC, RollControlParams())
dia = Diagram(roll_standard(AC))
rng = np.random.default_rng(5)
err, pmax = 0.0, 0
for k in range(4000):
    phr = rad(rng.uniform(-30, 30)) if k % 400 == 0 else phr
    phi = phr + rad(rng.normal(0, 4))
    p, Va = rad(rng.normal(0, 20)), rng.uniform(10, 30)
    ref = rc.step(phr, phi, p, Va, DT)
    out = dia.commands(dia.step({"phi_ref": deg(phr), "phi": deg(phi), "p": deg(p), "Va": Va},
                                k * DT, DT))[2]
    err = max(err, abs(deg(out - ref)))
    pmax += abs(rc.p_ref) >= RollControlParams().p_max - 1e-9
check("крен ≡ RollController", err < 1e-9,
      f"max|Δδa| = {err:.1e}°, p_ref в пределе {100 * pmax / 4000:.0f} %")

# Курс + крен + скольжение: старт φ_ref = φ = 0 (ограничитель уставки — с одного значения)
lp = LateralControlParams()
lc = LateralController(AC, lp)
dia = Diagram(lateral_standard(AC, lp))
rng = np.random.default_rng(6)
err_a = err_r = 0.0
chr_ = 0.0
for k in range(6000):
    if k % 600 == 0 and k:
        chr_ = rad(rng.uniform(-180, 180))
    chi = chr_ + rad(rng.normal(0, 5)) if k else 0.0
    kappa = rng.uniform(-0.01, 0.01) if k else 0.0
    m = {"chi": chi, "Vg": rng.uniform(12, 20), "phi": rad(rng.normal(0, 3)) if k else 0.0,
         "p": rad(rng.normal(0, 10)), "Va": rng.uniform(12, 20), "beta": rad(rng.normal(0, 2))}
    lc.set_course(chr_, kappa)
    da_ref, dr_ref = lc.step(m, DT)
    sig = {"chi_ref": deg(chr_), "chi": deg(chi), "Vg": m["Vg"], "chidot_ff": deg(m["Vg"] * kappa),
           "phi": deg(m["phi"]), "p": deg(m["p"]), "Va": m["Va"], "beta": deg(m["beta"])}
    c = dia.commands(dia.step(sig, k * DT, DT))
    err_a = max(err_a, abs(deg(c[2] - da_ref)))
    err_r = max(err_r, abs(deg(c[3] - dr_ref)))
check("курс + крен + скольжение ≡ LateralController", max(err_a, err_r) < 1e-9,
      f"max|Δδa| = {err_a:.1e}°, max|Δδr| = {err_r:.1e}°")

print("2. Блок ПИД против control.controllers.PID")
for name, lim, i_max in (("без пределов", 1e6, 0.0), ("предел интеграла", 0.3, 0.7 * 0.3)):
    pid = PID(PIDParams(Kp=1.3, Ki=0.7, Kd=0.2, tau=0.05, integral_limit=lim))
    dia = Diagram({"blocks": [_blk("c", "Input", 0, 0, signal="theta"),
                              _blk("r", "PID", 0, 0, Kp=1.3, Ki=0.7, Kd=0.2, Tf=0.05, i_max=i_max),
                              _blk("o", "Output", 0, 0, act="de")],
                   "wires": [_w("c", "r"), _w("r", "o")]})
    rng = np.random.default_rng(2)
    err = 0.0
    for k in range(2000):
        e_k = rng.normal() + 0.5                 # смещение — интеграл упирается в предел
        err = max(err, abs(pid.step(e_k, DT) - dia.step({"theta": e_k}, k * DT, DT)["de"]))
    check(f"совпадение выхода ({name})", err < 1e-12, f"max|Δ| = {err:.1e}")


def chain(btype, steps=500, u=1.0, **p):
    """Ступенька u на вход звена → выход по тактам."""
    d = Diagram({"blocks": [_blk("c", "Const", 0, 0, value=u), _blk("x", btype, 0, 0, **p),
                            _blk("o", "Output", 0, 0, act="dt")],
                 "wires": [_w("c", "x"), _w("x", "o")]})
    return np.array([d.step({}, k * DT, DT)["dt"] for k in range(steps)])


print("3. Звенья")
t = np.arange(500) * DT
y = chain("Aperiodic", K=2.0, T=0.5)
check("апериодическое K/(Ts+1): 2·(1 − e^(−t/T))", np.max(np.abs(y - 2 * (1 - np.exp(-t / 0.5)))) < 1e-12)
y = chain("Washout", T=0.5)
check("ФВЧ Ts/(Ts+1): постоянный вход срезан", np.all(y == 0.0),
      "(старт из установившегося входа: y = 0)")
y = chain("Delay", steps=10, u=3.0, n=5)
check("запаздывание z⁻⁵: 5 нулей, затем вход", list(y[:5]) == [0.0] * 5 and y[5] == 3.0)
y = chain("LeadLag", T1=0.3, T2=0.05)
check("корректирующее (T1s+1)/(T2s+1): установившееся = 1", abs(y[-1] - 1.0) < 1e-12)
y = chain("I", Ki=2.0, y_max=0.0)
check("интегратор: 2·t", abs(y[100] - 2.0) < 1e-12)
y = chain("RateLim", steps=3, u=5.0, rate=100.0)
check("ограничение производной: старт с входа", y[0] == 5.0)
d = Diagram({"blocks": [_blk("w", "Const", 0, 0, value=deg(0.5)), _blk("v", "Const", 0, 0, value=16.0),
                        _blk("tb", "TurnBank", 0, 0), _blk("o", "Output", 0, 0, act="da")],
             "wires": [_w("w", "tb", 0), _w("v", "tb", 1), _w("tb", "o")]})
phi = d.step({}, 0.0, DT)["da"]
check("крен разворота: arctg(V·ω/g)", abs(phi - deg(math.atan(16 * 0.5 / 9.81))) < 1e-12,
      f"φ = {phi:.2f}° при V = 16 м/с, ω = 28.6 °/с")

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
two["wires"] += [_w("in_theta", "out_de"), _w("in_q", "o2")]
m = err_of(two)
check("два выхода на δe — отказ", m and "два блока" in m, f"«{m}»")
two = pitch_blank()
two["wires"] += [_w("in_theta", "out_de"), _w("in_q", "out_de")]
m = err_of(two)
check("два провода на вход — отказ", m and "два провода" in m, f"«{m}»")
m = err_of(pitch_blank())
check("заготовка без проводов — «вход не подключён»", m and "не подключён" in m, f"«{m}»")
bad = [name for name, sch in templates(AC, 16.0).items()
       if not name.startswith("Заготовка") and err_of(sch) is not None]
check("все штатные шаблоны принимаются", not bad, ", ".join(bad))

print("5. Полёт: все каналы из блоков против SAU(lon='split')")
cfg = SimConfig(Va0=16.0, h0=100.0, dt=DT, t_end=60.0)
trim = compute_trim(AC, cfg.Va0)
wind = WindParams()


def fly(scheme):
    sau = SAU(AC, trim, TruthSensors(), SAUParams(lon="split"))
    sau.engage(theta=trim[0], h=cfg.h0, Va=cfg.Va0, chi=0.0)
    sau.set_mode(Mode.ALTITUDE)
    dia = Diagram(scheme) if scheme else None

    def fn(t, s, Va, alpha):
        if abs(t - 2.0) < DT / 2:                 # ступеньки уставок
            sau.refs.h += 15.0
            sau.refs.Va += 2.0
            sau.refs.chi = rad(90.0)
            sau.lon.reset_alt()
        _, _, beta = air_data(s, (0.0, 0.0, 0.0))
        c = sau.step(t, s, Va, alpha, beta, DT).copy()
        if dia is not None:
            for idx, v in dia.commands(dia.step(sau_signals(sau, s, alpha, beta), t, DT)).items():
                c[idx] = v
        return c

    return run(fn, AC, wind, cfg, state0=trim_state(AC, cfg))


ref, blk = fly(None), fly(full_standard(AC))
chi = lambda L: np.degrees(np.unwrap(np.arctan2(*np.array([earth_velocity(s)[:2] for s in L.state]).T[::-1])))
dh = np.max(np.abs(ref.state[:, H] - blk.state[:, H]))
dV = np.max(np.abs(ref.Va - blk.Va))
dchi = np.max(np.abs(chi(ref) - chi(blk)))
dphi = np.degrees(np.max(np.abs(ref.phi - blk.phi)))
i_end = len(ref.t) - 1
print(f"     в конце (60 с): h = {ref.state[i_end, H]:.2f} м, Va = {ref.Va[i_end]:.2f} м/с, "
      f"χ = {chi(ref)[-1]:.1f}°")
check("траектории совпадают", len(ref.t) == len(blk.t) and dh < 0.05 and dV < 0.02 and dchi < 0.1,
      f"max|Δh| = {dh:.1e} м, |ΔVa| = {dV:.1e} м/с, |Δχ| = {dchi:.1e}°, |Δφ| = {dphi:.1e}°")

print("\nALL BLOCK CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
