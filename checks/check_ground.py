# -*- coding: utf-8 -*-
"""
Проверка контакта с землёй (sim/ground.py, docs/physics.md 4.6) на ровной земле:

  1. Стоянка: осадка стоек против статики (доли веса по плечам), ЛА неподвижен.
  2. Разбег и взлёт: полный газ, руль высоты на себя после 9 м/с.
  3. Посадка: планирование на малом газе с выравниванием — скорость касания,
     пробег, касаний конструкцией нет.
  4. Разворот на земле рулём направления (носовое колесо). Быстрее ≈ 3.5 м/с
     на том же δr ЛА опрокидывается на законцовку: линия опрокидывания
     (носовое — основное колесо) в 0.16 м от ЦМ при высоте ЦМ 0.27 м → 0.59 g.
  5. Склон 6°: ЛА без тормозов катится вниз, касается только колёсами.
  6. Склон 2° (меньше угла трения качения atan 0.05 = 2.9°): ЛА стоит —
     ползучесть от регуляризации трения < 1 см/с.

Запуск: python checks/check_ground.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sim.config import AircraftParams
from sim.state import (U, W, Q, THETA, X, H, V, P, R, PHI, PSI, Y, N_STATES,
                       air_data, earth_velocity)
from sim.integrators import step_rk4
from sim.ground import contacts
from runner import compute_trim

ac = AircraftParams()
DT = 0.01
calm = lambda h, t: (0.0, 0.0, 0.0)
flat = lambda N, E: np.zeros_like(np.asarray(N, float))
ok_all = True


def check(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}  {info}")


def on_ground_state(h_g=0.0):
    s = np.zeros(N_STATES)
    s[H] = h_g + 0.268                         # колёса касаются земли
    return s


def run(s, ctrl_fn, T, ground=flat):
    """Прогон T с; ctrl_fn(t, s) -> controls. Возвращает (t, states, hard_touch)."""
    t, out, hard = 0.0, [s.copy()], False
    for _ in range(int(T / DT)):
        s = step_rk4(s, ctrl_fn(t, s), DT, t, ac, calm, ground)
        t += DT
        c = contacts(s, ac, ground)
        if ((c["depth"] > 0) & ~c["wheel"]).any():
            hard = True
        out.append(s.copy())
    return np.array(out), hard


# --- 1. стоянка ------------------------------------------------------------
print("1. Стоянка (газ 0)")
st, hard = run(on_ground_state(), lambda t, s: np.array([0, 0, 0, 0.0]), 3.0)
c = contacts(st[-1], ac, flat)
wx = np.array(ac.gear_wheels)[:, 0]
mg = ac.mass * ac.g
F_nose = mg * (-wx[1]) / (wx[0] - wx[1])        # равновесие моментов относительно ЦМ
F_main = (mg - F_nose) / 2
d_th = np.array([F_nose, F_main, F_main]) / ac.gear_k
d = c["depth"][:3]
print(f"   осадка, мм: модель {np.round(d * 1e3, 2)}  статика {np.round(d_th * 1e3, 2)}")
print(f"   θ = {np.degrees(st[-1, THETA]):.3f}°,  |V| = {np.linalg.norm(st[-1, [U, V, W]]):.1e} м/с")
check("осадка = статике (±5 %: θ ≠ 0 — стойки разной длины)", np.allclose(d, d_th, rtol=0.05))
check("ЛА неподвижен", np.linalg.norm(st[-1, [U, V, W, P, Q, R]]) < 1e-3)
check("касание только колёсами", not hard)

# --- 2. разбег и взлёт ----------------------------------------------------------
print("2. Разбег и взлёт (газ 100 %, с 10 м/с — тангаж 8°)")
def takeoff(t, s):
    Va = air_data(s, (0, 0, 0))[0]
    theta_ref = np.radians(8.0) if Va > 10.0 else 0.0
    de = -1.0 * (theta_ref - s[THETA]) + 0.08 * s[Q]     # δe < 0 — нос вверх
    return np.array([np.clip(de, -0.43, 0.43), 1.0, 0.0, 0.0])
st, hard = run(on_ground_state(), takeoff, 8.0)
air = st[:, H] > 0.268 + 0.05
i_lo = int(np.argmax(air))
Va_lo = np.hypot(st[i_lo, U], st[i_lo, W])
print(f"   отрыв: t = {i_lo * DT:.2f} с, разбег {st[i_lo, X]:.1f} м, Va = {Va_lo:.1f} м/с; "
      f"h(8 с) = {st[-1, H]:.1f} м")
check("взлетел", air.any() and st[-1, H] > 10)
check("скорость отрыва выше сваливания (7.5 м/с)", Va_lo > 7.5)
check("без касания конструкцией (хвост)", not hard)

# --- 3. посадка -----------------------------------------------------------------
print("3. Посадка: снижение на малом газе с 15 м, выравнивание с 3 м")
a_tr, de_tr, thr_tr = compute_trim(ac, 12.0)
s = np.zeros(N_STATES)
s[U], s[W], s[THETA], s[H] = 12 * np.cos(a_tr), 12 * np.sin(a_tr), a_tr, 15.0
def landing(t, s):
    vz = earth_velocity(s)[2]
    h = s[H] - 0.268
    # тангаж по скорости снижения: −1.5 м/с до 3 м, затем −0.3 м/с (выравнивание)
    vz_ref = -1.5 if h > 3.0 else -0.3
    theta_ref = a_tr + 0.08 * (vz_ref - vz)
    de = de_tr - 1.0 * (theta_ref - s[THETA]) + 0.08 * s[Q]
    return np.array([np.clip(de, -0.43, 0.43), 0.15 if h > 3 else 0.0, 0.0, 0.0])
st, hard = run(s, landing, 40.0)
touch = int(np.argmax(st[:, H] < 0.268 + 1e-3))
vz_touch = earth_velocity(st[touch - 1])[2]
stop = touch + int(np.argmax(np.hypot(st[touch:, U], st[touch:, W]) < 0.3))
print(f"   касание: t = {touch * DT:.2f} с, Vy = {vz_touch:.2f} м/с, "
      f"Va = {np.hypot(st[touch, U], st[touch, W]):.1f} м/с, θ = {np.degrees(st[touch, THETA]):.1f}°")
print(f"   пробег до остановки: {st[stop, X] - st[touch, X]:.0f} м за {(stop - touch) * DT:.1f} с; "
      f"max |φ| = {np.degrees(np.abs(st[touch:, PHI]).max()):.2f}°")
check("скорость касания < gear_v_crash", -vz_touch < ac.gear_v_crash)
check("без касания конструкцией", not hard)
check("остался на земле", st[-1, H] < 0.3)

# --- 4. разворот на земле --------------------------------------------------------
print("4. Руление: газ 20 %, δr = −10° (нос вправо)")
st, hard = run(on_ground_state(), lambda t, s: np.array([0, 0.2, 0, np.radians(-10)]), 15.0)
dpsi = np.degrees(np.unwrap(st[:, PSI]))[-1]
Vg = np.hypot(*earth_velocity(st[-1])[:2])
print(f"   за 15 с: Δψ = {dpsi:.0f}°, Vg = {Vg:.1f} м/с, h = {st[-1, H]:.3f} м")
check("поворачивает вправо", dpsi > 90)
check("без касания конструкцией", not hard)

# --- 5. склон 6° -----------------------------------------------------------------
print("5. Склон 6° (подъём на север), газ 0, без тормозов")
k = np.tan(np.radians(6.0))
slope = lambda N, E: k * np.asarray(N, float)
s = on_ground_state()
s[H] += 0.03
st, hard = run(s, lambda t, s: np.array([0, 0, 0, 0.0]), 5.0, slope)
Vx = earth_velocity(st[-1])[0]
a_th = -ac.g * (np.sin(np.radians(6)) - ac.gear_mu_roll * np.cos(np.radians(6)))
print(f"   Vx(5 с) = {Vx:.2f} м/с (без сопротивления воздуха: {a_th * 5:.2f})")
check("катится вниз (на юг)", Vx < -1.0)
check("без касания конструкцией", not hard)

# --- 6. склон 2° — стоит --------------------------------------------------------
print("6. Склон 2°, газ 0")
k = np.tan(np.radians(2.0))
slope = lambda N, E: k * np.asarray(N, float)
st, hard = run(on_ground_state(), lambda t, s: np.array([0, 0, 0, 0.0]), 5.0, slope)
Vx = earth_velocity(st[-1])[0]
print(f"   Vx(5 с) = {Vx * 100:.2f} см/с")
check("стоит (ползучесть < 1 см/с)", abs(Vx) < 0.01)

print("\nALL GROUND CHECKS PASSED" if ok_all else "\nSOME GROUND CHECKS FAILED")
