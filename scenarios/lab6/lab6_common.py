# -*- coding: utf-8 -*-
"""
Общие инструменты учебных заданий по Лекции 6 (B&M гл. 6, продольный автопилот).

Студенту этот файл править НЕ нужно. Здесь:
  - варианты заданий (VARIANTS);
  - точная балансировка горизонтального полёта (trim);
  - псевдодатчики (make_sensors);
  - численная линеаризация модели ЛА — НЕЗАВИСИМАЯ проверка формул студента
    (конечные разности по derivatives(), без аналитических формул);
  - теоретические отклики 2-го порядка, метрики переходного процесса;
  - готовые контуры для L2/L3 (тангаж, высота), чтобы не зависеть от
    результатов предыдущих заданий.

Симулятор (sim/, control/, runner.py) не изменяется — только импортируется.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import U, W, Q, THETA, H, N_STATES, air_velocity
from sim.dynamics import derivatives
from control.sensors import measure_gyro, measure_altitude, measure_airspeed


# ---------------------------------------------------------------------------
# Варианты заданий
# ---------------------------------------------------------------------------
# Va       — скорость балансировки (точка линеаризации), м/с
# dtheta   — ступенька по тангажу в L1, град
# wn_theta, zeta_theta — желаемые параметры контура тангажа (L1)
# W_h, zeta_h          — разнос полос и демпфирование контура высоты (L2)
# dh       — ступенька по высоте в L2/L3, м
# wn_V, zeta_V         — параметры контура скорости (L3)
# dVa      — ступенька по воздушной скорости в L3, м/с

VARIANTS = {
    1: dict(Va=25.0, dtheta=5.0, wn_theta=5.0, zeta_theta=0.707, W_h=10.0, zeta_h=0.9, dh=10.0, wn_V=0.8, zeta_V=0.9, dVa=3.0),
    2: dict(Va=22.0, dtheta=4.0, wn_theta=4.5, zeta_theta=0.8,   W_h=10.0, zeta_h=1.0, dh=8.0,  wn_V=0.7, zeta_V=1.0, dVa=3.0),
    3: dict(Va=28.0, dtheta=5.0, wn_theta=6.0, zeta_theta=0.707, W_h=12.0, zeta_h=0.8, dh=12.0, wn_V=1.0, zeta_V=0.8, dVa=-3.0),
    4: dict(Va=30.0, dtheta=6.0, wn_theta=6.0, zeta_theta=0.9,   W_h=10.0, zeta_h=0.9, dh=15.0, wn_V=0.8, zeta_V=0.9, dVa=-4.0),
    5: dict(Va=24.0, dtheta=5.0, wn_theta=5.0, zeta_theta=0.6,   W_h=8.0,  zeta_h=1.0, dh=10.0, wn_V=0.6, zeta_V=1.0, dVa=4.0),
    6: dict(Va=26.0, dtheta=4.0, wn_theta=5.5, zeta_theta=0.8,   W_h=10.0, zeta_h=0.8, dh=10.0, wn_V=0.9, zeta_V=0.8, dVa=2.0),
}


def get_variant(n: int) -> dict:
    if n not in VARIANTS:
        raise ValueError(f"Нет варианта {n}. Доступны: {sorted(VARIANTS)}")
    v = dict(VARIANTS[n])
    print(f"Вариант {n}: " + ", ".join(f"{k}={val}" for k, val in v.items()))
    return v


# ---------------------------------------------------------------------------
# Балансировка
# ---------------------------------------------------------------------------

def _no_wind(h, t):
    return (0.0, 0.0)


def trim(aircraft: AircraftParams, Va: float, h0: float = 100.0) -> dict:
    """
    Точная балансировка горизонтального полёта на скорости Va
    (Ньютон по [alpha, delta_e, throttle], gamma = 0 → theta = alpha).

    Возвращает dict: state, alpha, theta, delta_e, throttle, Va.
    """
    def make_state(a):
        s = np.zeros(N_STATES)
        s[U], s[W], s[THETA], s[H] = Va * np.cos(a), Va * np.sin(a), a, h0
        return s

    def resid(z):
        a, de, dt = z
        d = derivatives(make_state(a), np.array([de, dt]), 0.0, aircraft, _no_wind)
        return np.array([d[U], d[W], d[Q]])

    z = np.array([0.05, -0.05, 0.4])
    for _ in range(50):
        r = resid(z)
        if np.max(np.abs(r)) < 1e-10:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            dz = np.zeros(3)
            dz[j] = 1e-7
            J[:, j] = (resid(z + dz) - r) / 1e-7
        z = z - np.linalg.solve(J, r)

    a, de, dt = z
    if not (aircraft.throttle_min <= dt <= aircraft.throttle_max):
        raise RuntimeError(f"Трим на Va={Va} м/с недостижим (throttle={dt:.3f})")
    return dict(state=make_state(a), alpha=a, theta=a, delta_e=de, throttle=dt, Va=Va)


def print_trim(tr: dict):
    print(f"Трим Va={tr['Va']:.1f} м/с:  alpha*={np.degrees(tr['alpha']):.2f}°  "
          f"theta*={np.degrees(tr['theta']):.2f}°  delta_e*={np.degrees(tr['delta_e']):.2f}°  "
          f"throttle*={tr['throttle']:.3f}")


# ---------------------------------------------------------------------------
# Датчики
# ---------------------------------------------------------------------------

def make_sensors(noise: bool = True, seed: int = 42):
    """
    Возвращает функцию meas(state, Va) -> dict(theta, q, h, Va).
    theta — ИНС (упрощённо: истина + шум уровня гироскопа), как в s5.
    """
    sp = SensorParams()
    if not noise:
        sp = SensorParams(gyro_noise=0.0, baro_noise=0.0, airspeed_noise=0.0)
    rng = np.random.default_rng(seed)

    def meas(state, Va):
        return dict(
            theta=state[THETA] + (rng.normal(0.0, sp.gyro_noise) if sp.gyro_noise > 0 else 0.0),
            q=measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng),
            h=measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng),
            Va=measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng),
        )
    return meas


# ---------------------------------------------------------------------------
# Численная линеаризация (эталон для самопроверки)
# ---------------------------------------------------------------------------

def _f(state, de, dt, aircraft):
    return derivatives(state, np.array([de, dt]), 0.0, aircraft, _no_wind)


def numeric_pitch_coeffs(aircraft: AircraftParams, tr: dict) -> tuple:
    """
    a_theta1, a_theta2, a_theta3 из конечных разностей q_dot:
      q_dot ≈ −a1·q − a2·alpha + a3·delta_e   (около трима, Va = const)
    """
    s0, de0, dt0 = tr["state"], tr["delta_e"], tr["throttle"]
    Va, a0 = tr["Va"], tr["alpha"]
    f0 = _f(s0, de0, dt0, aircraft)[Q]
    eps = 1e-6

    s = s0.copy(); s[Q] += eps
    a1 = -(_f(s, de0, dt0, aircraft)[Q] - f0) / eps

    s = s0.copy(); s[U], s[W] = Va * np.cos(a0 + eps), Va * np.sin(a0 + eps)
    a2 = -(_f(s, de0, dt0, aircraft)[Q] - f0) / eps

    a3 = (_f(s0, de0 + eps, dt0, aircraft)[Q] - f0) / eps
    return a1, a2, a3


def numeric_speed_coeffs(aircraft: AircraftParams, tr: dict) -> tuple:
    """
    a_V1, a_V2 из конечных разностей Va_dot (alpha, theta = const):
      Va_dot ≈ −aV1·Va + aV2·delta_t
    """
    s0, de0, dt0 = tr["state"], tr["delta_e"], tr["throttle"]
    a0 = tr["alpha"]

    def Va_dot(Va, dt):
        s = s0.copy()
        s[U], s[W] = Va * np.cos(a0), Va * np.sin(a0)
        d = _f(s, de0, dt, aircraft)
        return (s[U] * d[U] + s[W] * d[W]) / Va

    eps = 1e-6
    f0 = Va_dot(tr["Va"], dt0)
    aV1 = -(Va_dot(tr["Va"] + eps, dt0) - f0) / eps
    aV2 = (Va_dot(tr["Va"], dt0 + eps) - f0) / eps
    return aV1, aV2


# ---------------------------------------------------------------------------
# Теория: отклики линейных систем 2-го порядка
# ---------------------------------------------------------------------------

def lti2_step(t, b1, b0, a1, a0, amp=1.0, t_step=0.0):
    """
    Отклик y(t) системы (b1·s + b0)/(s² + a1·s + a0) на ступеньку amp в t_step.
    Интегрируется RK4 мелким шагом (без scipy).
    """
    y = np.zeros_like(t)
    x = np.zeros(2)                         # управляемая каноническая форма
    A = np.array([[0.0, 1.0], [-a0, -a1]])
    B = np.array([0.0, 1.0])
    C = np.array([b0, b1])
    f = lambda x, u: A @ x + B * u
    n_sub = 10
    for i in range(1, len(t)):
        h = (t[i] - t[i - 1]) / n_sub
        for k in range(n_sub):
            tk = t[i - 1] + k * h
            u = amp if tk >= t_step else 0.0
            k1 = f(x, u); k2 = f(x + h / 2 * k1, u)
            k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[i] = C @ x
    return y


# ---------------------------------------------------------------------------
# Метрики переходного процесса
# ---------------------------------------------------------------------------

def step_metrics(t, y, y0, y_target, t_step, band=0.05):
    """
    Перерегулирование [%], время установления в трубку ±band [с],
    установившееся значение (среднее за последние 10% прогона) и ошибка.
    """
    dy = y_target - y0
    m = t >= t_step
    tt, yy = t[m], y[m]
    rel = (yy - y0) / dy                          # 0 → 1
    overshoot = max(0.0, (rel.max() - 1.0) * 100.0)
    outside = np.abs(rel - 1.0) > band
    if not outside.any():
        t_settle = 0.0
    elif outside[-1]:
        t_settle = np.inf
    else:
        t_settle = tt[np.where(outside)[0][-1] + 1] - t_step
    n_tail = max(1, len(yy) // 10)
    y_ss = yy[-n_tail:].mean()
    return dict(overshoot=overshoot, t_settle=t_settle, y_ss=y_ss, e_ss=y_target - y_ss)


# ---------------------------------------------------------------------------
# Самопроверка
# ---------------------------------------------------------------------------

_results = []


def check(name: str, ok: bool, detail: str = ""):
    mark = "✓" if ok else "✗"
    _results.append(ok)
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""))
    return ok


def check_close(name, value, ref, rtol=0.03):
    if value is None:
        return check(name, False, "не реализовано")
    ok = np.isfinite(value) and abs(value - ref) <= rtol * abs(ref)
    return check(name, ok, f"ваше {value:+.4f}, модель {ref:+.4f}")


def check_summary():
    n_ok, n = sum(_results), len(_results)
    print(f"\nИтог самопроверки: {n_ok}/{n}")
    if n and n_ok == n:
        print("Всё сходится. Переходите к исследовательским вопросам в начале файла.")


def check_flight_safe(log, aircraft, t_end):
    """ЛА долетел до конца прогона и не вышел за предупредительный УА."""
    check("ЛА не упал", log.t[-1] >= t_end - 0.05, f"конец прогона {log.t[-1]:.1f} с из {t_end:.0f} с")
    a_max = np.degrees(log.alpha.max())
    check("полёт без выхода на α_warning", a_max < np.degrees(aircraft.alpha_warning),
          f"α_max = {a_max:.1f}°, порог {np.degrees(aircraft.alpha_warning):.0f}°")


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab6")


def save_fig(fig, name):
    """Сохранить рисунок в results/lab6/<name>.png (для отчёта)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(RESULTS_DIR, name + ".png"))
    fig.savefig(path, dpi=120)
    print(f"Рисунок сохранён: {path}")


# ---------------------------------------------------------------------------
# Готовые контуры (для L2, L3)
# ---------------------------------------------------------------------------

class PitchLoopReady:
    """
    Готовый контур тангажа по B&M (слайд 19) — для L2/L3.
    delta_e = delta_e* + kp·(theta_c − theta) − kd·q,  насыщение руля.
    Коэффициенты считаются по численно линеаризованной модели.
    """

    def __init__(self, aircraft, tr, wn, zeta):
        a1, a2, a3 = numeric_pitch_coeffs(aircraft, tr)
        self.kp = (wn**2 - a2) / a3
        self.kd = (2 * zeta * wn - a1) / a3
        self.K_dc = self.kp * a3 / (a2 + self.kp * a3)
        self.de_trim = tr["delta_e"]
        self.de_min, self.de_max = aircraft.delta_e_min, aircraft.delta_e_max

    def __call__(self, theta_c, theta, q):
        de = self.de_trim + self.kp * (theta_c - theta) - self.kd * q
        return float(np.clip(de, self.de_min, self.de_max))


class AltitudeLoopReady:
    """
    Готовый контур высоты (ПИ, слайды 17, 21–22) — для L3.
    theta_c = theta* + kp·e + ki·∫e,  e = sat(h_c, h ± h_hold) − h.
    """

    def __init__(self, tr, wn_theta, K_dc, W_h, zeta_h, h_hold=15.0,
                 theta_max=np.radians(20.0)):
        wn_h = wn_theta / W_h
        self.kp = 2 * zeta_h * wn_h / (K_dc * tr["Va"])
        self.ki = wn_h**2 / (K_dc * tr["Va"])
        self.theta_trim = tr["theta"]
        self.h_hold, self.theta_max = h_hold, theta_max
        self.I = 0.0

    def __call__(self, h_c, h, dt):
        e = np.clip(h_c, h - self.h_hold, h + self.h_hold) - h
        self.I += e * dt
        th = self.theta_trim + self.kp * e + self.ki * self.I
        th_sat = float(np.clip(th, -self.theta_max, self.theta_max))
        if th_sat != th:
            self.I -= e * dt              # анти-виндап: не копить при насыщении
        return th_sat
