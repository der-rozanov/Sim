# -*- coding: utf-8 -*-
"""
Общая часть сценариев бокового канала (s12–s14): полная САУ 6DOF (control/sau.py).

Продольный канал — TECS (control/tecs.py):
  h_ref, Va_ref → [темпы энергии Ė, Ḃ] → throttle (Ė), theta_ref (Ḃ) → [каскад theta → q] → delta_e
Боковой канал — LateralController:
  chi_ref → [П chi] → phi_ref → [phi → p_ref → FF + ПИ p] → delta_a
  beta (зонд) → [ПИ beta] + микс δa → delta_r
Навигация (необязательно, s14) — WaypointNavigator: положение (GPS) → chi_ref, h_ref
(B&M гл. 10–11, control/navigation.py); без маршрута — расписание chi_ref_fn(t).

Источники измерений:
  h — барометр, ḣ — GPS, Va — СВС, q/p — гироскоп, theta/phi — ИНС,
  chi — GPS (путевой угол по земной скорости), beta — зонд УС.
"""

import numpy as np

from sim.state import H, air_data, earth_velocity
from runner import compute_trim
from control.controllers import PitchControlParams, LateralControlParams
from control.sau import SAU, SAUParams, NoisySensors, Mode


class FullSAU:
    """
    Полная САУ ЛА для сценариев: control.sau.SAU с псевдодатчиками (NoisySensors)
    и функцией controls_fn(t, state, Va, alpha) для runner.run().

    chi_ref_fn(t) -> chi_ref, рад — расписание уставки курса.
    nav — WaypointNavigator: пока у него есть маршрут, курс и высота — от него.
    Буферы *_buf заполняются на каждом шаге — для печати и логгера.
    """

    def __init__(self, aircraft, sp, cfg, rng, chi_ref_fn,
                 lat_params: LateralControlParams = None,
                 h_ref: float = None, Va_ref: float = None, nav=None):
        self.cfg, self.chi_ref_fn, self.nav = cfg, chi_ref_fn, nav
        trim = compute_trim(aircraft, cfg.Va0)
        self.alpha_trim = trim[0]
        params = SAUParams(pitch=PitchControlParams(Va_ref=cfg.Va0),
                           lateral=lat_params or LateralControlParams())
        # GPS-положение — только с навигатором: s12/s13 без изменений в последовательности шума
        sensors = NoisySensors(sp, rng, gps_pos=nav is not None)
        self.sau = SAU(aircraft, trim, sensors, params, nav=nav)
        self.sau.engage(theta=self.alpha_trim, h=cfg.h0 if h_ref is None else h_ref,
                        Va=cfg.Va0 if Va_ref is None else Va_ref, chi=0.0)
        self.sau.set_mode(Mode.ROUTE if nav is not None else Mode.ALTITUDE)

        self.chi_ref_buf, self.phi_ref_buf, self.chi_meas_buf = [], [], []
        self.theta_ref_buf, self.h_ref_buf = [], []

    @property
    def h_ref(self):
        return self.sau.refs.h

    def __call__(self, t, state, Va, alpha):
        sau = self.sau
        sau.refs.chi = self.chi_ref_fn(t)   # в ROUTE заменяется уставкой навигатора
        # УС: истинное значение берётся из того же вектора ветра, что видит
        # физика (runner вызывает controls_fn с Va, alpha этого же шага)
        _, _, beta = air_data(state, self._wind_call(state[H], t))
        controls = sau.step(t, state, Va, alpha, beta, self.cfg.dt)

        self.chi_ref_buf.append(sau.lat.chi_ref)
        self.phi_ref_buf.append(sau.lat.phi_ref)
        self.chi_meas_buf.append(sau.est["chi"])
        self.theta_ref_buf.append(sau.theta_ref)
        self.h_ref_buf.append(sau.refs.h)
        return controls

    def bind_wind(self, wind_call):
        """Передать функцию ветра wind(h, t) — нужна для истинного УС зонда."""
        self._wind_call = wind_call


def course_of(log):
    """Истинный путевой угол chi(t) по земной скорости, рад."""
    chi = np.empty(len(log.t))
    for i, s in enumerate(log.state):
        Vx, Vy, _ = earth_velocity(s)
        chi[i] = np.arctan2(Vy, Vx)
    return chi


def settle_time(t, err, tol, t0):
    """Время установления |err| ≤ tol после момента t0 (последний выход из трубки)."""
    mask = t >= t0
    tt, ee = t[mask], np.abs(err[mask])
    out = np.where(ee > tol)[0]
    if len(out) == 0:
        return 0.0
    if out[-1] == len(ee) - 1:
        return np.nan                     # так и не установился
    return tt[out[-1] + 1] - t0
