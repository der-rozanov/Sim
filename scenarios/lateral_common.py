# -*- coding: utf-8 -*-
"""
Общая часть сценариев бокового канала (s12, s13): полная САУ 6DOF.

Продольный канал — как в С6:
  h_ref → [ПИ h] → theta_ref → [ПИД theta + q] → delta_e
  Va_ref → [ПИД Va] → throttle
Боковой канал — LateralController:
  chi_ref → [ПИ chi] → phi_ref → [ПИ phi − Kd·p] → delta_a
  beta (зонд) → [ПИ beta] → delta_r
Навигация (необязательно, s14) — WaypointNavigator: положение (GPS) → chi_ref, h_ref
(B&M гл. 10–11, control/navigation.py); без маршрута — расписание chi_ref_fn(t).

Источники измерений:
  h — барометр, Va — СВС, q/p — гироскоп, theta/phi — ИНС,
  chi — GPS (путевой угол по земной скорости), beta — зонд УС.
"""

import numpy as np

from sim.state import THETA, Q, H, P, PHI, PSI, X, Y, air_data, earth_velocity
from runner import compute_trim
from control.controllers import (PitchController, PitchControlParams,
                                 SpeedController, SpeedControlParams,
                                 LateralController, LateralControlParams,
                                 AltitudeHold, AltitudeHoldParams)
from control.sensors import (measure_gyro, measure_altitude, measure_airspeed,
                             measure_attitude, measure_sideslip, measure_gps_course)



class FullSAU:
    """
    Полная САУ ЛА: высота + скорость + курс с координацией разворота.

    chi_ref_fn(t) -> chi_ref, рад — расписание уставки курса.
    nav — WaypointNavigator: пока у него есть маршрут, курс и высота — от него.
    Буферы *_buf заполняются на каждом шаге — для печати и логгера.
    """

    def __init__(self, aircraft, sp, cfg, rng, chi_ref_fn,
                 lat_params: LateralControlParams = None,
                 h_ref: float = None, Va_ref: float = None, nav=None):
        self.ac, self.sp, self.cfg, self.rng = aircraft, sp, cfg, rng
        self.chi_ref_fn = chi_ref_fn
        self.nav = nav
        self.h_ref  = cfg.h0  if h_ref  is None else h_ref
        self.Va_ref = cfg.Va0 if Va_ref is None else Va_ref

        self.alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)

        self.pitch = PitchController(aircraft, PitchControlParams(Va_ref=cfg.Va0))
        self.pitch.set_trim_throttle(thr_trim)
        self.pitch.set_trim_elevator(de_trim)   # упреждающий балансировочный δe
        self.pitch.reset({'theta': self.alpha_trim, 'q': 0.0, 'h': self.h_ref})

        self.speed = SpeedController(aircraft, SpeedControlParams())
        self.speed.set_trim_throttle(thr_trim)
        self.speed.set_Va_ref(self.Va_ref)
        self.speed.reset()

        self.lat = LateralController(aircraft, lat_params or LateralControlParams())
        self.alt = AltitudeHold(AltitudeHoldParams(), self.alpha_trim)   # ПИ h → θ_ref
        self.lat.reset()

        self.chi_ref_buf, self.phi_ref_buf, self.chi_meas_buf = [], [], []
        self.theta_ref_buf, self.h_ref_buf = [], []

    def __call__(self, t, state, Va, alpha):
        sp, rng, ac = self.sp, self.rng, self.ac

        # ---- Измерения -------------------------------------------------
        h_meas     = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
        Va_meas    = measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng)
        q_meas     = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
        p_meas     = measure_gyro(state[P], sp.gyro_bias, sp.gyro_noise, rng)
        theta_meas = measure_attitude(state[THETA], 0.0, sp.ins_angle_noise, rng)
        phi_meas   = measure_attitude(state[PHI],   0.0, sp.ins_angle_noise, rng)
        Vx, Vy, _  = earth_velocity(state)
        chi_meas   = measure_gps_course(Vx, Vy, sp.gps_vel_noise, rng)
        # УС: истинное значение берётся из того же вектора ветра, что видит
        # физика (runner вызывает controls_fn с Va, alpha этого же шага)
        beta_true  = self._beta_true(t, state)
        beta_meas  = measure_sideslip(beta_true, sp.probe_beta_bias, sp.probe_beta_noise, rng)

        # ---- Навигация (GPS-положение — только с навигатором: s12/s13 без
        # изменений в последовательности шума) ---------------------------
        ref = None
        if self.nav is not None:
            pn_meas = state[X] + rng.normal(0.0, sp.gps_pos_noise)
            pe_meas = state[Y] + rng.normal(0.0, sp.gps_pos_noise)
            ref = self.nav.step(pn_meas, pe_meas, chi_meas)
        if ref is not None:
            chi_ref, self.h_ref = ref
        else:
            chi_ref = self.chi_ref_fn(t)

        # ---- Продольный канал (как С6) ---------------------------------
        theta_ref = self.alt.step(self.h_ref, h_meas, self.cfg.dt)
        self.pitch.set_pitch_setpoint(theta_ref)
        delta_e = self.pitch.step(t, {'q': q_meas, 'theta': theta_meas,
                                      'h': h_meas, 'Va': Va_meas}, self.cfg.dt)[0]
        throttle = self.speed.step(Va_meas, self.cfg.dt)

        # ---- Боковой канал ---------------------------------------------
        self.lat.set_course(chi_ref, self.nav.kappa if ref is not None else 0.0)
        # путевая скорость — по истинной земной скорости (шум GPS скорости ~0.1 м/с
        # на уставку крена не влияет, а лишний вызов rng сдвинул бы шум сценариев)
        delta_a, delta_r = self.lat.step({'chi': chi_meas, 'Vg': np.hypot(Vx, Vy),
                                          'phi': phi_meas, 'p': p_meas, 'Va': Va_meas,
                                          'beta': beta_meas}, self.cfg.dt)

        self.chi_ref_buf.append(self.lat.chi_ref)
        self.phi_ref_buf.append(self.lat.phi_ref)
        self.chi_meas_buf.append(chi_meas)
        self.theta_ref_buf.append(theta_ref)
        self.h_ref_buf.append(self.h_ref)

        return np.array([delta_e, throttle, delta_a, delta_r])

    def bind_wind(self, wind_call):
        """Передать функцию ветра wind(h, t) — нужна для истинного УС зонда."""
        self._wind_call = wind_call

    def _beta_true(self, t, state):
        _, _, beta = air_data(state, self._wind_call(state[H], t))
        return beta


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
