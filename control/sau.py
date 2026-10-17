"""
САУ ЛА одним классом: датчики → оценка состояния → наведение → контуры → команды рулей.

    SAU.step(t, state, Va, alpha, beta, dt) -> [δe, δt, δa, δr]

Звенья — отдельные объекты, каждое заменяется другим с тем же методом:

    sensors    Truth|Noisy|RealisticSensors    measure(t, state, Va, alpha, beta) -> meas (dict)
    estimator  None — оценка = измерения       update(meas, dt) -> est (dict с теми же ключами)
    nav        WaypointNavigator | None        step(pn, pe, chi) -> (chi_ref, h_ref) | None; .kappa
    lon        TECS | AltSpeedHold             step(refs, est, dt, hold_alt) -> (θ_cmd, δt);
                                               reset(), reset_alt(), thr_trim
    prot       AngleOfAttackProtector | None   step(alpha, θ_cmd, δt_trim, dt) -> AUAOutput
    pitch      PitchController                 θ_ref → δe
    lat        LateralController               χ_ref → φ_ref → δa  (lat.roll — RollController);
                                               β → δr

Ключи meas / est: t, h, hdot, Va, q, p, theta, phi, chi, Vg, beta, [alpha], [pn, pe].

Режимы (Mode): PITCH — уставка тангажа и курса; ALTITUDE — высоты и курса;
ROUTE — курс и высота от навигатора. Скорость — во всех режимах (refs.Va).

Куда встают доработки (docs/control.md, раздел 9):
    TECS (высота + скорость энергией)    → lon = TECS (control/tecs.py), по умолчанию
    тангаж по схеме крена (θ → q_ref → FF + ПИ, упреждение q в развороте) → новый pitch
    взлёт, посадка, кружение, возврат домой → новые значения Mode + ветки в _guidance / _outer
    L1 / NPFG / траектории Дубинса        → новый nav с тем же step и kappa
    частоты, задержки, дрейф датчиков     → RealisticSensors (сделано, DEV_PLAN п. 5.2)
    комплементарный фильтр / EKF          → estimator
    LQR / INDI / NMPC                     → подкласс SAU с переопределённым _control
                                            (или новые lon / pitch / lat.roll)
    привод рулей — часть объекта управления, а не САУ: место ему в sim/ (решение автора)
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from sim.state import THETA, Q, H, P, R, PHI, PSI, X, Y, earth_velocity, rotation_body_to_earth
from .controllers import (PitchController, PitchControlParams,
                          SpeedController, SpeedControlParams,
                          LateralController, LateralControlParams,
                          AltitudeHold, AltitudeHoldParams)
from .sensors import (measure_gyro, measure_altitude, measure_airspeed, measure_attitude,
                      measure_sideslip, measure_gps_course, measure_angle_of_attack,
                      GaussMarkov, SampledChannel)
from .tecs import TECS, TECSParams


class Mode(Enum):
    PITCH = "тангаж + курс"
    ALTITUDE = "высота + курс"
    ROUTE = "маршрут"


@dataclass
class Refs:
    """Уставки САУ. theta — только в режиме PITCH; chi, h в режиме ROUTE — от навигатора."""
    theta: float = 0.0   # рад
    h: float = 100.0     # м
    Va: float = 16.0     # м/с
    chi: float = 0.0     # рад, путевой угол
    kappa: float = 0.0   # 1/м, кривизна заданной траектории (упреждение крена)
    hdot: float = 0.0    # м/с, производная уставки высоты (упреждение TECS; от навигатора)


@dataclass
class SAUParams:
    """
    Параметры контуров (поля — как у control.tuning.SAUDesign) и пределы высоты.
    lon — продольный внешний контур: "tecs" (TECS) или "split" (AltSpeedHold:
    раздельные ПИ высоты → θ и скорости → газ, B&M гл. 6).
    """
    lon: str = "tecs"
    tecs: TECSParams = field(default_factory=TECSParams)
    pitch: PitchControlParams = field(default_factory=PitchControlParams)
    speed: SpeedControlParams = field(default_factory=SpeedControlParams)
    lateral: LateralControlParams = field(default_factory=LateralControlParams)
    altitude: AltitudeHoldParams = field(default_factory=AltitudeHoldParams)
    h_min: float = -np.inf   # м, пределы h_ref от навигатора
    h_max: float = np.inf


# ---------------------------------------------------------------------------
# Датчики
# ---------------------------------------------------------------------------

class TruthSensors:
    """Истинные значения без шума (3D-тренажёр, стенд)."""

    def measure(self, t, state, Va, alpha, beta):
        Vx, Vy, _ = earth_velocity(state)
        return {"t": t, "h": state[H], "hdot": earth_velocity(state)[2], "Va": Va, "q": state[Q], "p": state[P],
                "theta": state[THETA], "phi": state[PHI],
                "chi": np.arctan2(Vy, Vx), "Vg": np.hypot(Vx, Vy),
                "beta": beta, "alpha": alpha, "pn": state[X], "pe": state[Y]}


class NoisySensors:
    """
    Псевдодатчики control/sensors.py с шумом и смещением:
    h — барометр, Va — ПВД, q/p — гироскоп, θ/φ — ИНС, χ — GPS, β — зонд УС,
    [pn, pe — GPS, если gps_pos], [α — зонд УА, если alpha_probe],
    [ḣ — вертикальная скорость GPS, если gps_vz: нужна TECS].
    Порядок вызовов rng закреплён (воспроизводимость s8–s10): необязательные
    измерения — в конце. Vg — истинная (шум GPS скорости на уставку крена не влияет).
    """

    def __init__(self, sensor_params, rng, gps_pos=False, alpha_probe=False, gps_vz=True):
        self.sp, self.rng = sensor_params, rng
        self.gps_pos, self.alpha_probe, self.gps_vz = gps_pos, alpha_probe, gps_vz

    def measure(self, t, state, Va, alpha, beta):
        sp, rng = self.sp, self.rng
        Vx, Vy, Vh = earth_velocity(state)
        m = {"t": t,
             "h":     measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng),
             "Va":    measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng),
             "q":     measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng),
             "p":     measure_gyro(state[P], sp.gyro_bias, sp.gyro_noise, rng),
             "theta": measure_attitude(state[THETA], 0.0, sp.ins_angle_noise, rng),
             "phi":   measure_attitude(state[PHI], 0.0, sp.ins_angle_noise, rng),
             "chi":   measure_gps_course(Vx, Vy, sp.gps_vel_noise, rng),
             "Vg":    np.hypot(Vx, Vy),
             "beta":  measure_sideslip(beta, sp.probe_beta_bias, sp.probe_beta_noise, rng)}
        if self.gps_pos:
            m["pn"] = state[X] + rng.normal(0.0, sp.gps_pos_noise)
            m["pe"] = state[Y] + rng.normal(0.0, sp.gps_pos_noise)
        if self.alpha_probe:
            m["alpha"] = measure_angle_of_attack(alpha, sp.probe_bias, sp.probe_noise, rng)
        if self.gps_vz:
            m["hdot"] = Vh + rng.normal(0.0, sp.gps_vel_noise)
        return m


class RealisticSensors:
    """
    Реалистичный набор датчиков (3D-режимы, --sensors real; docs/SENSORS.md, раздел 4):
    у каждого датчика — частота отсчётов, у GPS — запаздывание, между отсчётами выход
    удерживается; смещения гироскопа и акселерометра — остаток калибровки (случайный,
    постоянный в полёте) + у гироскопа дрейф (марковский процесс); координаты GPS
    медленно уходят (марковский процесс, B&M 7.5). Параметры — SensorParams.

    Ключи — как у TruthSensors + r (гироскоп), ax, ay, az (акселерометр, м/с², если
    задана accel_fn(t, state) -> удельная сила, sim.dynamics.specific_force),
    vn, ve, vd (GPS-скорость NED), mx, my, mz (магнитометр, Гс, связанная СК).
    θ, φ — «ИНС» как истина + шум (до оценивателя, DEV_PLAN п. 5.3).
    χ, Vg, ḣ — по GPS-скорости. Отметки времени отсчётов t_h, t_Va, t_gps — для
    оценивателя (новый отсчёт, его запаздывание). Новое t меньше прежнего (сброс
    полёта) — reset().
    """

    def __init__(self, sensor_params, rng, accel_fn=None):
        self.sp, self.rng, self.accel_fn = sensor_params, rng, accel_fn
        sp = sensor_params
        self.ch_imu = SampledChannel(1.0 / sp.imu_rate)
        self.ch_baro = SampledChannel(1.0 / sp.baro_rate)
        self.ch_va = SampledChannel(1.0 / sp.airspeed_rate)
        self.ch_gps = SampledChannel(1.0 / sp.gps_rate, sp.gps_delay)
        self.ch_mag = SampledChannel(1.0 / sp.mag_rate)
        I, D = np.radians(sp.mag_incl), np.radians(sp.mag_decl)
        self.mag_ned = sp.mag_field * np.array([np.cos(I) * np.cos(D), np.cos(I) * np.sin(D), np.sin(I)])
        self.reset()

    def reset(self):
        """Новый полёт: новые остатки калибровки, каналы пустые."""
        sp, rng = self.sp, self.rng
        self.gyro_b0 = sp.gyro_bias0 * rng.standard_normal(3) + sp.gyro_bias
        self.accel_b0 = sp.accel_bias0 * rng.standard_normal(3)
        self.gyro_drift = GaussMarkov(sp.gyro_drift_sigma, sp.gyro_drift_tau, rng, 3)
        self.gps_drift = GaussMarkov(sp.gps_pos_drift_sigma, sp.gps_pos_drift_tau, rng, 2)
        for ch in (self.ch_imu, self.ch_baro, self.ch_va, self.ch_gps, self.ch_mag):
            ch.reset()
        self.t_last = None

    def gyro_bias(self):
        """Полное смещение гироскопа (p, q, r) сейчас, рад/с — для проверок и журнала."""
        return self.gyro_b0 + self.gyro_drift.b

    def measure(self, t, state, Va, alpha, beta):
        sp, rng = self.sp, self.rng
        if self.t_last is not None and t < self.t_last:
            self.reset()
        T = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        self.gyro_drift.step(T)
        self.gps_drift.step(T)

        def imu():
            bg = self.gyro_bias()
            m = {"p": state[P] + bg[0] + rng.normal(0.0, sp.gyro_noise),
                 "q": state[Q] + bg[1] + rng.normal(0.0, sp.gyro_noise),
                 "r": state[R] + bg[2] + rng.normal(0.0, sp.gyro_noise),
                 "theta": measure_attitude(state[THETA], 0.0, sp.ins_angle_noise, rng),
                 "phi": measure_attitude(state[PHI], 0.0, sp.ins_angle_noise, rng),
                 "alpha": measure_angle_of_attack(alpha, sp.probe_bias, sp.probe_noise, rng),
                 "beta": measure_sideslip(beta, sp.probe_beta_bias, sp.probe_beta_noise, rng)}
            if self.accel_fn is not None:
                f = self.accel_fn(t, state) + self.accel_b0 + rng.normal(0.0, sp.accel_noise, 3)
                m["ax"], m["ay"], m["az"] = f
            return m

        def gps():
            Vx, Vy, Vh = earth_velocity(state)
            Vn, Ve, Vu = np.array([Vx, Vy, Vh]) + rng.normal(0.0, sp.gps_vel_noise, 3)
            dn, de = self.gps_drift.b + rng.normal(0.0, sp.gps_pos_white, 2)
            return {"pn": state[X] + dn, "pe": state[Y] + de, "hdot": Vu,
                    "chi": np.arctan2(Ve, Vn), "Vg": np.hypot(Vn, Ve),
                    "vn": Vn, "ve": Ve, "vd": -Vu, "t_gps": t}

        m = {"t": t}
        m.update(self.ch_baro.update(t, lambda: {
            "h": measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng), "t_h": t}))
        m.update(self.ch_va.update(t, lambda: {
            "Va": measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng), "t_Va": t}))
        m.update(self.ch_imu.update(t, imu))
        m.update(self.ch_gps.update(t, gps))

        def mag():
            Rbn = rotation_body_to_earth(state[PHI], state[THETA], state[PSI])
            b = Rbn.T @ self.mag_ned + rng.normal(0.0, sp.mag_noise, 3)
            return {"mx": b[0], "my": b[1], "mz": b[2], "t_mag": t}
        m.update(self.ch_mag.update(t, mag))       # последним: порядок шума прочих датчиков прежний
        return m


# ---------------------------------------------------------------------------
# Продольный внешний контур: высота и скорость
# ---------------------------------------------------------------------------

class AltSpeedHold:
    """
    Раздельные контуры (B&M гл. 6): высота → θ_cmd (ПИ AltitudeHold),
    скорость → газ (ПИ SpeedController). В режиме PITCH θ_cmd = refs.theta.
    Замена — TECS (control/tecs.py) с тем же интерфейсом.
    """

    def __init__(self, aircraft, alpha_trim, thr_trim,
                 alt_params: AltitudeHoldParams, speed_params: SpeedControlParams):
        self.alt = AltitudeHold(alt_params, alpha_trim)
        self.speed = SpeedController(aircraft, speed_params)
        self.speed.set_trim_throttle(thr_trim)

    @property
    def thr_trim(self):
        return self.speed.trim_throttle

    def reset(self):
        self.alt.reset()
        self.speed.reset()

    def reset_alt(self):
        self.alt.reset()

    def step(self, refs, est, dt, hold_alt=True):
        theta_cmd = self.alt.step(refs.h, est["h"], dt) if hold_alt else refs.theta
        self.speed.set_Va_ref(refs.Va)
        return theta_cmd, self.speed.step(est["Va"], dt)


# ---------------------------------------------------------------------------
# САУ
# ---------------------------------------------------------------------------

class SAU:
    """
    Полная САУ: режим + уставки → команды рулей и газа.

    Args:
        aircraft:  AircraftParams (ограничения рулей)
        trim:      (α*, δe*, δt*) — балансировка, runner.compute_trim(aircraft, Va)
        sensors:   TruthSensors() | NoisySensors(...)
        params:    SAUParams
        nav:       WaypointNavigator — нужен для режима ROUTE
        prot:      AngleOfAttackProtector — защита по α между lon и pitch
                   (датчики должны давать 'alpha')
        estimator: объект с update(meas, dt) -> est; None — оценка = измерения
    """

    def __init__(self, aircraft, trim, sensors, params: SAUParams = None,
                 nav=None, prot=None, estimator=None):
        self.ac = aircraft
        self.params = params or SAUParams()
        pr = self.params
        self.alpha_trim, self.de_trim, self.thr_trim = trim
        self.sensors, self.estimator, self.nav, self.prot = sensors, estimator, nav, prot

        if pr.lon == "tecs":
            self.lon = TECS(aircraft, pr.tecs)
        elif pr.lon == "split":
            self.lon = AltSpeedHold(aircraft, self.alpha_trim, self.thr_trim,
                                    pr.altitude, pr.speed)
        else:
            raise ValueError(f"lon = {pr.lon!r}: ожидается 'tecs' или 'split'")
        self.pitch = PitchController(aircraft, pr.pitch)
        self.pitch.set_trim_throttle(self.thr_trim)
        self.pitch.set_trim_elevator(self.de_trim)     # упреждающий балансировочный δe
        self.lat = LateralController(aircraft, pr.lateral)

        self.mode = Mode.PITCH
        self.refs = Refs()
        # Последний шаг — для журнала, окна стенда, HUD
        self.est = {}
        self.theta_cmd = self.alpha_trim   # θ от внешнего контура (до защиты по α)
        self.theta_ref = self.alpha_trim   # θ_ref тангажа (после защиты)
        self.prot_out = None
        self.controls = np.array([self.de_trim, self.thr_trim, 0.0, 0.0])

    # --- включение и режимы -------------------------------------------------
    def engage(self, theta, h, Va, chi):
        """Включение САУ: уставки = текущий режим, интегралы с нуля."""
        self.refs.theta, self.refs.h, self.refs.Va, self.refs.chi = theta, h, Va, chi
        self.refs.kappa = 0.0
        self.theta_cmd = theta
        self.pitch.reset({"theta": theta, "q": 0.0, "h": h})
        self.lon.reset()
        self.lat.reset()
        if self.prot is not None:
            self.prot.reset()

    def set_mode(self, mode: Mode):
        """
        Смена режима. Интеграл контура высоты — с нуля (как при новой уставке);
        в PITCH уставка тангажа подхватывает последнюю команду — без скачка.
        """
        if mode is self.mode:
            return
        if mode is Mode.PITCH:
            self.refs.theta = self.theta_cmd
        if mode is Mode.ROUTE and self.nav is None:
            raise ValueError("режим ROUTE требует навигатор (nav)")
        self.lon.reset_alt()
        self.mode = mode

    def fly_route(self, waypoints, start):
        """Маршрут с первой точки: start = (N, E, h) — где ЛА сейчас."""
        self.nav.set_route(waypoints, start)
        self.set_mode(Mode.ROUTE)
        self.lon.reset_alt()

    # --- шаг ------------------------------------------------------------------
    def observe(self, t, state, Va, alpha, beta, dt):
        """
        Датчики → оценка состояния, без управления. На борту оценка работает всегда, и
        при ручном полёте тоже (3D-тренажёр): тогда к включению САУ фильтр уже сошёлся.
        """
        meas = self.sensors.measure(t, state, Va, alpha, beta)
        self.est = meas if self.estimator is None else self.estimator.update(meas, dt)
        return self.est

    def step(self, t, state, Va, alpha, beta, dt):
        """Истинные состояние и воздушные углы → датчики → … → [δe, δt, δa, δr]."""
        est = self.observe(t, state, Va, alpha, beta, dt)
        self._guidance(est)
        self.controls = self._control(est, dt)
        return self.controls

    def _guidance(self, est):
        """
        Уставки курса и высоты от навигатора (ROUTE); маршрут кончился — прежние.
        Упреждение ḣ = уклон участка × путевая скорость вдоль него (не разность h_ref:
        она идёт от шумного GPS-положения).
        """
        self.refs.kappa = self.refs.hdot = 0.0
        if self.mode is not Mode.ROUTE:
            return
        ref = self.nav.step(est["pn"], est["pe"], est["chi"])
        if ref is not None:
            pr = self.params
            self.refs.chi = ref[0]
            self.refs.h = float(np.clip(ref[1], pr.h_min, pr.h_max))
            self.refs.kappa = self.nav.kappa
            self.refs.hdot = self.nav.h_slope * est["Vg"] * np.cos(est["chi"] - self.nav.chi_q)

    def _outer(self, est, dt):
        """Продольный внешний контур → (θ_cmd, δt)."""
        return self.lon.step(self.refs, est, dt, hold_alt=self.mode is not Mode.PITCH)

    def _control(self, est, dt):
        """Каскад: lon → [защита по α] → тангаж; курс → крен, β → РН."""
        theta_cmd, throttle = self._outer(est, dt)
        self.theta_cmd = theta_cmd
        theta_ref = theta_cmd
        if self.prot is not None:
            out = self.prot.step(est["alpha"], theta_cmd, self.lon.thr_trim, dt)
            self.prot_out, theta_ref = out, out.theta_ref
            if out.force_throttle is not None:
                throttle = out.force_throttle
        self.theta_ref = theta_ref

        self.pitch.set_pitch_setpoint(theta_ref)
        delta_e = self.pitch.step(est["t"], {"q": est["q"], "theta": est["theta"],
                                             "h": est["h"], "Va": est["Va"]}, dt)[0]

        self.lat.set_course(self.refs.chi, self.refs.kappa)
        delta_a, delta_r = self.lat.step({k: est[k] for k in
                                          ("chi", "Vg", "phi", "p", "Va", "beta")}, dt)
        return np.array([delta_e, throttle, delta_a, delta_r])
