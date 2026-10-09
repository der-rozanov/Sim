"""
САУ ЛА одним классом: датчики → оценка состояния → наведение → контуры → команды рулей.

    SAU.step(t, state, Va, alpha, beta, dt) -> [δe, δt, δa, δr]

Звенья — отдельные объекты, каждое заменяется другим с тем же методом:

    sensors    TruthSensors | NoisySensors     measure(t, state, Va, alpha, beta) -> meas (dict)
    estimator  None — оценка = измерения       update(meas, dt) -> est (dict с теми же ключами)
    nav        WaypointNavigator | None        step(pn, pe, chi) -> (chi_ref, h_ref) | None; .kappa
    lon        AltSpeedHold                    step(mode, refs, est, dt) -> (θ_cmd, δt)
    prot       AngleOfAttackProtector | None   step(alpha, θ_cmd, δt_trim, dt) -> AUAOutput
    pitch      PitchController                 θ_ref → δe
    lat        LateralController               χ_ref → φ_ref → δa  (lat.roll — RollController);
                                               β → δr

Ключи meas / est: t, h, Va, q, p, theta, phi, chi, Vg, beta, [alpha], [pn, pe].

Режимы (Mode): PITCH — уставка тангажа и курса; ALTITUDE — высоты и курса;
ROUTE — курс и высота от навигатора. Скорость — во всех режимах (refs.Va).

Куда встают доработки (docs/control.md, раздел 12):
    TECS (высота + скорость энергией)    → новый lon с тем же step
    тангаж по схеме крена (θ → q_ref → FF + ПИ, упреждение q в развороте) → новый pitch
    взлёт, посадка, кружение, возврат домой → новые значения Mode + ветки в _guidance / _outer
    L1 / NPFG / траектории Дубинса        → новый nav с тем же step и kappa
    частоты, задержки, дрейф датчиков     → новый sensors (обёртка над NoisySensors)
    комплементарный фильтр / EKF          → estimator
    LQR / INDI / NMPC                     → подкласс SAU с переопределённым _control
                                            (или новые lon / pitch / lat.roll)
    привод рулей — часть объекта управления, а не САУ: место ему в sim/ (решение автора)
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from sim.state import THETA, Q, H, P, PHI, X, Y, earth_velocity
from .controllers import (PitchController, PitchControlParams,
                          SpeedController, SpeedControlParams,
                          LateralController, LateralControlParams,
                          AltitudeHold, AltitudeHoldParams)
from .sensors import (measure_gyro, measure_altitude, measure_airspeed, measure_attitude,
                      measure_sideslip, measure_gps_course, measure_angle_of_attack)


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


@dataclass
class SAUParams:
    """Параметры контуров (поля — как у control.tuning.SAUDesign) и пределы высоты."""
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
        return {"t": t, "h": state[H], "Va": Va, "q": state[Q], "p": state[P],
                "theta": state[THETA], "phi": state[PHI],
                "chi": np.arctan2(Vy, Vx), "Vg": np.hypot(Vx, Vy),
                "beta": beta, "alpha": alpha, "pn": state[X], "pe": state[Y]}


class NoisySensors:
    """
    Псевдодатчики control/sensors.py с шумом и смещением:
    h — барометр, Va — ПВД, q/p — гироскоп, θ/φ — ИНС, χ — GPS, β — зонд УС,
    [pn, pe — GPS, если gps_pos], [α — зонд УА, если alpha_probe].
    Порядок вызовов rng закреплён (воспроизводимость s12–s14): необязательные
    измерения — в конце. Vg — истинная (шум GPS скорости на уставку крена не влияет).
    """

    def __init__(self, sensor_params, rng, gps_pos=False, alpha_probe=False):
        self.sp, self.rng = sensor_params, rng
        self.gps_pos, self.alpha_probe = gps_pos, alpha_probe

    def measure(self, t, state, Va, alpha, beta):
        sp, rng = self.sp, self.rng
        Vx, Vy, _ = earth_velocity(state)
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
        return m


# ---------------------------------------------------------------------------
# Продольный внешний контур: высота и скорость
# ---------------------------------------------------------------------------

class AltSpeedHold:
    """
    Раздельные контуры (B&M гл. 6): высота → θ_cmd (ПИ AltitudeHold),
    скорость → газ (ПИ SpeedController). В режиме PITCH θ_cmd = refs.theta.
    Замена — TECS с тем же step(mode, refs, est, dt) -> (θ_cmd, δt).
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

    def step(self, mode, refs, est, dt):
        theta_cmd = refs.theta if mode is Mode.PITCH else self.alt.step(refs.h, est["h"], dt)
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

        self.lon = AltSpeedHold(aircraft, self.alpha_trim, self.thr_trim, pr.altitude, pr.speed)
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
        self.lon.alt.reset()
        self.mode = mode

    def fly_route(self, waypoints, start):
        """Маршрут с первой точки: start = (N, E, h) — где ЛА сейчас."""
        self.nav.set_route(waypoints, start)
        self.set_mode(Mode.ROUTE)
        self.lon.alt.reset()

    # --- шаг ------------------------------------------------------------------
    def step(self, t, state, Va, alpha, beta, dt):
        """Истинные состояние и воздушные углы → датчики → … → [δe, δt, δa, δr]."""
        meas = self.sensors.measure(t, state, Va, alpha, beta)
        est = meas if self.estimator is None else self.estimator.update(meas, dt)
        self.est = est
        self._guidance(est)
        self.controls = self._control(est, dt)
        return self.controls

    def _guidance(self, est):
        """Уставки курса и высоты от навигатора (ROUTE); маршрут кончился — прежние."""
        self.refs.kappa = 0.0
        if self.mode is not Mode.ROUTE:
            return
        ref = self.nav.step(est["pn"], est["pe"], est["chi"])
        if ref is not None:
            pr = self.params
            self.refs.chi = ref[0]
            self.refs.h = float(np.clip(ref[1], pr.h_min, pr.h_max))
            self.refs.kappa = self.nav.kappa

    def _outer(self, est, dt):
        """Продольный внешний контур → (θ_cmd, δt)."""
        return self.lon.step(self.mode, self.refs, est, dt)

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
