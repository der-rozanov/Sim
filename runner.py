"""
Главный цикл симуляции. Возвращает лог всех переменных.
"""

from dataclasses import dataclass
import numpy as np

from config import AircraftParams, WindParams, SimConfig
from state import initial_state, air_velocity, total_energy, U, W, Q, THETA, H, N_STATES
from wind import wind as _wind
from integrators import step_rk4


@dataclass
class Log:
    """Результат одного прогона симулятора."""
    t:        np.ndarray   # (N,)      время, с
    state:    np.ndarray   # (N, 6)    вектор состояния
    controls: np.ndarray   # (N, 2)    [delta_e рад, throttle 0-1]
    Va:       np.ndarray   # (N,)      воздушная скорость, м/с
    alpha:    np.ndarray   # (N,)      УА, рад
    E_kin:    np.ndarray   # (N,)      кинетическая энергия, Дж
    E_pot:    np.ndarray   # (N,)      потенциальная энергия, Дж
    E_total:  np.ndarray   # (N,)      полная механическая энергия, Дж
    wind_vec: np.ndarray   # (N, 2)    [Vwx, Vwh] м/с


def run(controls_fn,
        aircraft:   AircraftParams,
        wind_params: WindParams,
        cfg:        SimConfig,
        integrator=None,
        state0: np.ndarray = None) -> Log:
    """
    Запустить симуляцию и вернуть лог.

    controls_fn(t, state, Va, alpha) -> np.ndarray([delta_e, throttle])
        Вызывается на каждом шаге. Управление заморожено до следующего шага.

    integrator: step(state, controls, dt, t, params, wind_fn) -> state
        По умолчанию RK4.
    """
    if integrator is None:
        integrator = step_rk4

    n = int(round(cfg.t_end / cfg.dt))
    log = Log(
        t        = np.empty(n),
        state    = np.empty((n, N_STATES)),
        controls = np.empty((n, 2)),
        Va       = np.empty(n),
        alpha    = np.empty(n),
        E_kin    = np.empty(n),
        E_pot    = np.empty(n),
        E_total  = np.empty(n),
        wind_vec = np.empty((n, 2)),
    )

    state = initial_state(cfg) if state0 is None else state0.copy()
    wind_call = lambda h, t: _wind(h, t, wind_params)
    t = 0.0

    for i in range(n):
        h = state[H]
        w_vec = wind_call(h, t)
        Va, alpha = air_velocity(state, w_vec)
        controls = controls_fn(t, state, Va, alpha)
        Ek, Ep, Et = total_energy(state, aircraft)

        log.t[i]        = t
        log.state[i]    = state.copy()
        log.controls[i] = controls
        log.Va[i]       = Va
        log.alpha[i]    = alpha
        log.E_kin[i]    = Ek
        log.E_pot[i]    = Ep
        log.E_total[i]  = Et
        log.wind_vec[i] = w_vec

        # Остановить если ЛА достиг земли
        if h < 0.0:
            log = _trim_log(log, i)
            break

        state = integrator(state, controls, cfg.dt, t, aircraft, wind_call)
        t += cfg.dt

    return log


def compute_trim(aircraft: AircraftParams, Va: float) -> tuple:
    """
    Балансировочные условия горизонтального полёта на скорости Va.

    Решает систему 2×2:
      CLa·α + CLde·δe = CL_req − CL0   (подъёмная сила = вес)
      Cma·α + Cmde·δe = −Cm0            (нулевой момент тангажа)

    Возвращает: (alpha_trim, delta_e_trim, throttle_trim)
    """
    rho, S, m, g = aircraft.rho, aircraft.S, aircraft.mass, aircraft.g
    CL_req = 2.0 * m * g / (rho * Va**2 * S)

    A = np.array([[aircraft.CLa,  aircraft.CLde],
                  [aircraft.Cma,  aircraft.Cmde]])
    b = np.array([CL_req - aircraft.CL0, -aircraft.Cm0])
    alpha_tr, de_tr = np.linalg.solve(A, b)

    # Тяга = сопротивление
    AR     = aircraft.b**2 / S
    CL_lin = aircraft.CL0 + aircraft.CLa * alpha_tr
    CD_tr  = aircraft.CDp + CL_lin**2 / (np.pi * aircraft.e_oswald * AR)
    D_tr   = 0.5 * rho * Va**2 * S * CD_tr
    k      = aircraft.k_motor
    rhs    = D_tr / (0.5 * rho * aircraft.S_prop * aircraft.C_prop) + Va**2
    thr_tr = np.sqrt(max(rhs, 0.0)) / k

    return alpha_tr, de_tr, thr_tr


def trim_state(aircraft: AircraftParams, cfg) -> np.ndarray:
    """Вектор состояния в точке балансировки (горизонтальный полёт)."""
    alpha_tr, _, _ = compute_trim(aircraft, cfg.Va0)
    s = np.zeros(N_STATES)
    s[U]     = cfg.Va0 * np.cos(alpha_tr)
    s[W]     = cfg.Va0 * np.sin(alpha_tr)
    s[THETA] = alpha_tr
    s[H]     = cfg.h0
    return s


def print_summary(log: Log, params: AircraftParams, label: str = ""):
    """Распечатать числовые итоги прогона."""
    from state import THETA, H as H_idx
    t, Va, alpha, h = log.t, log.Va, log.alpha, log.state[:, H_idx]

    alpha_warn = params.alpha_warning
    alpha_crit = params.alpha_crit

    warn_mask = alpha > alpha_warn
    n_warn = int(np.sum(warn_mask))
    t_warn = n_warn * (t[1] - t[0]) if len(t) > 1 else 0.0

    title = f"=== {label} ===" if label else "=== Summary ==="
    print(title)
    print(f"  Duration       : {t[-1]:.2f} s")
    print(f"  Va  final/min  : {Va[-1]:.1f} / {Va.min():.1f} m/s")
    print(f"  h   final      : {h[-1]:.1f} m  (start {h[0]:.1f} m)")
    print(f"  alpha max      : {np.degrees(alpha.max()):.1f} deg")
    print(f"  alpha_warning  : {np.degrees(alpha_warn):.1f} deg")
    print(f"  alpha_crit     : {np.degrees(alpha_crit):.1f} deg")
    print(f"  Exceedances    : {n_warn} steps  ({t_warn:.2f} s above warning)")
    dE = log.E_total[-1] - log.E_total[0]
    print(f"  Delta E_total  : {dE:+.1f} J  ({100*dE/log.E_total[0]:+.2f}%)")
    print()


# ---------------------------------------------------------------------------
# внутренние утилиты
# ---------------------------------------------------------------------------

def _trim_log(log: Log, last_i: int) -> Log:
    """Обрезать лог до индекса last_i (если ЛА упал раньше конца)."""
    return Log(
        t        = log.t[:last_i],
        state    = log.state[:last_i],
        controls = log.controls[:last_i],
        Va       = log.Va[:last_i],
        alpha    = log.alpha[:last_i],
        E_kin    = log.E_kin[:last_i],
        E_pot    = log.E_pot[:last_i],
        E_total  = log.E_total[:last_i],
        wind_vec = log.wind_vec[:last_i],
    )
