"""
Расчёт коэффициентов САУ по модели ЛА — последовательное замыкание контуров
(Beard & McLain, гл. 5–6). Для другого ЛА: design_sau(его AircraftParams, Va).

    coeffs = plant_coeffs(aircraft, Va)        — коэффициенты передаточных функций
    d      = design_sau(aircraft, Va, specs)   — коэффициенты регуляторов
    d.pitch, d.speed, d.lateral, d.roll, d.KH — готовые параметры классов
                                                 control/controllers.py
    d.tecs                                     — параметры TECS (control/tecs.py)

Передаточные функции (линеаризация около горизонтального полёта на Va):
    крен      φ̈ = −a_φ1·φ̇ + a_φ2·δa                 (B&M ур. 5.26)
    курс      χ̇ = (g/Vg)·φ                            (координированный разворот)
    скольжение β̇ = −a_β1·β + a_β2·δr                  (B&M 2-е изд., ур. 5.27)
    тангаж    θ̈ = −a_θ1·θ̇ − a_θ2·θ + a_θ3·δe         (B&M ур. 5.29)
    высота    ḣ = Va·θ                                (малый γ, α ≈ const)
    скорость  V̇a = −a_V1·Va + a_V2·δt                  (B&M ур. 5.36)

Законы управления (как в control/controllers.py):
    p_ref = (φc − φ)/τ_φ,  δa = FF·p_ref + kp_p·(p_ref − p) + ki_p·∫   ← крен, схема PX4/ArduPilot
    φc = atan(Vg·K_χ·(χc − χ)/g), |φ̇c| ≤ φ̇_max                         ← курс, П-контур
    δr = −(kp_β·β + ki_β·∫β) + k_mix·δa                                 ← скольжение + микс
    δr = −(kp_β·β + ki_β·∫β)
    δe = δe* + kp_θ·(θc − θ) − kd_θ·q    ← каскад PitchController:
         q_ref = Kθ·(θc − θ),  δe = δe* − Kq·(q_ref − q),
         Kq = −kd_θ,  Kθ = kp_θ / kd_θ   (тождественно ПД-закону B&M)
    θc = α* + kp_h·(hc − h) + ki_h·∫     ← AltitudeHold (B&M 6.4.3), полоса ω_h, ζ_h
    θc = α* + KH·(hc − h)                ← упрощённый П-контур сценариев s5–s10
    δt = δt* + kp_V·(Vc − Va) + ki_V·∫
    TECS (control/tecs.py): газ ← Ė = γ + V̇/g,  тангаж ← Ḃ = wγ − (2−w)V̇/g
"""

from dataclasses import dataclass, field

import numpy as np

from sim.dynamics import thrust, _gammas
from sim.aero import coef_CL
from .controllers import (PitchControlParams, SpeedControlParams,
                          LateralControlParams, RollControlParams, AltitudeHoldParams)
from .tecs import TECSParams


@dataclass
class SAUSpecs:
    """
    Требования к контурам: собственная частота ωn [рад/с] и демпфирование ζ.
    Внешние контуры задаются разносом полос W (во сколько раз медленнее
    внутреннего), B&M рекомендует 5–15.
    """
    tau_phi: float = 0.25     # с, постоянная времени крена (PX4 FW_R_TC = 0.5,
                              # ArduPilot RLL2SRV_TCONST = 0.45; FPV-самолёт резвее)
    p_max: float = 120.0      # °/с, предел заданной скорости крена (p_уст(δa = 20°) ≈ 210)
    lam_p: float = 75.0       # рад/с, полюс контура скорости крена (собственный a_φ1 ≈ 56)
    zeta_chi: float = 1.0     # курс: ζ контура χ с учётом инерции крена τ_φ
    phi_ref_rate: float = 90.0  # °/с, предел скорости уставки крена (PX4 FW_PN_R_SLEW_MAX)
    kp_beta: float = 0.5      # скольжение: ограничено шумом зонда УС (0.6°)
    zeta_beta: float = 0.7
    wn_theta: float = 15.0    # тангаж
    zeta_theta: float = 0.8
    W_h: float = 30.0         # высота: ω_h = ωn_θ / W_h
    zeta_h: float = 1.5       # ПИ даёт нуль: при 0.9 перерегулирование 18 % на малой ступеньке
    theta_max: float = 15.0   # град, предел θ_ref контура высоты
    wn_V: float = 1.0         # скорость газом
    zeta_V: float = 0.8
    # TECS (control/tecs.py)
    tecs_K_h: float = 0.5     # 1/с, контур высоты (как ω_h раздельных контуров)
    tecs_K_V: float = 0.5     # 1/с, контур скорости
    tecs_w_T: float = 2.0     # рад/с, полоса контура полной энергии (газ)
    tecs_w_B: float = 2.0     # рад/с, полоса контура баланса (тангаж)
    tecs_tau_f: float = 0.15  # с, фильтр Va перед дифференцированием (шум ПВД 0.2 м/с)
    tecs_w: float = 1.0       # вес высоты в балансе
    tecs_margin: float = 0.7  # доля располагаемой тяги на набор/разгон и газа на снижение
    tecs_Va_min: float = 1.25 # начало защиты от потери скорости, доля Va_stall
    tecs_Va_band: float = 1.5 # м/с
    tecs_Va_tab: tuple = (10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0)  # таблица балансировок


@dataclass
class SAUDesign:
    """Результат расчёта: коэффициенты объекта, законов и параметры классов САУ."""
    Va: float
    coeffs: dict
    gains: dict
    pitch: PitchControlParams
    speed: SpeedControlParams
    lateral: LateralControlParams
    roll: RollControlParams
    altitude: AltitudeHoldParams
    tecs: TECSParams
    KH: float
    specs: SAUSpecs = field(default_factory=SAUSpecs)


def plant_coeffs(aircraft, Va: float) -> dict:
    """
    Коэффициенты передаточных функций в горизонтальном полёте на скорости Va.
    Аэродинамические — аналитически по B&M гл. 5; производные тяги — численно
    (модель винта не дифференцируется в явном виде).
    """
    from runner import compute_trim          # runner импортирует control → поздний импорт
    ac = aircraft
    alpha_t, de_t, dt_t = compute_trim(ac, Va)
    G1, G2, G3, G4, G5, G6, G7, G8 = _gammas(ac)

    qS  = 0.5 * ac.rho * Va**2 * ac.S
    Cp_p  = G3 * ac.Croll_p  + G4 * ac.Cn_p
    Cp_da = G3 * ac.Croll_da + G4 * ac.Cn_da

    a_phi1 = -qS * ac.b * Cp_p * ac.b / (2 * Va)
    a_phi2 =  qS * ac.b * Cp_da
    a_beta1 = -ac.rho * Va * ac.S * ac.CY_beta / (2 * ac.mass)
    a_beta2 =  ac.rho * Va * ac.S * ac.CY_dr / (2 * ac.mass)
    a_theta1 = -qS * ac.c * ac.Cmq * ac.c / (2 * Va) / ac.Jy
    a_theta2 = -qS * ac.c * ac.Cma / ac.Jy
    a_theta3 =  qS * ac.c * ac.Cmde / ac.Jy

    # Скорость: m·V̇a = T(δt, Va) − D(Va) − m·g·sin γ, α = α* (B&M 5.4)
    eV, et = 0.05, 0.01
    dT_dVa = (thrust(dt_t, Va + eV, ac) - thrust(dt_t, Va - eV, ac)) / (2 * eV)
    dT_ddt = (thrust(dt_t + et, Va, ac) - thrust(dt_t - et, Va, ac)) / (2 * et)
    AR = ac.b**2 / ac.S
    CD_t = (ac.CDp + (ac.CL0 + ac.CLa * alpha_t)**2 / (np.pi * ac.e_oswald * AR)
            + ac.CDde * abs(de_t))
    a_V1 = ac.rho * Va * ac.S * CD_t / ac.mass - dT_dVa / ac.mass
    a_V2 = dT_ddt / ac.mass

    # Располагаемые темпы энергии: Ė = (T − D)/(m·g), D = T(δt*) в горизонтальном полёте
    E_max = (thrust(1.0, Va, ac) - thrust(dt_t, Va, ac)) / (ac.mass * ac.g)
    E_min = (thrust(0.0, Va, ac) - thrust(dt_t, Va, ac)) / (ac.mass * ac.g)
    al = np.radians(np.linspace(0.0, 30.0, 601))
    CL_max = max(coef_CL(a, 0.0, 0.0, Va, ac) for a in al)
    Va_stall = np.sqrt(2 * ac.mass * ac.g / (ac.rho * ac.S * CL_max))

    return dict(Va=Va, alpha_trim=alpha_t, de_trim=de_t, thr_trim=dt_t,
                a_phi1=a_phi1, a_phi2=a_phi2, a_beta1=a_beta1, a_beta2=a_beta2,
                a_theta1=a_theta1, a_theta2=a_theta2, a_theta3=a_theta3,
                a_V1=a_V1, a_V2=a_V2, E_max=E_max, E_min=E_min,
                CL_max=CL_max, Va_stall=Va_stall)


def design_sau(aircraft, Va: float, specs: SAUSpecs = None) -> SAUDesign:
    """Коэффициенты всех контуров САУ для ЛА aircraft в точке Va (B&M гл. 6)."""
    sp = specs or SAUSpecs()
    k = plant_coeffs(aircraft, Va)
    g = aircraft.g

    # --- Крен, каскад угол → скорость крена (PX4/ArduPilot) ---
    # Объект по скорости крена: ṗ = −a_φ1·p + a_φ2·δa. Упреждение FF = a_φ1/a_φ2 даёт
    # p → p_ref в установившемся режиме; с P-частью ṗ = −(a_φ1 + a_φ2·kp_p)(p − p_ref):
    # полюс контура p равен lam_p. Нуль ПИ — на 1/τ_φ (интеграл — против момента винта).
    # Внешний контур p_ref = (φc − φ)/τ_φ: при быстром контуре p  φ/φc ≈ 1/(τ_φ s + 1).
    FF_p = k["a_phi1"] / k["a_phi2"]
    kp_p = max((sp.lam_p - k["a_phi1"]) / k["a_phi2"], 0.0)
    ki_p = kp_p / sp.tau_phi

    # --- Курс: χ̇ = (g/Vg)·tan φ ≈ χ̇_ref при φ = atan(Vg·χ̇_ref/g), с запаздыванием крена:
    # χ/χc = K/(τ_φ s² + s + K),  ζ = 1/(2·sqrt(K·τ_φ))  →  K = 1/(4 ζ² τ_φ)
    K_chi = 1.0 / (4.0 * sp.zeta_chi**2 * sp.tau_phi)

    # --- Микс элеронов в РН (ArduPilot KFF_RDDRMIX): ṙ от δa и δr (B&M 3.17):
    # ṙ ∝ Г4·Croll + Г8·Cn;  δr = K·δa обнуляет ṙ от элеронов → K = −N_δa / N_δr
    G4, G8 = _gammas(aircraft)[3], _gammas(aircraft)[7]
    N_da = G4 * aircraft.Croll_da + G8 * aircraft.Cn_da
    N_dr = G4 * aircraft.Croll_dr + G8 * aircraft.Cn_dr
    k_mix = -N_da / N_dr

    # --- Скольжение (B&M 2-е изд., ур. 6.14) ---
    kp_beta = sp.kp_beta
    ki_beta = ((k["a_beta1"] + k["a_beta2"] * kp_beta) / (2 * sp.zeta_beta))**2 / k["a_beta2"]

    # --- Тангаж: θ/θc = kp·a_θ3 / (s² + (a_θ1 + kd·a_θ3)s + (a_θ2 + kp·a_θ3)) ---
    kp_theta = (sp.wn_theta**2 - k["a_theta2"]) / k["a_theta3"]
    kd_theta = (2 * sp.zeta_theta * sp.wn_theta - k["a_theta1"]) / k["a_theta3"]
    K_theta_DC = kp_theta * k["a_theta3"] / (k["a_theta2"] + kp_theta * k["a_theta3"])
    if not (kp_theta < 0 and kd_theta < 0):
        raise ValueError(f"тангаж: kp_θ={kp_theta:.3f}, kd_θ={kd_theta:.3f} — каскад требует "
                         "оба < 0 (Cmde < 0); уменьшите ζ_θ или увеличьте ωn_θ")
    Kq = -kd_theta
    Ktheta = kp_theta / kd_theta

    # --- Высота: П-контур, h/hc = ω_h/(s + ω_h), ω_h = K·Va·KH ---
    # K_θDC модели B&M (θ̈ = … − a_θ2·θ) занижен: в ней α ≡ θ, а в полном движении
    # α возвращается к балансировочному и θ отрабатывает θc полностью (check_sau.py:
    # 3° → 3°, а не 0.615·3°). Для внешнего контура берём K = 1.
    w_h = sp.wn_theta / sp.W_h
    KH = w_h / Va
    # ПИ (B&M 6.4.3): h/hc = (kp s + ki)Va / (s² + kp Va s + ki Va), K = 1
    kp_h = 2 * sp.zeta_h * w_h / Va
    ki_h = w_h**2 / Va

    # --- Скорость газом: V/Vc = (kp s + ki)a_V2 / (s² + (a_V1 + a_V2 kp)s + a_V2 ki) ---
    kp_V = (2 * sp.zeta_V * sp.wn_V - k["a_V1"]) / k["a_V2"]
    ki_V = sp.wn_V**2 / k["a_V2"]

    # --- TECS (control/tecs.py) ---
    # Газ: Ė = (T − D)/(m·g) → ∂Ė/∂δt = a_V2/g, упреждение K_E = g/a_V2. Измерение Ė
    # запаздывает на фильтр τ_f: объект (a_V2/g)/(τ_f s + 1). Регулятор — И:
    # L = ω_T/(s(τ_f s + 1)), срез ω_T, запас по фазе 90° − atan(ω_T·τ_f) = 73°.
    # П-часть (PX4 FW_T_THR_DAMP) не берём: шум ПВД проходит через дифференцирование
    # V̇ с усилением ω_T·K_E·σ_V/g независимо от τ_f (σ_газ 0.049 против 0.018 без
    # шума ПВД); интеграл V̇ — сама отфильтрованная Va, почти без шума.
    # Тангаж: Ḃ = 2γ − (2 − w)Ė → ∂Ḃ/∂θ ≈ 2 (γ ≈ θ − α*), так же: И с полосой ω_B.
    K_E = g / k["a_V2"]
    tf = sp.tecs_tau_f
    from runner import compute_trim
    trims = [compute_trim(aircraft, v) for v in sp.tecs_Va_tab]
    W, sin_th = aircraft.mass * g, np.sin(np.radians(sp.theta_max))
    E_max_tab = [(thrust(1.0, v, aircraft) - thrust(t[2], v, aircraft)) / W
                 for v, t in zip(sp.tecs_Va_tab, trims)]
    E_min_tab = [(thrust(0.0, v, aircraft) - thrust(t[2], v, aircraft)) / W
                 for v, t in zip(sp.tecs_Va_tab, trims)]
    tecs = TECSParams(
        K_h=sp.tecs_K_h, K_V=sp.tecs_K_V,
        acc_max=sp.tecs_margin * k["E_max"] * g,
        w=sp.tecs_w, tau_f=tf, K_E=K_E,
        thr_Kp=0.0, thr_Ki=sp.tecs_w_T * K_E,
        pit_Kp=0.0, pit_Ki=sp.tecs_w_B / 2,
        theta_max=np.radians(sp.theta_max), theta_min=-np.radians(sp.theta_max),
        Va_min=sp.tecs_Va_min * k["Va_stall"], Va_band=sp.tecs_Va_band,
        Va_tab=tuple(sp.tecs_Va_tab), thr_tab=tuple(float(t[2]) for t in trims),
        alpha_tab=tuple(float(t[0]) for t in trims),
        # Пределы угла траектории на каждой скорости: набор — по тяге на газе 1 и
        # пределу тангажа, снижение — по газу 0 (при постоянной скорости γ = Ė) и пределу
        gamma_climb_tab=tuple(sp.tecs_margin * min(Ex, sin_th) for Ex in E_max_tab),
        gamma_sink_tab=tuple(sp.tecs_margin * min(-En, sin_th) for En in E_min_tab))

    gains = dict(tau_phi=sp.tau_phi, FF_p=FF_p, kp_p=kp_p, ki_p=ki_p, K_chi=K_chi,
                 k_mix=k_mix,
                 kp_beta=kp_beta, ki_beta=ki_beta,
                 kp_theta=kp_theta, kd_theta=kd_theta, K_theta_DC=K_theta_DC,
                 Kq=Kq, Ktheta=Ktheta, w_h=w_h, KH=KH, kp_h=kp_h, ki_h=ki_h,
                 kp_V=kp_V, ki_V=ki_V, K_E=K_E)

    # Предел q_ref каскада не должен отрезать руль: при Kq·q_max < δe_max каскад
    # перестаёт быть ПД-законом B&M (ЛА «застревал» в наборе — полёт 2026-10-07).
    # Берём q_max = Kθ·40° — ограничение не срабатывает при |θc − θ| ≤ 40°;
    # команду ограничивает предел θ_ref (theta_max).
    q_max = Ktheta * np.radians(40.0)
    pitch = PitchControlParams(theta_Kp=Ktheta, theta_Ki=0.0, theta_Kd=0.0,
                               q_Kp=Kq, q_Ki=0.0, q_Kd=0.0, Va_ref=Va, KH=KH,
                               q_max=q_max, q_min=-q_max)
    altitude = AltitudeHoldParams(Kp=kp_h, Ki=ki_h, theta_max=np.radians(sp.theta_max),
                                  integral_limit=np.radians(sp.theta_max) / ki_h)
    # предел интеграла — чтобы интегральная часть могла пройти весь диапазон газа
    speed = SpeedControlParams(Va_Kp=kp_V, Va_Ki=ki_V, Va_Kd=0.0,
                               Va_integral_limit=1.0 / ki_V)
    # предел интеграла: вклад в δa ≤ 0.06 рад ≈ 3.4° (момент винта требует ≲ 2°)
    roll = RollControlParams(tau_phi=sp.tau_phi, p_max=np.radians(sp.p_max), FF=FF_p,
                             Kp=kp_p, Ki=ki_p, integral_limit=0.06 / ki_p, Va_ref=Va)
    lateral = LateralControlParams(roll=roll, chi_K=K_chi,
                                   phi_ref_rate=np.radians(sp.phi_ref_rate),
                                   beta_Kp=kp_beta, beta_Ki=ki_beta, da_dr_mix=k_mix)
    return SAUDesign(Va=Va, coeffs=k, gains=gains, pitch=pitch, speed=speed,
                     lateral=lateral, roll=roll, altitude=altitude, tecs=tecs, KH=KH,
                     specs=sp)


def print_design(d: SAUDesign):
    """Распечатать коэффициенты объекта и регуляторов."""
    k, gn = d.coeffs, d.gains
    print(f"Точка настройки Va = {d.Va} м/с:  α* = {np.degrees(k['alpha_trim']):.2f}°  "
          f"δe* = {np.degrees(k['de_trim']):.2f}°  δt* = {k['thr_trim']:.3f}")
    print("Объект:  " + "  ".join(f"{n}={k[n]:.4g}" for n in
          ("a_phi1", "a_phi2", "a_beta1", "a_beta2", "a_theta1", "a_theta2",
           "a_theta3", "a_V1", "a_V2", "E_max", "E_min", "Va_stall")))
    for n, v in gn.items():
        print(f"  {n:<14} {v:10.4f}")
    print("TECS:", d.tecs)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from sim.config import AircraftParams
    print_design(design_sau(AircraftParams(), 16.0))
