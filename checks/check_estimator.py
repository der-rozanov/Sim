# -*- coding: utf-8 -*-
"""
Проверка оценки состояния — инерциального EKF (control/estimator.py, docs/control.md, 9.3).

Полёт: FPV, Va = 16 м/с, ветер 5 м/с на восток (старт в балансировке относительно воздуха),
слабая турбулентность; разворот χ 0 → 90° на 20 с, набор h 100 → 130 м на 80 с.
Датчики — RealisticSensors. Положения рулей для акселерометра — копия привода runner.run.

  1. Якобианы: матрица ошибок A и модели измерений H против конечных разностей.
  2. Разомкнутый контур (САУ летит по истине, EKF наблюдает), N прогонов с разными seed:
     точность против сырых измерений и согласованность — средний NEES по группам
     (≈ размерности группы, если ковариация P честная). Положение — относительно
     «истина + уход GPS»: сам уход без другого источника положения не наблюдаем.
  3. Устойчивость: петля и бочка (кватернион, без вырождения), выброс GPS 50 м на 10 с,
     умеренная турбулентность, скачок бокового ветра при t = 0.
  4. Замкнутый контур: САУ на истине / на сырых измерениях / на EKF.
  5. Выходной предиктор (как в PX4) против повторного прогноза: одни и те же измерения,
     расхождение выходов, переходный процесс выхода на скачок оценки фильтра, время шага.

График — checks/estimator.png. Запуск: python checks/check_estimator.py [--quick]
(--quick — 2 прогона вместо 6 в разделе 2).
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import (H, X, Y, V, Q, air_data, earth_velocity,
                       rotation_body_to_earth, full_controls, euler, rotation)
from sim.dynamics import specific_force
from sim.actuators import actuator_init, actuator_step
from sim.wind import make_wind_fn
from runner import run, compute_trim, trim_state
from control.controllers import PitchControlParams, wrap_angle
from control.sau import SAU, SAUParams, Mode, TruthSensors, RealisticSensors
from control.estimator import (StateEstimator, EstimatorParams, rot_err, quat_to_R, quat_exp, quat_mult, euler_to_quat,
                               meas_mag, meas_airspeed, meas_sideslip, N_ERR)

ok_all = True
deg = np.degrees
QUICK = "--quick" in sys.argv


def check(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}  {info}")


ac, sp = AircraftParams(), SensorParams()
trim = compute_trim(ac, 16.0)


# ---------------------------------------------------------------------------
# Полёт: kind = "observe" (САУ на истине, EKF наблюдает) | "truth" | "raw" | "ekf"
# ---------------------------------------------------------------------------
def flight(kind, seed=42, turb="light", T=180.0, h0=100.0, maneuver=True, override=None,
           glitch=None, wind_start=True, shadow=None):
    cfg = SimConfig(Va0=16.0, h0=h0, dt=0.01, t_end=T)
    wp = WindParams(Vw_cross=5.0, turb_level=turb, turb_seed=seed)
    wind_fn = make_wind_fn(wp)
    rng = np.random.default_rng(seed)
    act = {"surf": None}
    sens = RealisticSensors(sp, rng, accel_fn=lambda t, s: specific_force(s, act["surf"], t, ac, wind_fn))
    ekf = StateEstimator(sp)
    sau_sens = TruthSensors() if kind in ("observe", "truth") else sens
    sau = SAU(ac, trim, sau_sens, SAUParams(pitch=PitchControlParams(Va_ref=16.0)),
              estimator=ekf if kind == "ekf" else None)
    sau.engage(theta=trim[0], h=h0, Va=16.0, chi=0.0)
    sau.set_mode(Mode.ALTITUDE)
    rec = {"est": [], "meas": [], "nees": [], "hist": [], "shadow": []}

    def fn(t, s, Va, alpha):
        if maneuver:
            sau.refs.chi = 0.0 if t < 20 else np.radians(90.0)
            sau.refs.h = h0 if t < 80 else h0 + 30.0
        beta = air_data(s, wind_fn(s[H], t))[2]
        if kind == "observe":
            c = sau.step(t, s, Va, alpha, beta, cfg.dt)
            if override is not None:
                c = override(t, s, c)
            act["surf"] = (actuator_init(full_controls(c), ac) if act["surf"] is None
                           else actuator_step(act["surf"], full_controls(c), cfg.dt, ac))
            m = sens.measure(t, s, Va, alpha, beta)
            if glitch is not None and glitch[0] <= m["t_gps"] < glitch[1]:
                m = dict(m)
                m["pn"] += glitch[2]
            e = ekf.update(m, cfg.dt)
            if shadow is not None:                    # второй оцениватель на тех же измерениях
                rec["shadow"].append(dict(shadow.update(m, cfg.dt)))
            _nees_sample(rec, t, s, ekf, sens, wind_fn)
        else:
            if act["surf"] is None:
                act["surf"] = actuator_init(full_controls(sau.controls), ac)
            c = sau.step(t, s, Va, alpha, beta, cfg.dt)
            act["surf"] = actuator_step(act["surf"], full_controls(c), cfg.dt, ac)
            e = sau.est
        rec["est"].append(dict(e))
        return c

    s0 = trim_state(ac, cfg)
    if wind_start:
        s0[V] += wp.Vw_cross
    log = run(fn, ac, wp, cfg, state0=s0)
    return log, rec, ekf, sens


def _nees_sample(rec, t, s, ekf, sens, wind_fn):
    """Истина на момент t; NEES — для задержанного горизонта фильтра (t_f)."""
    Vx, Vy, Vh = earth_velocity(s)
    w = wind_fn(s[H], t)
    rec["hist"].append((t, rotation(s), np.array([Vx, Vy, -Vh]),
                        np.array([s[X] + sens.gps_drift.b[0], s[Y] + sens.gps_drift.b[1], -s[H]]),
                        sens.gyro_bias().copy(), sens.accel_b0.copy(), np.array([w[0], w[2]])))
    k = int(round((ekf.t_f - rec["hist"][0][0]) / 0.01))
    if 0 <= k < len(rec["hist"]) and t > 10.0:
        _, Rt, vt, pt, bgt, bat, wt = rec["hist"][k]
        dx = np.r_[rot_err(Rt, quat_to_R(ekf.q)), vt - ekf.v, pt - ekf.p, bgt - ekf.bg, bat - ekf.ba, wt - ekf.w]
        rec["nees"].append((dx, np.diag(ekf.P).copy(), ekf.P.copy()))


def att_err(log, E):
    """Угол рассогласования истинной и оценённой ориентации, °."""
    return np.array([deg(np.linalg.norm(rot_err(rotation(s),
                                                quat_to_R(e["quat"])))) for s, e in zip(log.state, E)])


# ---------------------------------------------------------------------------
print("1. Якобианы против конечных разностей")
rng = np.random.default_rng(0)
I, D = np.radians(sp.mag_incl), np.radians(sp.mag_decl)
m_ned = sp.mag_field * np.array([np.cos(I) * np.cos(D), np.cos(I) * np.sin(D), np.sin(I)])


def inject(q, v, w, dx):
    return quat_mult(quat_exp(dx[0:3]), q), v + dx[3:6], w + dx[15:17]


worst_h = 0.0
for _ in range(100):                                      # физичные состояния: воздух — вперёд
    q = euler_to_quat(*rng.uniform([-1, -1, -3], [1, 1, 3]))
    w = rng.normal(0, 4, 2)
    v = quat_to_R(q) @ np.r_[rng.uniform(8, 30), rng.normal(0, 2, 2)] + np.r_[w, 0.0]
    for fn in (lambda q, v, w: meas_mag(q, m_ned), meas_airspeed, meas_sideslip):
        h0, H0 = fn(q, v, w)
        Hn = np.zeros_like(H0)
        for i in list(range(0, 6)) + [15, 16]:
            dx = np.zeros(N_ERR)
            dx[i] = 1e-6
            hp = fn(*inject(q, v, w, dx))[0]
            hm = fn(*inject(q, v, w, -dx))[0]
            Hn[:, i] = (hp - hm) / 2e-6
        worst_h = max(worst_h, np.abs(Hn - H0).max())
check("модели измерений (магнитометр, Va, β): |H − H_числ| < 1e-5", worst_h < 1e-5, f"{worst_h:.1e}")

est = StateEstimator(sp)
worst_a = 0.0
for _ in range(50):
    est.q = euler_to_quat(*rng.uniform([-1, -1, -3], [1, 1, 3]))
    est.v, est.p = rng.normal(0, 10, 3), rng.normal(0, 100, 3)
    est.bg, est.ba = rng.normal(0, 0.01, 3), rng.normal(0, 0.1, 3)
    om, f, dt = rng.normal(0, 1, 3), rng.normal([0, 0, -9.8], 3), 1e-4
    A = est.error_dynamics(f)
    base = est._propagate(est.q, est.v, est.p, om, f, dt)
    for i in range(15):
        dx = np.zeros(N_ERR)
        dx[i] = 1e-5
        e2 = StateEstimator(sp)
        e2.bg, e2.ba = est.bg + dx[9:12], est.ba + dx[12:15]
        q1, v1, p1 = e2._propagate(quat_mult(quat_exp(dx[0:3]), est.q), est.v + dx[3:6],
                                   est.p + dx[6:9], om, f, dt)
        d_out = np.r_[rot_err(quat_to_R(q1), quat_to_R(base[0])), v1 - base[1], p1 - base[2],
                      dx[9:15]]
        num = (d_out - dx[:15]) / dt                       # ≈ A·δx
        worst_a = max(worst_a, np.abs(num - (A @ dx)[:15]).max() / 1e-5)
check("уравнение ошибок: A против разности прогнозов возмущённого и номинального",
      worst_a < 1e-2, f"макс. относительное расхождение {worst_a:.1e}")


# ---------------------------------------------------------------------------
N_MC = 2 if QUICK else 6
print(f"\n2. Разомкнутый контур: {N_MC} прогонов × 120 с (САУ на истине, EKF наблюдает)")
groups = {"ориентация": (slice(0, 3), 3), "скорость": (slice(3, 6), 3),
          "смещ. гироскопа": (slice(9, 12), 3),
          "смещ. акселер.": (slice(12, 15), 3), "ветер": (slice(15, 17), 2)}
nees = {k: [] for k in groups}
acc = {k: [] for k in ("φ, °", "θ, °", "ψ, °", "h, м", "ḣ, м/с", "Va, м/с", "χ, °", "Vg, м/с")}
raw = {k: [] for k in ("h, м", "ḣ, м/с", "Va, м/с", "χ, °", "Vg, м/с")}
rej = {}
for seed in range(1, N_MC + 1):
    log, rec, ekf, sens = flight("observe", seed=seed, T=120.0)
    for name, (sl, _) in groups.items():
        nees[name] += [dx[sl] @ np.linalg.solve(P[sl, sl], dx[sl]) for dx, _, P in rec["nees"]]
    E, st, m = rec["est"], log.state, log.t > 10.0
    Vv = np.array([earth_velocity(s) for s in st])
    chi_t, Vg_t = np.arctan2(Vv[:, 1], Vv[:, 0]), np.hypot(Vv[:, 0], Vv[:, 1])
    g = lambda k: np.array([e[k] for e in E])
    r = lambda a: float(np.sqrt(np.mean(a[m] ** 2)))
    acc["φ, °"].append(deg(r(g("phi") - euler(st)[0])))
    acc["θ, °"].append(deg(r(g("theta") - euler(st)[1])))
    acc["ψ, °"].append(deg(r(wrap_angle(g("psi") - euler(st)[2]))))
    acc["h, м"].append(r(g("h") - st[:, H]))
    acc["ḣ, м/с"].append(r(g("hdot") - Vv[:, 2]))
    acc["Va, м/с"].append(r(g("Va") - log.Va))
    acc["χ, °"].append(deg(r(wrap_angle(g("chi") - chi_t))))
    acc["Vg, м/с"].append(r(g("Vg") - Vg_t))
    # сырые измерения тех же датчиков: переигрываем датчики на том же полёте
    rs = RealisticSensors(sp, np.random.default_rng(seed), accel_fn=lambda t, s: np.zeros(3))
    M = [rs.measure(t, s, Va, a, b) for t, s, Va, a, b in zip(log.t, st, log.Va, log.alpha, log.beta)]
    gm = lambda k: np.array([x[k] for x in M])
    raw["h, м"].append(r(gm("h") - st[:, H]))
    raw["ḣ, м/с"].append(r(gm("hdot") - Vv[:, 2]))
    raw["Va, м/с"].append(r(gm("Va") - log.Va))
    raw["χ, °"].append(deg(r(wrap_angle(gm("chi") - chi_t))))
    raw["Vg, м/с"].append(r(gm("Vg") - Vg_t))
    for k, (u, j) in ekf.stats.items():
        rej.setdefault(k, [0, 0])
        rej[k][0] += u
        rej[k][1] += j

print("   СКО ошибки (среднее по прогонам)   " + "".join(f"{k:>9}" for k in acc))
print("   EKF                                " + "".join(f"{np.mean(v):9.3f}" for v in acc.values()))
print("   сырые измерения                    " + "".join(f"{np.mean(raw[k]):9.3f}" if k in raw else f"{'—':>9}"
                                                       for k in acc))
for k in ("h, м", "ḣ, м/с", "χ, °", "Vg, м/с"):
    check(f"EKF точнее сырых измерений: {k}", np.mean(acc[k]) < np.mean(raw[k]),
          f"{np.mean(acc[k]):.3f} против {np.mean(raw[k]):.3f}")
check("ошибка углов EKF φ, θ < 0.3°, ψ < 1°",
      max(np.mean(acc["φ, °"]), np.mean(acc["θ, °"])) < 0.3 and np.mean(acc["ψ, °"]) < 1.0,
      f"φ {np.mean(acc['φ, °']):.3f}°, θ {np.mean(acc['θ, °']):.3f}°, ψ {np.mean(acc['ψ, °']):.3f}°")
print("   Согласованность: средний NEES / размерность (норма ≈ 1; > 2 — фильтр самоуверен)")
for name, (sl, dim) in groups.items():
    val = np.mean(nees[name]) / dim
    print(f"     {name:>16}: {val:5.2f}")
    check(f"NEES {name} в пределах 0.3…2", 0.3 < val < 2.0)
# Положение: уход GPS (марковский процесс, τ = 1100 с) фильтр считает белым шумом и
# подтягивается к нему с запаздыванием — согласованность по положению проверяется
# на прогоне без ухода, а влияние ухода показывается отдельно.
sp0 = SensorParams(gps_pos_drift_sigma=0.0)
sp_saved, sp = sp, sp0
_, rec0, _, _ = flight("observe", seed=1, T=60.0)
sp = sp_saved
val = np.mean([d[6:9] @ np.linalg.solve(P[6:9, 6:9], d[6:9]) for d, _, P in rec0["nees"]]) / 3
print(f"           положение:  {val:5.2f}   (прогон без ухода GPS)")
check("NEES положение (без ухода GPS) в пределах 0.3…2", 0.3 < val < 2.0)
e_drift = np.mean([np.linalg.norm(d[6:8]) for d, _, _ in rec["nees"]])
print(f"   с уходом GPS: ошибка положения относительно «истина + уход» {e_drift:.2f} м в среднем —"
      " фильтр сглаживает уход (уход сам по себе не наблюдаем: 5–8 м)")
print("   Отбраковано, %: " + ", ".join(f"{k} {100 * j / max(u + j, 1):.2f}" for k, (u, j) in rej.items()))
check("доля отбракованных измерений < 2 % по каждой группе",
      all(j / max(u + j, 1) < 0.02 for u, j in rej.values()))


# ---------------------------------------------------------------------------
print("\n3. Устойчивость")


def aerobatics(t, s, c):
    if 10.0 <= t < 14.5:                                  # петля: РВ на себя, полный газ
        return np.array([np.radians(-18.0), 1.0, 0.0, 0.0])
    if 25.0 <= t < 26.8:                                  # бочка: элероны 20°
        return np.array([c[0], 1.0, np.radians(20.0), 0.0])
    return c


log, rec, ekf, _ = flight("observe", seed=1, T=45.0, h0=300.0, maneuver=False, override=aerobatics)
ae = att_err(log, rec["est"])
loop = deg(np.sum(log.state[(log.t >= 10) & (log.t < 16), Q]) * 0.01)
finite = all(np.isfinite(e["quat"]).all() for e in rec["est"])
check("петля и бочка: оценка конечна, ошибка ориентации < 3°",
      finite and ae[log.t > 10].max() < 3.0,
      f"∫q = {loop:.0f}°, θ до {deg(log.theta).max():.0f}°, макс. ошибка {ae[log.t > 10].max():.2f}°")
log_g, rec_g, ekf_g, _ = flight("observe", seed=2, T=60.0, glitch=(20.0, 30.0, 50.0))
e_pn = np.array([e["pn"] for e in rec_g["est"]]) - log_g.state[:, X]
check("выброс GPS +50 м на 10 с отбракован: |ошибка pn| < 6 м, сбросов нет",
      np.abs(e_pn[log_g.t > 10]).max() < 6.0 and ekf_g.n_gps_reset == 0,
      f"макс. {np.abs(e_pn[log_g.t > 10]).max():.1f} м, отбраковано {ekf_g.stats['gps_pos'][1]} отсчётов")
for name, kw in (("умеренная турбулентность", dict(turb="moderate")),
                 ("скачок бокового ветра при t = 0", dict(wind_start=False))):
    lg, rc, ek, _ = flight("observe", seed=3, T=60.0, **kw)
    a = att_err(lg, rc["est"])
    check(f"{name}: СКО ошибки ориентации после 10 с < 1°",
          np.sqrt(np.mean(a[lg.t > 10] ** 2)) < 1.0,
          f"СКО {np.sqrt(np.mean(a[lg.t > 10] ** 2)):.2f}°, макс. {a.max():.1f}° (при t = {lg.t[np.argmax(a)]:.1f} с)")


# ---------------------------------------------------------------------------
print("\n4. Замкнутый контур (рули — СКО приращения за шаг 0.01 с)")
res = {k: flight(k) for k in ("truth", "raw", "ekf")}


def loop_metrics(log):
    Vv = np.array([earth_velocity(s)[:2] for s in log.state])
    chi = np.arctan2(Vv[:, 1], Vv[:, 0])
    st, t = log.state, log.t
    return {"полёт": t[-1] > 179.9,
            "φ перебег, °": -deg(euler(st[(t > 24) & (t < 32)])[0].min()),
            "|e_χ| 60…80 с, °": deg(np.abs(wrap_angle(chi[(t > 60) & (t < 80)] - np.radians(90))).max()),
            "|Δh| 140…180 с, м": float(np.abs(st[t > 140, H] - 130.0).max()),
            "δe, °": deg(np.std(np.diff(log.controls[:, 0]))),
            "δa, °": deg(np.std(np.diff(log.controls[:, 2]))),
            "δt": float(np.std(np.diff(log.controls[:, 1])))}


L = {k: loop_metrics(v[0]) for k, v in res.items()}
print(f"   {'':>7}" + "".join(f"{k:>18}" for k in L["truth"]))
for k, mtr in L.items():
    print(f"   {k:>7}" + "".join((f"{'да' if v else 'НЕТ':>18}" if isinstance(v, (bool, np.bool_))
                                  else f"{v:18.3f}") for v in mtr.values()))
for k in res:
    check(f"{k}: полёт 180 с", L[k]["полёт"])
check("EKF: рули дёргаются меньше, чем на сырых измерениях (δe, δa, δt)",
      all(L["ekf"][k] < L["raw"][k] for k in ("δe, °", "δa, °", "δt")))
check("EKF: перебег крена после разворота меньше, чем на сырых измерениях",
      L["ekf"]["φ перебег, °"] < L["raw"]["φ перебег, °"],
      f"{L['ekf']['φ перебег, °']:.1f}° против {L['raw']['φ перебег, °']:.1f}°")

# ---------------------------------------------------------------------------
print("\n5. Выходной предиктор против повторного прогноза (одни и те же измерения)")
print("   СКО ошибки против истины, t > 10 с     ориент., °   ψ, °   vN, м/с   h, м   Va, м/с   χ, °")
worst = {}
tm = {"predictor": [], "replay": []}
for seed in (1, 2):
    sh = StateEstimator(sp, EstimatorParams(output="replay"))
    lg, rc, ek, _ = flight("observe", seed=seed, T=120.0, shadow=sh)
    st, mm = lg.state, lg.t > 10.0
    Vv = np.array([earth_velocity(s) for s in st])
    chi_t = np.arctan2(Vv[:, 1], Vv[:, 0])
    tm["predictor"].append(ek.t_ms)
    tm["replay"].append(sh.t_ms)
    for name, E in (("predictor", rc["est"]), ("replay", rc["shadow"])):
        g = lambda k: np.array([e[k] for e in E])
        r = lambda a: float(np.sqrt(np.mean(a[mm] ** 2)))
        print(f"   seed {seed} {name:>10}                  " + "".join(f"{v:9.3f}" for v in (
            r(att_err(lg, E)), deg(r(wrap_angle(g("psi") - euler(st)[2]))), r(g("vn") - Vv[:, 0]),
            r(g("h") - st[:, H]), r(g("Va") - lg.Va), deg(r(wrap_angle(g("chi") - chi_t))))))
    gp = lambda k: np.array([e[k] for e in rc["est"]])
    gr = lambda k: np.array([e[k] for e in rc["shadow"]])
    dq = np.array([deg(np.linalg.norm(rot_err(quat_to_R(a["quat"]), quat_to_R(b["quat"]))))
                   for a, b in zip(rc["est"], rc["shadow"])])
    for k, v in (("ориентация, °", dq), ("v, м/с", np.abs(np.c_[gp("vn") - gr("vn"), gp("ve") - gr("ve"),
                                                                  gp("vd") - gr("vd")]).max(1)),
                 ("h, м", np.abs(gp("h") - gr("h"))), ("pN, pE, м", np.maximum(np.abs(gp("pn") - gr("pn")),
                                                                              np.abs(gp("pe") - gr("pe"))))):
        worst[k] = max(worst.get(k, 0.0), float(v[mm].max()))
print("   наибольшее расхождение выходов (t > 10 с): " + ", ".join(f"{k} {v:.3f}" for k, v in worst.items()))
check("расхождение предиктора и повторного прогноза: ориентация < 1°, v < 0.2 м/с, h и p < 0.3 м",
      worst["ориентация, °"] < 1.0 and worst["v, м/с"] < 0.2 and worst["h, м"] < 0.3 and worst["pN, pE, м"] < 0.3)


def output_step(kick, read):
    """Скачок оценки фильтра при t = 1 с; инерциальное счисление без измерений и шума."""
    e = StateEstimator(sp)
    m = {"t": 0.0, "ax": 0.0, "ay": 0.0, "az": -9.81, "mx": 0.2, "my": 0.0, "mz": 0.4,
         "vn": 16.0, "ve": 0.0, "vd": 0.0, "pn": 0.0, "pe": 0.0, "h": 100.0, "Va": 16.0,
         "p": 0.0, "q": 0.0, "r": 0.0}
    t, y = [], []
    for k in range(600):
        m["t"] = k * 0.01
        if k == 100:
            kick(e)
        y.append(read(e.update(m, 0.01), e))
        t.append(m["t"])
    t, y = np.array(t) - 1.0, np.array(y)
    y = y[t >= 0.0] / y[-1]
    t = t[t >= 0.0]
    return t[np.argmax(y >= 0.632)], t[np.argmax(y >= 0.95)], 100.0 * max(y.max() - 1.0, 0.0)


q_err = lambda est, e: deg(np.linalg.norm(rot_err(quat_to_R(e.q), quat_to_R(est["quat"]))))
steps = {
    "pN +5 м": output_step(lambda e: e.p.__setitem__(0, e.p[0] + 5.0),
                           lambda est, e: est["pn"] - 16.0 * est["t"]),
    "vE +1 м/с": output_step(lambda e: e.v.__setitem__(1, e.v[1] + 1.0), lambda est, e: est["ve"]),
    "ориентация 2° (ось N)": output_step(lambda e: setattr(e, "q", quat_mult(quat_exp(np.radians([2.0, 0, 0])), e.q)),
                                         lambda est, e: 2.0 - q_err(est, e)),
    "ориентация 2° (ось D)": output_step(lambda e: setattr(e, "q", quat_mult(quat_exp(np.radians([0, 0, 2.0])), e.q)),
                                         lambda est, e: 2.0 - q_err(est, e)),
}
print("   Выход на скачок оценки фильтра: 63 %, 95 % (с), перерегулирование (%)")
for k, (t63, t95, ov) in steps.items():
    print(f"     {k:>22}: {t63:5.2f} {t95:5.2f} {ov:5.1f}")
check("выход отрабатывает скачок фильтра без колебаний: 95 % за < 1 с, перерегулирование < 10 %",
      all(t95 < 1.0 and ov < 10.0 for _, t95, ov in steps.values()))
print(f"   время шага update, мс (скользящее среднее): предиктор {np.mean(tm['predictor']):.3f},"
      f" повторный прогноз {np.mean(tm['replay']):.3f}")

# ---------------------------------------------------------------------------
log_e, rec_e = res["ekf"][0], res["ekf"][1]
log_r = res["raw"][0]
t = log_e.t
E = rec_e["est"]
fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
C_T, C_R, C_E = "#222222", "#d9822b", "#3b75af"
m = (t > 18) & (t < 30)
axs[0, 0].plot(t[m], deg(log_e.phi[m]), color=C_T, lw=1.5, label="истина")
axs[0, 0].plot(t[m], deg(np.array([e["phi"] for e in E])[m]), color=C_E, lw=1.2, ls="--", label="EKF")
axs[0, 0].set_title("Крен в развороте (САУ на EKF)", fontsize=10)
axs[0, 0].set_ylabel("φ, °")
m = (t > 79) & (t < 84)
axs[0, 1].plot(t[m], log_e.state[m, H], color=C_T, lw=1.5, label="истина")
axs[0, 1].plot(t[m], [E[i]["h"] for i in np.where(m)[0]], color=C_E, lw=1.3, label="EKF")
rs = RealisticSensors(sp, np.random.default_rng(42), accel_fn=lambda t, s: np.zeros(3))
M = [rs.measure(tt, s, Va, a, b) for tt, s, Va, a, b in zip(t, log_e.state, log_e.Va, log_e.alpha, log_e.beta)]
axs[0, 1].step(t[m], [M[i]["h"] for i in np.where(m)[0]], where="post", color=C_R, lw=0.9,
               label="барометр 20 Гц")
axs[0, 1].set_title("Высота в начале набора", fontsize=10)
axs[0, 1].set_ylabel("h, м")
m = (t > 19.5) & (t < 24)
Vv = np.array([earth_velocity(s)[:2] for s in log_e.state[m]])
axs[1, 0].plot(t[m], deg(np.unwrap(np.arctan2(Vv[:, 1], Vv[:, 0]))), color=C_T, lw=1.5, label="истина")
axs[1, 0].step(t[m], deg(np.unwrap([M[i]["chi"] for i in np.where(m)[0]])), where="post", color=C_R,
               lw=0.9, label="GPS 5 Гц, запаздывание 0.2 с")
axs[1, 0].plot(t[m], deg(np.unwrap([E[i]["chi"] for i in np.where(m)[0]])), color=C_E, lw=1.2, ls="--",
               label="EKF")
axs[1, 0].set_title("Путевой угол в развороте", fontsize=10)
axs[1, 0].set_ylabel("χ, °")
m = (t > 100) & (t < 110)
axs[1, 1].plot(t[m], deg(log_r.controls[m, 0]), color=C_R, lw=1.0, label="САУ на сырых измерениях")
axs[1, 1].plot(t[m], deg(log_e.controls[m, 0]), color=C_E, lw=1.3, label="САУ на EKF")
axs[1, 1].plot(t[m], deg(res["truth"][0].controls[m, 0]), color=C_T, lw=1.0, label="САУ на истине")
axs[1, 1].set_title("Руль высоты в горизонтальном полёте", fontsize=10)
axs[1, 1].set_ylabel("δe, °")
for ax in axs.flat:
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_xlabel("t, с")
    ax.legend(fontsize=8)
fig.suptitle("Инерциальный EKF: FPV, Va = 16 м/с, ветер 5 м/с, слабая турбулентность, "
             "RealisticSensors (моделирование)", fontsize=11)
fig.tight_layout()
png = os.path.join(os.path.dirname(__file__), "estimator.png")
fig.savefig(png, dpi=110)
print(f"\n   график: {os.path.relpath(png, ROOT)}")

print("\nALL ESTIMATOR CHECKS PASSED" if ok_all else "\nSOME ESTIMATOR CHECKS FAILED")
