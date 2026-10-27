"""
Пилотаж: полёт по пространственной траектории фигуры (побочная разработка,
docs/DEV_PLAN.md, «Пилотажный комплекс C-11»).

Фигура — последовательность отрезков в системе координат входа в фигуру:
  line(L, roll)    — прямая длиной L, м; за отрезок ЛА поворачивается по крену на roll, рад
                     (с постоянной скоростью вращения; roll = 0 — просто прямая)
  arc(R, angle)    — дуга радиуса R, м, в плоскости тангажа на угол angle, рад
                     (angle > 0 — «на себя», к оси −z ЛА; < 0 — «от себя»)
В каждой точке траектории задана опорная система координат R_p = [t, y_p, z_p]
(столбцы в земной СК NED): t — касательная, y_p — правое крыло, z_p — «вниз» по
кабине. Вне вращения по крену она переносится вдоль траектории без поворота вокруг t.

Закон управления (каскад, сверху вниз):

1. Наведение. Ошибка положения в плоскости, перпендикулярной t:
       e_y = (p − p_ref)·y_p,  e_z = (p − p_ref)·z_p;   ė — по земной скорости.
   Требуемое ускорение, м/с²: центростремительное дуги + ПД (собственная частота ω_n,
   демпфирование ζ):
       a_y = −ω_n²·e_y − 2ζω_n·ė_y,
       a_z = −V²·κ − ω_n²·e_z − 2ζω_n·ė_z,         κ = ±1/R на дуге, 0 на прямой.
   Требуемая удельная сила (её создаёт аэродинамика): f = a − g, g = (0, 0, g) в NED.
2. Обращение модели сил (B&M 4.2): по f_z — подъёмная сила, по f_y — боковая:
       α_c = (−m·f_z/(q̄S) − CL0) / CLα,     β_c = m·f_y / (q̄S·CY_β),
   с ограничениями |α_c| ≤ α_warning, |β_c| ≤ β_max.
3. Ориентация. Опорная СК R_v: x_v — направление воздушной скорости, y_v — ось y_p (с
   доворотом μ, см. ниже), спроецированная перпендикулярно x_v. Желаемая матрица связанная →
   земная: R_des = R_v · R_b→w(α_c, β_c) — ЛА повёрнут к потоку на (α_c, β_c) и создаёт
   требуемую силу. (Привязка к касательной t вместо x_v превратила бы команду по α в
   наклон вектора скорости: контур положения стал бы в V·m·ω_n²/(q̄S·CLα) ≈ 0.1 1/с.)
   Ошибка — вектор поворота e = log(Rᵀ·R_des) в связанных осях. Уставки угловых скоростей:
       ω_c = ω_ff + e / τ_att,    ω_ff — угловая скорость опорной СК: V·(dφ/ds, κ, 0).
4. Угловые скорости → рули. Упреждение — балансировка моментов при (α_c, β_c, ω_c):
       Cm0 + Cmα·α + Cmq·c·q/(2V) + Cmδe·δe = 0,
       Croll_β·β + (Croll_p·p + Croll_r·r)·b/(2V) + Croll_δa·δa + Croll_δr·δr = 0,
       Cn_β·β    + (Cn_p·p    + Cn_r·r)·b/(2V)    + Cn_δa·δa    + Cn_δr·δr    = 0;
   обратная связь: ПИ по p (коэффициенты контура крена, РЕШ-22), П + И по q (Kq каскада
   тангажа), П по r (полюс λ_r). Коэффициенты рассчитаны на Va_design = 16 м/с, П/И-части
   масштабируются (Va_design/Va)², как в контуре крена (эффективность рулей ∝ Va²).
5. Газ: δt = δt*(Va_ref) + K_E·sin γ_ref + ПИ(Va_ref − Va), sin γ_ref = −t_z; K_E = g/a_V2 —
   газ, компенсирующий составляющую веса вдоль траектории (как в TECS).

Измерения — навигационная оценка step_nav(nav): истина (step, nav_from_state) или EKF
(control/sau.py, режим AEROBATIC). Траектория — в земной СК, α_c и β_c — относительно потока. Коэффициенты — расчёт control/tuning.py на Va_design.
"""

from dataclasses import dataclass, replace
import numpy as np

from sim.state import P, Q, R, X, Y, H, rotation, earth_velocity, air_data
from sim.dynamics import _gammas
from .controllers import PID, PIDParams, saturation


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------

def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R_body_to_wind(alpha, beta):
    """Матрица перехода связанная → скоростная СК (B&M ур. 2.5)."""
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    return np.array([[cb * ca, sb, cb * sa],
                     [-sb * ca, cb, -sb * sa],
                     [-sa, 0.0, ca]])


def rotvec(Rm):
    """Вектор поворота (ось·угол) матрицы поворота Rm, рад."""
    v = 0.5 * np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    c = np.clip(0.5 * (np.trace(Rm) - 1.0), -1.0, 1.0)
    ang = np.arccos(c)
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return v                                   # малый угол
        # поворот на 180°: ось — собственный вектор Rm с собственным числом 1
        w, V = np.linalg.eigh(0.5 * (Rm + Rm.T))
        return np.pi * V[:, np.argmax(w)]
    return v * ang / s


@dataclass
class Segment:
    kind: str          # "line" | "arc"
    length: float      # м (у дуги — длина по дуге)
    kappa: float = 0.0  # 1/м, кривизна в плоскости тангажа (> 0 — на себя)
    roll: float = 0.0   # рад, поворот по крену за отрезок (только у прямой)
    label: str = ""


def line(L, roll_deg=0.0, label=""):
    """Прямая длиной L, м, с поворотом по крену roll_deg, ° (по умолчанию без вращения)."""
    return Segment("line", float(L), 0.0, np.radians(roll_deg), label)


def arc(R, angle_deg, label=""):
    """Дуга радиуса R, м, в плоскости тангажа на angle_deg, ° (> 0 — на себя)."""
    a = np.radians(angle_deg)
    return Segment("arc", float(R) * abs(a), np.sign(a) / float(R), 0.0, label)


class Figure:
    """
    Траектория фигуры. p0 — точка входа (NED), R0 — опорная СК входа (столбцы t, y, z).
    ref(i, s) — (p, R_p, κ, dφ/ds) на отрезке i при длине дуги s от его начала.
    """

    def __init__(self, name, segments, p0, R0):
        self.name, self.segs = name, list(segments)
        self.p_start, self.R_start = [], []
        p, Rm = np.asarray(p0, float), np.asarray(R0, float)
        for sg in self.segs:
            self.p_start.append(p)
            self.R_start.append(Rm)
            p, Rm, _, _ = self._ref_local(sg, p, Rm, sg.length)
        self.p_end, self.R_end = p, Rm
        self.length = sum(sg.length for sg in self.segs)

    @classmethod
    def from_entry(cls, name, segments, pos, vel):
        """Фигура со входом в точке pos (NED) по путевому углу скорости vel (NED), горизонтально."""
        chi = np.arctan2(vel[1], vel[0])
        c, s = np.cos(chi), np.sin(chi)
        R0 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return cls(name, segments, pos, R0)

    @classmethod
    def from_state(cls, name, segments, state):
        """Фигура со входом в текущей точке вектора состояния (истина)."""
        Vx, Vy, Vh = earth_velocity(state)
        return cls.from_entry(name, segments, [state[X], state[Y], -state[H]], [Vx, Vy, -Vh])

    @staticmethod
    def _ref_local(sg, p0, R0, s):
        t0, z0 = R0[:, 0], R0[:, 2]
        if sg.kind == "line":
            rho = sg.roll / sg.length
            return p0 + s * t0, R0 @ _Rx(rho * s), 0.0, rho
        k = sg.kappa
        th = k * s
        p = p0 + (np.sin(th) / k) * t0 - ((1.0 - np.cos(th)) / k) * z0
        return p, R0 @ _Ry(th), k, 0.0

    def ref(self, i, s):
        if i >= len(self.segs):                      # после фигуры — прямая по выходу
            return self.p_end + s * self.R_end[:, 0], self.R_end, 0.0, 0.0
        return self._ref_local(self.segs[i], self.p_start[i], self.R_start[i], s)

    def ahead(self, i, s, ds):
        """(i, s) точки траектории на ds дальше по длине дуги (с переходом через отрезки)."""
        s += ds
        while i < len(self.segs) and s >= self.segs[i].length:
            s -= self.segs[i].length
            i += 1
        return i, s

    def points(self, ds=5.0):
        """Точки траектории через ds по длине дуги, NED, (K, 3)."""
        out = []
        for i, sg in enumerate(self.segs):
            for s in np.arange(0.0, sg.length, ds):
                out.append(self.ref(i, s)[0])
        out.append(self.p_end)
        return np.array(out)

    def project(self, i, s_prev, pos):
        """Ближайшая точка траектории около (i, s_prev): возвращает (i, s)."""
        while True:
            if i >= len(self.segs):
                return i, float((pos - self.p_end) @ self.R_end[:, 0])
            sg, p0, R0 = self.segs[i], self.p_start[i], self.R_start[i]
            d = pos - p0
            if sg.kind == "line":
                s = float(d @ R0[:, 0])
            else:
                k = sg.kappa
                v = d + R0[:, 2] / k                  # от центра дуги
                th = np.arctan2(k * (v @ R0[:, 0]), k * (v @ R0[:, 2]))
                th_prev = k * s_prev
                th += 2 * np.pi * np.round((th_prev - th) / (2 * np.pi))   # без скачка на ±π
                s = float(th / k)
            if s < sg.length:
                return i, max(s, 0.0)
            i, s_prev = i + 1, 0.0


# ---------------------------------------------------------------------------
# Фигуры и комплекс
# ---------------------------------------------------------------------------
# Каждая фигура начинается и кончается в горизонтальном полёте «на ногах»; направление
# выхода — то же или обратное (checks/check_aerobatics.py проверяет конечную СК).

def fig_loop(R=15.0):
    """Петля."""
    return [arc(R, 360)]


def fig_wingover(R=15.0, climb=45.0):
    """
    Горка с разворотом (замена поворота на вертикали — нет обдувки РН винтом): подъём на
    climb, четверть бочки, разворот на 180° дугой в наклонной плоскости, четверть бочки
    обратно, выход в горизонт. Выход — обратным курсом на той же высоте, сдвиг вбок 2R.
    """
    return [arc(R, climb), line(10, 90), arc(R, 180), line(10, -90), arc(R, climb)]


def fig_square_loop(R=10.0, side=20.0):
    """Квадратная петля: стороны side, углы — дуги R."""
    return [arc(R, 90), line(side)] * 3 + [arc(R, 90)]


def fig_climb_half_roll_loop(R1=10.0, L=40.0, R2=20.0):
    """Подъём под 45° с полубочкой, затем 5/8 петли из положения вверх ногами — выход обратным курсом."""
    return [arc(R1, 45), line(L, 180), arc(R2, 225)]


def fig_roll(L=50.0):
    """Бочка: 360° по крену на прямой длиной L."""
    return [line(L, 360)]


def fig_humpty(R=10.0, L=30.0, R_top=8.0, R_out=12.0):
    """Вертикаль вверх, полупетля, вертикаль вниз с полубочкой, выход обратным курсом."""
    return [arc(R, 90), line(L), arc(R_top, 180), line(L, 180), arc(R_out, 90)]


def fig_cuban_eight(R=15.0):
    """Кубинская восьмёрка: 5/8 петли, полубочка на нисходящей 45°, 3/4 петли, полубочка, 1/8 петли."""
    return [arc(R, 225), line(2 * R, 180), arc(R, 270), line(2 * R, 180), arc(R, 45)]


def fig_immelmann(R=15.0, L=20.0):
    """Иммельман: полупетля и полубочка — выход обратным курсом на 2R выше."""
    return [arc(R, 180), line(L, 180)]


# Комплекс C-11 (Starting Schedule C-11, январь 2010) без штопора (фигура 9 — решение
# автора 2026-10-27); фигура 2 «поворот на вертикали» заменена горкой с разворотом
C11 = [
    ("1. Петля", fig_loop()),
    ("2. Горка с разворотом", fig_wingover()),
    ("3. Квадратная петля", fig_square_loop()),
    ("4. Подъём 45° с полубочкой, 5/8 петли", fig_climb_half_roll_loop()),
    ("5. Бочка", fig_roll()),
    ("6. Вертикаль, полупетля, вертикаль с полубочкой", fig_humpty()),
    ("7. Кубинская восьмёрка", fig_cuban_eight()),
    ("8. Иммельман", fig_immelmann()),
]


def programme(figures, lead=60.0, link=60.0):
    """
    Комплекс одной траекторией: прямая входа lead, затем фигуры, после каждой — прямая link
    (выход из фигуры и разгон перед следующей). Метка отрезка — название фигуры
    (у прямых между фигурами — пустая).
    """
    segs = [line(lead)]
    for name, fsegs in figures:
        segs += [replace(sg, label=name) for sg in fsegs]
        segs.append(line(link))
    return segs


# ---------------------------------------------------------------------------
# Регулятор
# ---------------------------------------------------------------------------

@dataclass
class AeroParams:
    Va_ref: float = 25.0                 # м/с, уставка скорости (скорость входа в фигуру)
    Va_design: float = 16.0              # м/с, точка расчёта коэффициентов (tuning.py, РЕШ-21)
    # контур положения: ω_n = 2.5 рад/с — в 2.7 раза медленнее контура ориентации (1/τ_att);
    # при 3.5 рад/с в углах квадратной петли ошибка ориентации 31° против 19° (перебор
    # 1.5…3.5 рад/с и τ_lead 0…0.2 с, checks/check_aerobatics.py — моделирование)
    w_pos: float = 2.5                   # рад/с, собственная частота контура положения
    zeta_pos: float = 0.9
    tau_att: float = 0.15                # с, постоянная времени контура ориентации
    tau_lead: float = 0.1                # с, упреждение: κ и dφ/ds — в точке на V·τ_lead впереди
    mu_max: float = np.radians(20.0)     # рад, предел доворота подъёмной силы креном
    lam_r: float = 10.0                  # рад/с, полюс контура скорости рыскания
    q_Ki_zero: float = 2.0               # рад/с, нуль ПИ контура q (Ki = Kq·q_Ki_zero)
    beta_max: float = np.radians(15.0)   # рад, предел β_c
    rate_max: tuple = (np.radians(200.0), np.radians(150.0), np.radians(60.0))  # p, q, r
    scale_min: float = 0.5
    scale_max: float = 2.0


class AerobaticController:
    """
    Пилотажный регулятор: траектория фигуры → (δe, δt, δa, δr). Схема — docstring модуля.
    step(t, state, Va, alpha, dt) — интерфейс controls_fn прогона run() (с dt).
    """

    def __init__(self, aircraft, params: AeroParams = None, wind_fn=None):
        from .tuning import design_sau
        from runner import compute_trim
        self.ac, self.pr = aircraft, params or AeroParams()
        self.wind_fn = wind_fn or (lambda h, t: (0.0, 0.0, 0.0))
        d = design_sau(aircraft, self.pr.Va_design)
        ac, Va = aircraft, self.pr.Va_design
        g = d.gains
        self.K_E = g["K_E"]
        self.thr_trim = compute_trim(aircraft, self.pr.Va_ref)[2]
        # контуры угловых скоростей
        self.pid_p = PID(PIDParams(Kp=g["kp_p"], Ki=g["ki_p"], integral_limit=0.06 / g["ki_p"]), "p")
        self.pid_q = PID(PIDParams(Kp=g["Kq"], Ki=g["Kq"] * self.pr.q_Ki_zero,
                                   integral_limit=0.1 / (g["Kq"] * self.pr.q_Ki_zero)), "q")
        # объект по r (B&M 3.17): ṙ = −a_r1·r + a_r2·δr
        G4, G8 = _gammas(ac)[3], _gammas(ac)[7]
        qS_b = 0.5 * ac.rho * Va**2 * ac.S * ac.b
        a_r1 = -qS_b * (G4 * ac.Croll_r + G8 * ac.Cn_r) * ac.b / (2 * Va)   # демпфирование, 1/с
        a_r2 = qS_b * (G4 * ac.Croll_dr + G8 * ac.Cn_dr)                  # < 0: δr > 0 — нос влево
        self.kr = (self.pr.lam_r - a_r1) / a_r2
        self.pid_V = PID(PIDParams(Kp=d.speed.Va_Kp, Ki=d.speed.Va_Ki,
                                   integral_limit=d.speed.Va_integral_limit), "Va")
        self.fig, self.i, self.s = None, 0, 0.0
        self.log = {}                                # последний шаг — для отладки и логов

    def set_figure(self, fig: Figure):
        self.fig, self.i, self.s = fig, 0, 0.0

    def reset(self):
        """Интегралы контуров — с нуля (при включении режима)."""
        for pid in (self.pid_p, self.pid_q, self.pid_V):
            pid.reset()

    @property
    def done(self):
        return self.fig is None or self.i >= len(self.fig.segs)

    @property
    def figure_name(self):
        """Название выполняемой фигуры; "" — на прямых между фигурами."""
        return "" if self.done else self.fig.segs[self.i].label

    # ---------------------------------------------------------------
    def nav_from_state(self, t, state):
        """Навигационная оценка по истинному состоянию (ветер — wind_fn)."""
        wv = self.wind_fn(state[H], t)
        Vx, Vy, Vh = earth_velocity(state)
        vel = np.array([Vx, Vy, -Vh])
        return dict(pos=np.array([state[X], state[Y], -state[H]]), vel=vel,
                    v_air=vel - np.array([wv[0], wv[2] if len(wv) > 2 else 0.0, -wv[1]]),
                    R=rotation(state), omega=np.array([state[P], state[Q], state[R]]),
                    Va=air_data(state, wv)[0])

    def step(self, t, state, Va, alpha, dt):
        """Интерфейс controls_fn прогона run(): измерения — истинные."""
        return self.step_nav(self.nav_from_state(t, state), dt)

    def step_nav(self, nav, dt):
        """
        Шаг по навигационной оценке nav (истина или EKF — control/sau.py):
          pos, vel — положение и земная скорость в NED, м, м/с;  v_air — воздушная скорость
          в NED (vel − ветер);  R — матрица связанная → NED;  omega — (p, q, r), рад/с;  Va, м/с.
        """
        ac, pr = self.ac, self.pr
        pos, vel, Rb, Va_ = nav["pos"], nav["vel"], nav["R"], nav["Va"]

        # 1. Наведение
        self.i, self.s = self.fig.project(self.i, self.s, pos)
        p_ref, Rp, _, _ = self.fig.ref(self.i, self.s)
        tt, yp, zp = Rp[:, 0], Rp[:, 1], Rp[:, 2]
        e = pos - p_ref
        V = max(vel @ tt, 1.0)
        _, _, kappa, rho = self.fig.ref(*self.fig.ahead(self.i, self.s, V * pr.tau_lead))
        wn, zt = pr.w_pos, pr.zeta_pos
        a_y = -wn**2 * (e @ yp) - 2 * zt * wn * (vel @ yp)
        a_z = -V**2 * kappa - wn**2 * (e @ zp) - 2 * zt * wn * (vel @ zp)
        f_y = a_y - ac.g * yp[2]
        f_z = a_z - ac.g * zp[2]
        # доворот подъёмной силы к требуемой силе поворотом вокруг t на μ (остаток — скольжением):
        # μ = atan(f_y / (−f_z)), |μ| ≤ μ_max, с весом f_z²/(f_y² + f_z²) — без скачка ±μ_max,
        # когда f_z проходит через 0 («нож» в бочке: там силу даёт только скольжение)
        f2 = f_y**2 + f_z**2
        mu = 0.0 if f2 < 1e-9 or abs(f_z) < 1e-9 else \
            saturation(np.arctan(-f_y / f_z), -pr.mu_max, pr.mu_max) * f_z**2 / f2
        Rp = Rp @ _Rx(mu)
        f_y, f_z = np.cos(mu) * f_y + np.sin(mu) * f_z, -np.sin(mu) * f_y + np.cos(mu) * f_z

        # 2. Обращение модели сил
        qS = 0.5 * ac.rho * max(Va_, 1.0)**2 * ac.S
        alpha_c = (-ac.mass * f_z / qS - ac.CL0) / ac.CLa
        alpha_c = saturation(alpha_c, -ac.alpha_warning, ac.alpha_warning)
        beta_c = saturation(ac.mass * f_y / (qS * ac.CY_beta), -pr.beta_max, pr.beta_max)

        # 3. Ориентация
        # опорная СК — по вектору воздушной скорости (x_v), крыло — y опорной СК траектории,
        # спроецированное перпендикулярно x_v: тогда α_c, β_c дают заданную силу, а значит
        # и ускорение (при привязке к касательной t команда лишь наклоняла бы вектор скорости)
        v_air = nav["v_air"]
        xv = v_air / max(np.linalg.norm(v_air), 1e-6)
        yv = Rp[:, 1] - (Rp[:, 1] @ xv) * xv
        yv /= max(np.linalg.norm(yv), 1e-6)
        Rv = np.column_stack([xv, yv, np.cross(xv, yv)])
        Rbw = R_body_to_wind(alpha_c, beta_c)
        R_des = Rv @ Rbw
        e_att = rotvec(Rb.T @ R_des)
        w_ff = Rbw.T @ (V * np.array([rho, kappa, 0.0]))
        w_c = w_ff + e_att / pr.tau_att
        w_c = np.clip(w_c, -np.array(pr.rate_max), np.array(pr.rate_max))
        p_c, q_c, r_c = w_c

        # 4. Угловые скорости → рули: упреждение по балансировке моментов + обратная связь
        Vs = max(Va_, 1.0)
        de_ff = -(ac.Cm0 + ac.Cma * alpha_c + ac.Cmq * ac.c * q_c / (2 * Vs)) / ac.Cmde
        rhs = -np.array([ac.Croll_beta * beta_c + (ac.Croll_p * p_c + ac.Croll_r * r_c) * ac.b / (2 * Vs),
                         ac.Cn_beta * beta_c + (ac.Cn_p * p_c + ac.Cn_r * r_c) * ac.b / (2 * Vs)])
        da_ff, dr_ff = np.linalg.solve([[ac.Croll_da, ac.Croll_dr], [ac.Cn_da, ac.Cn_dr]], rhs)
        sc = saturation(pr.Va_design / Vs, pr.scale_min, pr.scale_max) ** 2
        e_p, e_q, e_r = w_c - nav["omega"]
        da_raw = da_ff + sc * self.pid_p.step(e_p, dt)
        de_raw = de_ff - sc * self.pid_q.step(e_q, dt)
        dr_raw = dr_ff + sc * self.kr * e_r
        da = saturation(da_raw, -ac.delta_a_max, ac.delta_a_max)
        de = saturation(de_raw, ac.delta_e_min, ac.delta_e_max)
        dr = saturation(dr_raw, -ac.delta_r_max, ac.delta_r_max)
        if da != da_raw and e_p * da_raw > 0:
            self.pid_p.unwind(e_p, dt)               # руль в упоре — интеграл не копим
        if de != de_raw and -e_q * de_raw > 0:
            self.pid_q.unwind(e_q, dt)

        # 5. Газ
        e_V = pr.Va_ref - Va_
        thr_raw = self.thr_trim + self.K_E * (-tt[2]) + self.pid_V.step(e_V, dt)
        thr = saturation(thr_raw, ac.throttle_min, ac.throttle_max)
        if thr != thr_raw and e_V * (thr_raw - thr) > 0:
            self.pid_V.unwind(e_V, dt)

        self.log = dict(seg=self.i, s=self.s, e_y=e @ yp, e_z=e @ zp, alpha_c=alpha_c, beta_c=beta_c, mu=mu,
                        att_err=np.linalg.norm(e_att), p_c=p_c, q_c=q_c, r_c=r_c)
        return np.array([de, thr, da, dr])
