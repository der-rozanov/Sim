# -*- coding: utf-8 -*-
"""
animate.py -- анимация результатов симуляции по готовому логу.

Не зависит от симулятора: принимает Log и рисует анимацию.
Запуск напрямую: python animate.py  (использует встроенный демо-прогон)

API:
    from animate import animate_log
    fig, anim = animate_log(log, params=aircraft, speed=5.0)
    plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon

from state import X, H as H_IDX, THETA, Q


# ---------------------------------------------------------------------------
# Силуэт ЛА (вид сбоку, нос в сторону +x)
#
# Координаты нормированы: тело длиной 2.0, нос в точке x=+1.0.
# Вращение и масштабирование выполняется в пикселях экрана — это гарантирует
# правильную геометрию независимо от соотношения масштабов осей x и h.
# ---------------------------------------------------------------------------

_PLANE_SCALE_PX = 38  # «радиус» тела в пикселях (полная длина тела ≈ 76 px)

# (вершины, facecolor, edgecolor, alpha, zorder)
_PLANE_PARTS = (
    # крыло (вид с ребра — хорда)
    (np.array([[ 0.22,  0.06], [-0.20,  0.06], [-0.27, -0.04], [ 0.15, -0.04]]),
     "steelblue",  "navy", 0.90, 5),
    # горизонтальное оперение
    (np.array([[-0.64,  0.04], [-0.96,  0.04], [-0.96, -0.02], [-0.64, -0.02]]),
     "steelblue",  "navy", 0.90, 5),
    # вертикальный киль
    (np.array([[-0.66,  0.12], [-0.62,  0.46], [-0.94,  0.46], [-0.98,  0.12]]),
     "steelblue",  "navy", 0.90, 5),
    # фюзеляж
    (np.array([[ 1.00,  0.00],
               [ 0.62,  0.14], [ 0.10,  0.16],
               [-0.50,  0.12], [-1.00,  0.02],
               [-0.88, -0.06], [-0.42, -0.12],
               [ 0.10, -0.14], [ 0.62, -0.13]]),
     "royalblue", "darkblue", 1.00, 6),
)

# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def animate_log(log,
                params=None,
                speed: float = 5.0,
                fps: int = 25,
                title: str = "",
                save_path: str = None):
    """
    Анимировать лог симуляции.

    Parameters
    ----------
    log       : runner.Log
    params    : AircraftParams — нужен для линий α_предупреждение/α_критический
    speed     : множитель скорости воспроизведения (5 = в 5 раз быстрее реального)
    fps       : кадров в секунду
    title     : заголовок окна
    save_path : путь для сохранения (.mp4 или .gif), None = только показать

    Returns
    -------
    fig, anim  (держать ссылку на anim чтобы анимация не удалилась сборщиком мусора)
    """
    # --- Прореживание кадров ---
    dt_sim     = log.t[1] - log.t[0] if len(log.t) > 1 else 0.01
    interval_s = 1.0 / fps
    stride     = max(1, int(round(speed * interval_s / dt_sim)))
    idx_list   = list(range(0, len(log.t), stride))
    n_frames   = len(idx_list)

    # --- Данные ---
    t_all     = log.t
    Va_all    = log.Va
    alpha_all = np.degrees(log.alpha)
    h_all     = log.state[:, H_IDX]
    x_all     = log.state[:, X]
    theta_all = log.state[:, THETA]   # угол тангажа, рад
    t_end     = t_all[-1]

    # --- Компоновка ---
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(title if title else "Анимация полёта", fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           width_ratios=[1.3, 1],
                           hspace=0.5, wspace=0.4)

    ax_traj = fig.add_subplot(gs[:, 0])      # траектория — вся левая колонка
    ax_Va   = fig.add_subplot(gs[0, 1])
    ax_al   = fig.add_subplot(gs[1, 1], sharex=ax_Va)
    ax_h    = fig.add_subplot(gs[2, 1], sharex=ax_Va)

    # -----------------------------------------------------------------------
    # Левый subplot: траектория
    # -----------------------------------------------------------------------
    _pad_x = max((x_all.max() - x_all.min()) * 0.06, 10.0)
    _pad_h = max((h_all.max() - h_all.min()) * 0.15, 10.0)

    ax_traj.set_xlim(x_all.min() - _pad_x, x_all.max() + _pad_x)
    ax_traj.set_ylim(h_all.min() - _pad_h, h_all.max() + _pad_h)
    ax_traj.set_xlabel("Горизонтальная дальность x, м", fontsize=9)
    ax_traj.set_ylabel("Высота h, м", fontsize=9)
    ax_traj.set_title("Траектория в вертикальной плоскости", fontsize=9)
    ax_traj.grid(True, linestyle="--", alpha=0.5)

    # Полная траектория бледно (ориентир)
    ax_traj.plot(x_all, h_all, color="lightsteelblue", linewidth=1.2,
                 alpha=0.45, zorder=1, label="полный путь")
    ax_traj.plot(x_all[0],  h_all[0],  "go", markersize=8, zorder=5, label="старт")
    ax_traj.plot(x_all[-1], h_all[-1], "rs", markersize=8, zorder=5, label="финиш")
    ax_traj.legend(fontsize=8, loc="upper left")

    # Динамические объекты — пройденный путь и силуэт ЛА
    traj_line, = ax_traj.plot([], [], "b-", linewidth=2.0, zorder=3)

    # Создаём патчи силуэта; вершины будут обновляться каждый кадр
    plane_patches = []
    for verts_norm, fc, ec, al, zo in _PLANE_PARTS:
        p = MplPolygon(verts_norm, closed=True,
                       facecolor=fc, edgecolor=ec,
                       linewidth=0.8, alpha=al, zorder=zo,
                       transform=ax_traj.transData)
        ax_traj.add_patch(p)
        plane_patches.append(p)

    # Блок с текущими параметрами (прямо на графике траектории)
    info_box = ax_traj.text(
        0.98, 0.04, "",
        transform=ax_traj.transAxes,
        fontsize=8.5, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85),
        family="monospace"
    )

    # -----------------------------------------------------------------------
    # Правые subplots: Va, α, h
    # -----------------------------------------------------------------------
    for ax in (ax_Va, ax_al, ax_h):
        ax.set_xlim(0.0, t_end)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=8)

    ax_Va.set_ylabel("Va, м/с",   fontsize=9)
    ax_al.set_ylabel("УА α, °",   fontsize=9)
    ax_h.set_ylabel("Высота, м",  fontsize=9)
    ax_h.set_xlabel("Время, с",   fontsize=9)
    plt.setp(ax_Va.get_xticklabels(), visible=False)
    plt.setp(ax_al.get_xticklabels(), visible=False)

    # Y-диапазоны
    _v_pad = max((Va_all.max() - Va_all.min()) * 0.15, 0.5)
    ax_Va.set_ylim(Va_all.min() - _v_pad, Va_all.max() + _v_pad)

    _a_lo = alpha_all.min() - 2.0
    _a_hi = alpha_all.max() + 2.0
    if params is not None:
        warn = np.degrees(params.alpha_warning)
        crit = np.degrees(params.alpha_crit)
        _a_hi = max(_a_hi, warn * 0.35)
        ax_al.axhline(warn, color="orange", lw=1.1, ls="--",
                      label=f"пред. {warn:.0f}°", alpha=0.8)
        ax_al.axhline(crit, color="red",    lw=1.1, ls="--",
                      label=f"крит. {crit:.0f}°", alpha=0.8)
        ax_al.legend(fontsize=7, loc="upper right")
    ax_al.set_ylim(min(_a_lo, -1.0), max(_a_hi, 5.0))
    ax_h.set_ylim(h_all.min() - _pad_h, h_all.max() + _pad_h)

    # Бледные опорные линии (весь прогон)
    ax_Va.plot(t_all, Va_all,    color="lightsteelblue", lw=1.0, alpha=0.45)
    ax_al.plot(t_all, alpha_all, color="lightcoral",     lw=1.0, alpha=0.45)
    ax_h.plot( t_all, h_all,     color="lightsteelblue", lw=1.0, alpha=0.45)

    # Живые линии
    ln_Va, = ax_Va.plot([], [], color="steelblue", lw=1.8)
    ln_al, = ax_al.plot([], [], color="crimson",   lw=1.8)
    ln_h,  = ax_h.plot( [], [], color="royalblue", lw=1.8)

    # Точки текущего значения
    pt_Va, = ax_Va.plot([], [], "o", color="steelblue", ms=5, zorder=5)
    pt_al, = ax_al.plot([], [], "o", color="crimson",   ms=5, zorder=5)
    pt_h,  = ax_h.plot( [], [], "o", color="royalblue", ms=5, zorder=5)

    # Вертикальный курсор времени
    vlines = [ax.axvline(0, color="gray", lw=0.8, alpha=0.6, ls=":")
              for ax in (ax_Va, ax_al, ax_h)]

    # -----------------------------------------------------------------------
    # init / update
    # -----------------------------------------------------------------------
    _all_artists = (traj_line, *plane_patches, info_box,
                    ln_Va, ln_al, ln_h,
                    pt_Va, pt_al, pt_h,
                    *vlines)

    def _set_plane(i: int):
        """Обновить вершины силуэта для шага i (ротация в пикселях → данные)."""
        theta = theta_all[i]
        c, s  = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        # позиция ЛА в пикселях экрана
        xy_px = ax_traj.transData.transform((x_all[i], h_all[i]))
        inv   = ax_traj.transData.inverted()
        for patch, (verts_norm, *_) in zip(plane_patches, _PLANE_PARTS):
            rotated = (R @ (verts_norm * _PLANE_SCALE_PX).T).T + xy_px
            patch.set_xy(inv.transform(rotated))

    def init():
        traj_line.set_data([], [])
        info_box.set_text("")
        for ln in (ln_Va, ln_al, ln_h):
            ln.set_data([], [])
        for pt in (pt_Va, pt_al, pt_h):
            pt.set_data([], [])
        for vl in vlines:
            vl.set_xdata([0])
        _set_plane(0)
        return _all_artists

    def update(fn):
        i = idx_list[fn]
        t_cur = t_all[i]

        # Траектория + силуэт
        traj_line.set_data(x_all[:i+1], h_all[:i+1])
        _set_plane(i)

        # Инфо-блок
        info_box.set_text(
            f"t = {t_cur:5.1f} с\n"
            f"Va = {Va_all[i]:5.1f} м/с\n"
            f"α  = {alpha_all[i]:5.2f} °\n"
            f"h  = {h_all[i]:6.1f} м"
        )

        # Временны́е ряды
        ts = t_all[:i+1]
        ln_Va.set_data(ts, Va_all[:i+1])
        ln_al.set_data(ts, alpha_all[:i+1])
        ln_h.set_data( ts, h_all[:i+1])

        pt_Va.set_data([t_cur], [Va_all[i]])
        pt_al.set_data([t_cur], [alpha_all[i]])
        pt_h.set_data( [t_cur], [h_all[i]])

        for vl in vlines:
            vl.set_xdata([t_cur])

        return _all_artists

    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        init_func=init,
        interval=1000.0 / fps,
        blit=True
    )

    if save_path is not None:
        _save(anim, save_path, fps)

    return fig, anim


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

def _save(anim: FuncAnimation, path: str, fps: int):
    if path.endswith(".gif"):
        try:
            from matplotlib.animation import PillowWriter
            anim.save(path, writer=PillowWriter(fps=fps))
            print(f"Сохранено: {path}")
        except Exception as e:
            print(f"Не удалось сохранить GIF (нужен Pillow): {e}")
    else:
        try:
            anim.save(path, fps=fps, extra_args=["-vcodec", "libx264"])
            print(f"Сохранено: {path}")
        except Exception as e:
            print(f"Не удалось сохранить MP4 (нужен ffmpeg): {e}")


# ---------------------------------------------------------------------------
# Запуск напрямую: python animate.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from config import AircraftParams, WindParams, SimConfig
    from runner import run, compute_trim, trim_state
    import numpy as np

    print("Запуск демо-прогона для анимации...")

    aircraft    = AircraftParams()
    wind_params = WindParams()
    cfg         = SimConfig(Va0=30.0, h0=100.0, dt=0.01, t_end=20.0)

    alpha_t, de_t, thr_t = compute_trim(aircraft, cfg.Va0)
    ctrl = np.array([de_t, thr_t])
    log  = run(lambda t, s, V, a: ctrl, aircraft, wind_params, cfg,
               state0=trim_state(aircraft, cfg))

    save = sys.argv[1] if len(sys.argv) > 1 else None

    fig, anim = animate_log(
        log,
        params=aircraft,
        speed=1.0,
        fps=25,
        title="Тримовый полёт — анимация",
        save_path=save,
    )
    plt.show()
