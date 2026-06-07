# -*- coding: utf-8 -*-
"""
Вьюер полётных логов (.flightlog).

Запуск:
    python viewer.py                              # диалог выбора файла
    python viewer.py results/s8.flightlog         # открыть напрямую
    python viewer.py results/s8.flightlog --speed 3.0   # скорость анимации
    python viewer.py results/s8.flightlog --static      # статичный дашборд

Управление анимацией:
    Пробел — пауза / продолжение
    R      — перемотка в начало

Структура (для переноса в отдельное приложение):
    FlightLogViewer  — класс вьюера, принимает data dict из load_log()
    main()           — CLI + диалог открытия файла
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# flight_logger.py должен быть рядом с viewer.py
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flight_logger import load_log, print_log_info

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Цветовая схема
# ---------------------------------------------------------------------------
CLR = dict(
    traj        = "royalblue",
    h           = "royalblue",
    h_ref       = "black",
    Va          = "steelblue",
    alpha_true  = "gray",
    alpha_probe = "steelblue",
    alpha_est   = "tomato",
    delta_e     = "saddlebrown",
    throttle    = "darkgreen",
    E_thrust    = "purple",
    warn_zone   = "orange",
    crit_zone   = "red",
)

ALERT_BG    = ["#e8f5e9", "#fff9c4", "#ffe0b2", "#ffcdd2"]
ALERT_LABEL = ["НОРМ", "ПРЕД", "КРИТ", "СРЫВ"]
ALERT_CLR   = ["green", "goldenrod", "darkorange", "crimson"]


# ---------------------------------------------------------------------------
class FlightLogViewer:
    """
    Вьюер одного полётного лога.

    Параметры
    ---------
    data        : dict из load_log()
    anim_speed  : множитель скорости воспроизведения (1.0 = реальное время)
    fps         : частота кадров анимации
    """

    N_RIGHT = 6          # число правых субплотов (фиксировано)

    def __init__(self, data: dict, anim_speed: float = 2.0, fps: int = 25):
        self.data  = data
        self.meta  = data["meta"]
        self.speed = anim_speed
        self.fps   = fps
        self._paused = False
        self._anim   = None

        self._prepare_arrays()

    # ------------------------------------------------------------------
    # Подготовка данных
    # ------------------------------------------------------------------
    def _prepare_arrays(self):
        d = self.data
        m = self.meta

        self.t         = d["t"]
        self.n         = len(self.t)
        self.dt        = self.t[1] - self.t[0] if self.n > 1 else 0.01

        self.h         = d["h"]
        self.x         = d["x"]
        self.Va        = d["Va"]
        self.theta_deg = np.degrees(d["theta"])
        self.q_deg     = np.degrees(d["q"])
        self.alpha_deg = np.degrees(d["alpha_true"])
        self.gamma_deg = np.degrees(d["gamma"])
        self.delta_e_deg = np.degrees(d["delta_e"])
        self.throttle  = d["throttle"]
        self.E_thrust  = d["E_thrust"]
        self.alert     = d["alert"].astype(int)

        self.has_probe    = m.get("has_probe", False)
        self.has_est      = m.get("has_est",   False)
        self.has_h_ref    = m.get("has_h_ref", False)
        self.has_thr_ref  = m.get("has_theta_ref", False)
        self.has_paired   = m.get("has_paired", False)

        self.alpha_probe_deg = (np.degrees(d["alpha_probe"])
                                if self.has_probe else None)
        self.alpha_est_deg   = (np.degrees(d["alpha_est"])
                                if self.has_est   else None)
        self.h_ref           = (d["h_ref"]
                                if self.has_h_ref  else None)
        self.theta_ref_deg   = (np.degrees(d["theta_ref"])
                                if self.has_thr_ref else None)

        al = m.get("alert_levels", {}).get("thresholds_deg", {})
        self.warn_deg  = al.get("warning", 15.0)
        self.crit_deg  = al.get("crit",    20.0)
        self.stall_deg = al.get("stall",   27.0)

        self.events = m.get("events", [])
        trim = m.get("trim", {})
        self.thr_trim = trim.get("throttle")

        # Прореживание кадров
        stride = max(1, int(round(self.speed / self.fps / self.dt)))
        self._idx = list(range(0, self.n, stride))
        self._frame_count = len(self._idx)

    # ------------------------------------------------------------------
    # Построение фигуры
    # ------------------------------------------------------------------
    def _build_figure(self):
        m   = self.meta
        title_top = (f"{m.get('scenario', '?')}\n"
                     f"{m.get('description', '')}   "
                     f"| Ветер Vwx={m.get('wind',{}).get('Vw_const', 0):+.1f} м/с   "
                     f"| Сохранён: {m.get('saved_at','?')}")

        fig = plt.figure(figsize=(15, 14))
        fig.suptitle(title_top, fontsize=9, fontweight="bold")

        gs = gridspec.GridSpec(
            self.N_RIGHT, 2, figure=fig,
            width_ratios=[1.4, 1],
            hspace=0.68, wspace=0.40,
            left=0.06, right=0.97, top=0.93, bottom=0.05,
        )

        ax_traj = fig.add_subplot(gs[:, 0])
        ax_h   = fig.add_subplot(gs[0, 1])
        ax_Va  = fig.add_subplot(gs[1, 1], sharex=ax_h)
        ax_al  = fig.add_subplot(gs[2, 1], sharex=ax_h)
        ax_de  = fig.add_subplot(gs[3, 1], sharex=ax_h)
        ax_thr = fig.add_subplot(gs[4, 1], sharex=ax_h)
        ax_E   = fig.add_subplot(gs[5, 1], sharex=ax_h)

        return fig, ax_traj, ax_h, ax_Va, ax_al, ax_de, ax_thr, ax_E

    # ------------------------------------------------------------------
    # Статичная отрисовка фона (вызывается один раз)
    # ------------------------------------------------------------------
    def _draw_static(self, ax_traj, ax_h, ax_Va, ax_al, ax_de, ax_thr, ax_E):
        t = self.t
        t_end = t[-1]

        # ---- Траектория ------------------------------------------------
        _px = max((self.x.max() - self.x.min()) * 0.06, 10.0)
        _ph = max((self.h.max() - self.h.min()) * 0.28, 20.0)
        ax_traj.set_xlim(self.x.min() - _px, self.x.max() + _px)
        ax_traj.set_ylim(self.h.min() - _ph, self.h.max() + _ph)
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.set_xlabel("x, м", fontsize=9)
        ax_traj.set_ylabel("h, м", fontsize=9)
        ax_traj.set_title("Траектория", fontsize=9)
        ax_traj.grid(True, ls="--", alpha=0.5)
        ax_traj.plot(self.x, self.h, color="lightsteelblue", lw=1.2, alpha=0.4)
        ax_traj.plot(self.x[0],  self.h[0],  "go", ms=8, zorder=5, label="старт")
        ax_traj.plot(self.x[-1], self.h[-1], "rs", ms=8, zorder=5, label="финиш")
        if self.has_h_ref:
            for hv in np.unique(self.h_ref[~np.isnan(self.h_ref)]):
                ax_traj.axhline(hv, color="gray", lw=1.0, ls="--", alpha=0.5)
        ax_traj.legend(fontsize=8, loc="upper left")

        # ---- Общие настройки правых -------------------------------------
        right_axes = (ax_h, ax_Va, ax_al, ax_de, ax_thr, ax_E)
        for ax in right_axes:
            ax.set_xlim(0.0, t_end)
            ax.grid(True, ls="--", alpha=0.5)
            ax.tick_params(labelsize=8)
            for ev in self.events:
                ax.axvline(ev["t"], color=ev.get("color","gray"),
                           lw=0.9, ls=":", alpha=0.7,
                           label=(f"{ev['label']} t={ev['t']:.0f}с"
                                  if ax is ax_h else None))
        for ax in right_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax_E.set_xlabel("Время, с", fontsize=9)
        if self.events:
            ax_h.legend(fontsize=7, loc="upper right")

        # ---- Высота -----------------------------------------------------
        ax_h.set_ylabel("h, м", fontsize=9)
        _pad = max((self.h.max() - self.h.min()) * 0.25, 8.0)
        ax_h.set_ylim(self.h.min() - _pad, self.h.max() + _pad)
        ax_h.plot(t, self.h, color="lightsteelblue", lw=1.0, alpha=0.4)
        if self.has_h_ref:
            ax_h.plot(t, self.h_ref, color=CLR["h_ref"], lw=1.1,
                      ls="--", alpha=0.55, label="h_ref")
            ax_h.legend(fontsize=7, loc="upper right")

        # ---- Воздушная скорость -----------------------------------------
        ax_Va.set_ylabel("Va, м/с", fontsize=9)
        _vpad = max((self.Va.max() - self.Va.min()) * 0.25, 1.5)
        ax_Va.set_ylim(self.Va.min() - _vpad, self.Va.max() + _vpad)
        ax_Va.plot(t, self.Va, color="lightsteelblue", lw=1.0, alpha=0.45)

        # ---- Угол атаки -------------------------------------------------
        ax_al.set_ylabel("α, °", fontsize=9)
        _al_vals = [self.alpha_deg]
        if self.alpha_probe_deg is not None: _al_vals.append(self.alpha_probe_deg)
        if self.alpha_est_deg   is not None: _al_vals.append(self.alpha_est_deg)
        _all_al = np.concatenate(_al_vals)
        _alpad  = max((_all_al.max() - _all_al.min()) * 0.2, 1.0)
        _al_lo  = _all_al.min() - _alpad
        _al_hi  = max(_all_al.max() + _alpad, self.stall_deg + 5.0)
        ax_al.set_ylim(_al_lo, _al_hi)
        ax_al.axhline(0, color="gray", lw=0.8, ls=":")
        if self.alert.max() > 0:
            ax_al.axhspan(self.warn_deg, self.crit_deg,
                          color=CLR["warn_zone"], alpha=0.10, zorder=0)
            ax_al.axhspan(self.crit_deg, _al_hi + 5,
                          color=CLR["crit_zone"], alpha=0.08, zorder=0)
            ax_al.axhline(self.warn_deg,  color="orange",  lw=1.0, ls="--",
                          alpha=0.8, label=f"пред {self.warn_deg:.0f}°")
            ax_al.axhline(self.crit_deg,  color="red",     lw=1.0, ls="--",
                          alpha=0.8, label=f"крит {self.crit_deg:.0f}°")
            ax_al.axhline(self.stall_deg, color="darkred", lw=1.2, ls="-",
                          alpha=0.7, label=f"срыв {self.stall_deg:.0f}°")
        ax_al.plot(t, self.alpha_deg, color="silver", lw=1.2, alpha=0.5,
                   label="истинный")
        if self.alpha_probe_deg is not None:
            ax_al.plot(t, self.alpha_probe_deg, color=CLR["alpha_probe"],
                       lw=0.8, alpha=0.3)
        if self.alpha_est_deg is not None:
            ax_al.plot(t, self.alpha_est_deg, color=CLR["alpha_est"],
                       lw=0.8, alpha=0.3)
        ax_al.legend(fontsize=7, loc="upper right")

        # ---- Руль высоты ------------------------------------------------
        ax_de.set_ylabel("δe, °", fontsize=9)
        _depad = max(abs(self.delta_e_deg).max() * 0.18, 1.5)
        ax_de.set_ylim(self.delta_e_deg.min() - _depad,
                       self.delta_e_deg.max() + _depad)
        ax_de.axhline(0, color="gray", lw=0.8, ls=":")
        ax_de.plot(t, self.delta_e_deg, color="burlywood", lw=1.0, alpha=0.5)

        # ---- Тяга -------------------------------------------------------
        ax_thr.set_ylabel("Тяга, о.е.", fontsize=9)
        ax_thr.set_ylim(-0.05, 1.05)
        if self.thr_trim is not None:
            ax_thr.axhline(self.thr_trim, color="gray", lw=1.0, ls="--",
                           alpha=0.5, label=f"trim={self.thr_trim:.2f}")
            ax_thr.legend(fontsize=7, loc="upper right")
        ax_thr.plot(t, self.throttle, color="lightgreen", lw=1.0, alpha=0.5)

        # ---- Энергия ----------------------------------------------------
        ax_E.set_ylabel("E, о.е.", fontsize=9)
        ax_E.set_ylim(0, self.E_thrust.max() * 1.08)
        ax_E.set_title(
            f"Накопленная энергия двигателя  "
            f"E_итог = {self.E_thrust[-1]:.1f}",
            fontsize=8, pad=2)
        ax_E.plot(t, self.E_thrust, color="plum", lw=1.0, alpha=0.4)

    # ------------------------------------------------------------------
    # Анимация
    # ------------------------------------------------------------------
    def animate(self):
        fig, ax_traj, ax_h, ax_Va, ax_al, ax_de, ax_thr, ax_E = \
            self._build_figure()
        self._draw_static(ax_traj, ax_h, ax_Va, ax_al, ax_de, ax_thr, ax_E)

        right_axes = (ax_h, ax_Va, ax_al, ax_de, ax_thr, ax_E)

        # Динамические линии
        traj_line,   = ax_traj.plot([], [], color=CLR["traj"], lw=2.0, zorder=3)
        traj_marker, = ax_traj.plot([], [], "b^", ms=11, zorder=6,
                                    markeredgecolor="navy")
        info_box = ax_traj.text(
            0.98, 0.04, "", transform=ax_traj.transAxes, fontsize=8.0,
            ha="right", va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.88),
        )

        ln_h,   pt_h   = ax_h.plot([], [], color=CLR["h"],    lw=1.8)[0], \
                         ax_h.plot([], [], "o", color=CLR["h"], ms=5, zorder=5)[0]
        ln_Va,  pt_Va  = ax_Va.plot([], [], color=CLR["Va"],   lw=1.8)[0], \
                         ax_Va.plot([], [], "o", color=CLR["Va"], ms=5, zorder=5)[0]
        ln_al,  pt_al  = ax_al.plot([], [], color=CLR["alpha_true"],  lw=1.5)[0], \
                         ax_al.plot([], [], "o", color="gray", ms=5, zorder=5)[0]
        ln_de,  pt_de  = ax_de.plot([], [], color=CLR["delta_e"],  lw=1.8)[0], \
                         ax_de.plot([], [], "o", color=CLR["delta_e"], ms=5, zorder=5)[0]
        ln_thr, pt_thr = ax_thr.plot([], [], color=CLR["throttle"], lw=1.8)[0], \
                         ax_thr.plot([], [], "o", color=CLR["throttle"], ms=5, zorder=5)[0]
        ln_E,   pt_E   = ax_E.plot([], [], color=CLR["E_thrust"],  lw=1.8)[0], \
                         ax_E.plot([], [], "o", color=CLR["E_thrust"], ms=5, zorder=5)[0]

        ln_al_probe = (ax_al.plot([], [], color=CLR["alpha_probe"],
                                  lw=1.5, label="зонд")[0]
                       if self.has_probe else None)
        ln_al_est   = (ax_al.plot([], [], color=CLR["alpha_est"],
                                  lw=1.5, ls="--", label="ИНС+GPS")[0]
                       if self.has_est else None)
        if self.has_probe or self.has_est:
            ax_al.legend(fontsize=7, loc="upper right")

        vlines = [ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.65)
                  for ax in right_axes]

        dyn_artists = [traj_line, traj_marker, info_box,
                       ln_h, pt_h, ln_Va, pt_Va,
                       ln_al, pt_al, ln_de, pt_de,
                       ln_thr, pt_thr, ln_E, pt_E,
                       *vlines]
        if ln_al_probe: dyn_artists.append(ln_al_probe)
        if ln_al_est:   dyn_artists.append(ln_al_est)

        def init():
            for art in dyn_artists:
                if hasattr(art, "set_data"):
                    art.set_data([], [])
                elif hasattr(art, "set_text"):
                    art.set_text("")
                elif hasattr(art, "set_xdata"):
                    art.set_xdata([0])
            return dyn_artists

        def update(fn):
            if self._paused:
                return dyn_artists
            i    = self._idx[fn]
            t_cur = self.t[i]
            ts    = self.t[:i+1]

            traj_line.set_data(self.x[:i+1], self.h[:i+1])
            traj_marker.set_data([self.x[i]], [self.h[i]])

            lv  = int(self.alert[i])
            info_box.get_bbox_patch().set_facecolor(ALERT_BG[lv])
            lines = [
                f"t      = {t_cur:6.1f} с",
                f"h      = {self.h[i]:6.1f} м",
                f"Va     = {self.Va[i]:5.1f} м/с",
                f"α_ист  = {self.alpha_deg[i]:+5.2f}°",
            ]
            if self.alpha_probe_deg is not None:
                lines.append(f"α_зонд = {self.alpha_probe_deg[i]:+5.2f}°")
            if self.alpha_est_deg is not None:
                lines.append(f"α_оц   = {self.alpha_est_deg[i]:+5.2f}°")
            lines += [
                f"δe     = {self.delta_e_deg[i]:+5.1f}°",
                f"тяга   = {self.throttle[i]:.3f}",
                f"E      = {self.E_thrust[i]:6.1f}",
                f"ИНД    = {ALERT_LABEL[lv]}",
            ]
            info_box.set_text("\n".join(lines))

            ln_h.set_data(ts, self.h[:i+1]);           pt_h.set_data([t_cur], [self.h[i]])
            ln_Va.set_data(ts, self.Va[:i+1]);          pt_Va.set_data([t_cur], [self.Va[i]])
            ln_al.set_data(ts, self.alpha_deg[:i+1]);   pt_al.set_data([t_cur], [self.alpha_deg[i]])
            ln_de.set_data(ts, self.delta_e_deg[:i+1]); pt_de.set_data([t_cur], [self.delta_e_deg[i]])
            ln_thr.set_data(ts, self.throttle[:i+1]);   pt_thr.set_data([t_cur], [self.throttle[i]])
            ln_E.set_data(ts, self.E_thrust[:i+1]);     pt_E.set_data([t_cur], [self.E_thrust[i]])
            if ln_al_probe is not None:
                ln_al_probe.set_data(ts, self.alpha_probe_deg[:i+1])
            if ln_al_est is not None:
                ln_al_est.set_data(ts, self.alpha_est_deg[:i+1])

            for vl in vlines:
                vl.set_xdata([t_cur])
            return dyn_artists

        def on_key(event):
            if event.key == " ":
                self._paused = not self._paused
            elif event.key in ("r", "R"):
                self._paused = False
                if self._anim:
                    self._anim.frame_seq = self._anim.new_frame_seq()

        fig.canvas.mpl_connect("key_press_event", on_key)

        self._anim = FuncAnimation(
            fig, update, frames=self._frame_count,
            init_func=init,
            interval=1000.0 / self.fps,
            blit=True, repeat=True,
        )

        fig.text(0.5, 0.005,
                 "Пробел — пауза/пуск  |  R — перемотка",
                 ha="center", fontsize=8, color="gray")
        plt.show()

    # ------------------------------------------------------------------
    # Статичный дашборд (без анимации)
    # ------------------------------------------------------------------
    def show_static(self):
        fig, ax_traj, ax_h, ax_Va, ax_al, ax_de, ax_thr, ax_E = \
            self._build_figure()
        self._draw_static(ax_traj, ax_h, ax_Va, ax_al, ax_de, ax_thr, ax_E)

        t = self.t
        ax_h.plot(t, self.h,             color=CLR["h"],        lw=1.8)
        ax_Va.plot(t, self.Va,            color=CLR["Va"],       lw=1.8)
        ax_al.plot(t, self.alpha_deg,     color=CLR["alpha_true"], lw=1.8)
        ax_de.plot(t, self.delta_e_deg,   color=CLR["delta_e"],  lw=1.8)
        ax_thr.plot(t, self.throttle,     color=CLR["throttle"], lw=1.8)
        ax_E.plot(t, self.E_thrust,       color=CLR["E_thrust"], lw=1.8)
        if self.has_probe:
            ax_al.plot(t, self.alpha_probe_deg, color=CLR["alpha_probe"],
                       lw=1.5, label="зонд")
        if self.has_est:
            ax_al.plot(t, self.alpha_est_deg, color=CLR["alpha_est"],
                       lw=1.5, ls="--", label="ИНС+GPS")
        if self.has_probe or self.has_est:
            ax_al.legend(fontsize=7, loc="upper right")

        ax_traj.plot(self.x, self.h, color=CLR["traj"], lw=2.0, zorder=3)

        # Сводка цифр
        m = self.meta
        trim = m.get("trim", {})
        summary = (
            f"alpha_trim = {trim.get('alpha_deg', '?'):.2f}°\n"
            f"thr_trim   = {trim.get('throttle', '?'):.3f}\n"
            f"alpha_max  = {self.alpha_deg.max():.2f}°\n"
            f"Va_min     = {self.Va.min():.1f} м/с\n"
            f"E_total    = {self.E_thrust[-1]:.1f}\n"
            f"t_stall    = {np.sum(self.alert >= 3) * self.dt:.2f} с"
        )
        ax_traj.text(
            0.02, 0.04, summary,
            transform=ax_traj.transAxes, fontsize=8,
            ha="left", va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.88),
        )
        plt.show()


# ---------------------------------------------------------------------------
# Открытие файла
# ---------------------------------------------------------------------------

def _open_dialog() -> str:
    """Открыть диалог выбора файла. Возвращает путь или '' если отменён."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Открыть полётный лог",
            filetypes=[("Flight log", "*.flightlog"), ("NPZ", "*.npz"),
                       ("Все файлы", "*.*")],
        )
        root.destroy()
        return path
    except Exception as e:
        print(f"Диалог недоступен: {e}")
        return ""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Вьюер полётных логов (.flightlog)")
    parser.add_argument("file", nargs="?", default=None,
                        help="Путь к файлу .flightlog")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Скорость анимации (по умолчанию 2.0)")
    parser.add_argument("--fps",   type=int,   default=25,
                        help="Частота кадров (по умолчанию 25)")
    parser.add_argument("--static", action="store_true",
                        help="Статичный дашборд без анимации")
    args = parser.parse_args()

    path = args.file
    if not path:
        path = _open_dialog()
    if not path:
        print("Файл не выбран. Выход.")
        return

    if not os.path.exists(path) and os.path.exists(path + ".npz"):
        path = path + ".npz"

    print(f"Загрузка: {path}")
    data = load_log(path)
    print_log_info(data)

    viewer = FlightLogViewer(data, anim_speed=args.speed, fps=args.fps)

    if args.static:
        viewer.show_static()
    else:
        viewer.animate()


if __name__ == "__main__":
    main()
