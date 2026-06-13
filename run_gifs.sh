#!/bin/bash
export MPLBACKEND=Agg
cd "$(dirname "$0")"
echo "=== s6 ===" && python scenarios/s6_speed_control.py results/s6.gif && echo "s6 done"
echo "=== s7 ===" && python scenarios/s7_alpha_estimation.py results/s7.gif && echo "s7 done"
echo "=== s8 ===" && python scenarios/s8_energy_comparison.py results/s8.gif && echo "s8 done"
echo "=== s8b ===" && python scenarios/s8_b_motor_work.py results/s8b.gif && echo "s8b done"
echo "=== s9 ===" && python scenarios/s9_paired_probe_comparison.py results/s9.gif && echo "s9 done"
echo "=== s10 (статичный PNG) ===" && python scenarios/s10_aua_gust_protection.py results/s10.png && echo "s10 done"
echo "=== s10b ===" && python scenarios/s10_b_probe_vs_estimate.py results/s10b.gif && echo "s10b done"
echo "=== s11 ===" && python scenarios/s11_lqr_climb.py results/s11.gif && echo "s11 done"
echo "=== ВСЕ ГОТОВО ==="
