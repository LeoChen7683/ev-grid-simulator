"""
EV Charging Load Impact Simulator
Week 3: Smart Charging — Time-Shifting to Protect the Grid
Author: Leo

What this script does:
- Takes the worst-case 50% EV penetration scenario
- Implements a simple "valley-fill" smart charging algorithm
- Shifts EV charging away from the 6-9pm peak into overnight low-demand hours
- Compares dumb vs smart charging side-by-side
- Quantifies the improvement in voltage and line loading
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)


def get_base_load_profile():
    return np.array([
        0.60, 0.55, 0.52, 0.50, 0.51, 0.55, 0.65, 0.80,
        0.85, 0.82, 0.78, 0.75, 0.73, 0.72, 0.74, 0.78,
        0.85, 0.95, 1.00, 0.98, 0.95, 0.88, 0.78, 0.68
    ])


def get_dumb_ev_profile(penetration=0.50, charger_kw=7.2):
    """Everyone plugs in when they get home — worst case."""
    ev_schedule = np.array([
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.02, 0.10, 0.35,
        0.60, 0.75, 0.70, 0.45, 0.20, 0.05
    ])
    return (charger_kw / 1000) * penetration * ev_schedule


def get_smart_ev_profile(penetration=0.50, charger_kw=7.2):
    """
    Valley-fill smart charging: same total energy, spread into
    overnight low-demand hours (11pm - 6am).
    """
    dumb = get_dumb_ev_profile(penetration, charger_kw)
    total_energy = dumb.sum()

    smart = np.zeros(24)
    overnight_hours = [23, 0, 1, 2, 3, 4, 5, 6]
    energy_per_hour = total_energy / len(overnight_hours)
    for h in overnight_hours:
        smart[h] = energy_per_hour

    print(f"\nDumb charging total energy  : {dumb.sum():.6f} MW·h/home")
    print(f"Smart charging total energy : {smart.sum():.6f} MW·h/home")
    print("(Must be equal — same energy, just shifted in time)")
    return smart


def run_simulation(net, base_profile, ev_profile, label):
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()
    homes  = 150

    voltage_records   = []
    line_load_records = []

    for h in range(24):
        net.load.p_mw   = base_p * base_profile[h] + ev_profile[h] * homes
        net.load.q_mvar = base_q * base_profile[h]

        v_row = {"hour": h}
        l_row = {"hour": h}

        try:
            pp.runpp(net, algorithm="nr", numba=False, max_iteration=50)
            for bus_idx in net.res_bus.index:
                v_row[f"bus_{bus_idx}_pu"] = net.res_bus.at[bus_idx, "vm_pu"]
            for line_idx in net.res_line.index:
                l_row[f"line_{line_idx}_pct"] = net.res_line.at[line_idx, "loading_percent"]
        except pp.powerflow.LoadflowNotConverged:
            print(f"  WARNING: Grid overloaded at hour {h} — power flow did not converge!")
            for bus_idx in net.bus.index:
                v_row[f"bus_{bus_idx}_pu"] = 0.85
            for line_idx in net.line.index:
                l_row[f"line_{line_idx}_pct"] = 120.0

        voltage_records.append(v_row)
        line_load_records.append(l_row)

    net.load.p_mw   = base_p
    net.load.q_mvar = base_q

    voltages      = pd.DataFrame(voltage_records).set_index("hour")
    line_loadings = pd.DataFrame(line_load_records).set_index("hour")
    voltages.to_csv(f"results/{label}_voltages.csv")
    line_loadings.to_csv(f"results/{label}_line_loadings.csv")
    return voltages, line_loadings


def plot_dumb_vs_smart(base_profile, dumb_ev, smart_ev, results_dumb, results_smart):
    v_cols = [c for c in results_dumb[0].columns if c.startswith("bus_")]
    l_cols = [c for c in results_dumb[1].columns if c.startswith("line_")]
    hours  = range(24)
    homes  = 150

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # Plot 1: Total feeder load
    total_dumb  = base_profile + dumb_ev  * homes
    total_smart = base_profile + smart_ev * homes
    axes[0].fill_between(hours, base_profile, alpha=0.5, color="#2a9d8f", label="Base residential load")
    axes[0].plot(hours, total_dumb,  color="#e63946", linewidth=2, label="+ Dumb EV charging")
    axes[0].plot(hours, total_smart, color="#457b9d", linewidth=2, linestyle="--", label="+ Smart EV charging")
    axes[0].set_ylabel("Load Multiplier")
    axes[0].set_title("Total Feeder Load Profile — Dumb vs Smart Charging (50% EV Penetration)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Plot 2: Minimum voltage
    min_v_dumb  = results_dumb[0][v_cols].min(axis=1)
    min_v_smart = results_smart[0][v_cols].min(axis=1)
    axes[1].plot(hours, min_v_dumb.values,  marker="o", color="#e63946", linewidth=2, label="Dumb charging")
    axes[1].plot(hours, min_v_smart.values, marker="s", color="#457b9d", linewidth=2, linestyle="--", label="Smart charging")
    axes[1].axhline(0.95, color="orange", linestyle=":", linewidth=1.5, label="ANSI Lower Limit (0.95)")
    axes[1].set_ylabel("Min Bus Voltage (p.u.)")
    axes[1].set_title("Minimum Bus Voltage — Dumb vs Smart")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0.85, 1.06)

    # Plot 3: Max line loading
    max_l_dumb  = results_dumb[1][l_cols].max(axis=1)
    max_l_smart = results_smart[1][l_cols].max(axis=1)
    axes[2].plot(hours, max_l_dumb.values,  marker="o", color="#e63946", linewidth=2, label="Dumb charging")
    axes[2].plot(hours, max_l_smart.values, marker="s", color="#457b9d", linewidth=2, linestyle="--", label="Smart charging")
    axes[2].axhline(80,  color="orange", linestyle=":", linewidth=1.5, label="Warning (80%)")
    axes[2].axhline(100, color="red",    linestyle=":", linewidth=1.5, label="Thermal limit (100%)")
    axes[2].set_ylabel("Max Line Loading (%)")
    axes[2].set_xlabel("Hour of Day")
    axes[2].set_title("Worst-Case Line Loading — Dumb vs Smart")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(range(24))

    plt.tight_layout()
    plt.savefig("results/dumb_vs_smart_comparison.png", dpi=150)
    plt.show()
    print("Comparison saved: results/dumb_vs_smart_comparison.png")


def print_improvement_summary(results_dumb, results_smart):
    v_cols = [c for c in results_dumb[0].columns if c.startswith("bus_")]
    l_cols = [c for c in results_dumb[1].columns if c.startswith("line_")]

    dumb_min_v  = results_dumb[0][v_cols].values.min()
    smart_min_v = results_smart[0][v_cols].values.min()
    dumb_max_l  = results_dumb[1][l_cols].values.max()
    smart_max_l = results_smart[1][l_cols].values.max()
    dumb_v_viols  = (results_dumb[0][v_cols].min(axis=1)  < 0.95).sum()
    smart_v_viols = (results_smart[0][v_cols].min(axis=1) < 0.95).sum()
    dumb_l_viols  = (results_dumb[1][l_cols].max(axis=1)  > 80).sum()
    smart_l_viols = (results_smart[1][l_cols].max(axis=1) > 80).sum()

    print("\n" + "="*55)
    print("  SMART CHARGING IMPROVEMENT SUMMARY (50% Penetration)")
    print("="*55)
    print(f"{'Metric':<35} {'Dumb':>8}  {'Smart':>8}")
    print("-"*55)
    print(f"{'Min bus voltage (p.u.)':<35} {dumb_min_v:>8.4f}  {smart_min_v:>8.4f}")
    print(f"{'Voltage improvement':<35} {'':>8}  {smart_min_v - dumb_min_v:>+8.4f}")
    print(f"{'Hours w/ voltage violations':<35} {dumb_v_viols:>8}  {smart_v_viols:>8}")
    print(f"{'Max line loading (%)':<35} {dumb_max_l:>8.2f}  {smart_max_l:>8.2f}")
    print(f"{'Line loading improvement':<35} {'':>8}  {smart_max_l - dumb_max_l:>+8.2f}%")
    print(f"{'Hours w/ line warnings (>80%)':<35} {dumb_l_viols:>8}  {smart_l_viols:>8}")
    print("="*55)
    print("\nConclusion: Smart charging shifts load to off-peak hours,")
    print("reducing peak stress without reducing EV convenience.")
    print("This is the core principle behind LIPA's demand response programs.\n")


if __name__ == "__main__":
    net          = pn.case33bw()
    base_profile = get_base_load_profile()
    dumb_ev      = get_dumb_ev_profile(penetration=0.50)
    smart_ev     = get_smart_ev_profile(penetration=0.50)

    print("\nRunning dumb charging simulation (50% EV)...")
    results_dumb  = run_simulation(net, base_profile, dumb_ev,  label="dumb_50pct")

    print("\nRunning smart charging simulation (50% EV)...")
    results_smart = run_simulation(net, base_profile, smart_ev, label="smart_50pct")

    print_improvement_summary(results_dumb, results_smart)

    print("Generating comparison plots...")
    plot_dumb_vs_smart(base_profile, dumb_ev, smart_ev, results_dumb, results_smart)

    print("\nAll done! Your full simulation is complete.")
    print("Check the results/ folder — you now have CSVs and plots for everything.")