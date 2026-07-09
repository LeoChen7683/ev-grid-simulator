"""
EV Charging Load Impact Simulator
Week 2-3: EV Load Modeling & Grid Stress Analysis
Author: Leo

What this script does:
- Models "dumb" EV charging (everyone plugs in when they get home)
- Tests 3 penetration scenarios: 10%, 25%, 50% of homes with EVs
- Compares results against the no-EV baseline
- Flags voltage violations and overloaded lines
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# ─── Load profile (same as week 1) ────────────────────────────────────────────
def get_base_load_profile():
    return np.array([
        0.60, 0.55, 0.52, 0.50, 0.51, 0.55, 0.65, 0.80,
        0.85, 0.82, 0.78, 0.75, 0.73, 0.72, 0.74, 0.78,
        0.85, 0.95, 1.00, 0.98, 0.95, 0.88, 0.78, 0.68
    ])


# ─── EV charging profile ───────────────────────────────────────────────────────
def get_ev_charging_profile(penetration_pct, charger_kw=7.2):
    ev_schedule = np.array([
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.02, 0.10, 0.35,
        0.60, 0.75, 0.70, 0.45, 0.20, 0.05
    ])
    ev_load_per_home_mw = (charger_kw / 1000) * penetration_pct * ev_schedule
    return ev_load_per_home_mw


# ─── Estimate homes per bus ────────────────────────────────────────────────────
def estimate_homes_per_bus(net):
    homes_per_bus = 150
    print(f"Assuming {homes_per_bus} homes per bus (Long Island residential estimate)")
    return homes_per_bus


# ─── Run simulation with EV load ───────────────────────────────────────────────
def run_ev_simulation(net, base_profile, ev_profile, label="ev_10pct"):
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()
    homes  = estimate_homes_per_bus(net)

    voltage_records   = []
    line_load_records = []

    for h in range(24):
        scaled_p = base_p * base_profile[h]
        scaled_q = base_q * base_profile[h]
        ev_addition_mw = ev_profile[h] * homes

        net.load.p_mw   = scaled_p + ev_addition_mw
        net.load.q_mvar = scaled_q

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


# ─── Comparison plot ───────────────────────────────────────────────────────────
def plot_scenario_comparison(results_dict):
    colors = {
        "baseline" : "#2a9d8f",
        "ev_10pct" : "#e9c46a",
        "ev_25pct" : "#f4a261",
        "ev_50pct" : "#e63946",
    }
    labels_nice = {
        "baseline" : "No EVs (Baseline)",
        "ev_10pct" : "10% EV Penetration",
        "ev_25pct" : "25% EV Penetration",
        "ev_50pct" : "50% EV Penetration",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label, (voltages, line_loadings) in results_dict.items():
        v_cols = [c for c in voltages.columns if c.startswith("bus_")]
        l_cols = [c for c in line_loadings.columns if c.startswith("line_")]
        min_v  = voltages[v_cols].min(axis=1)
        max_l  = line_loadings[l_cols].max(axis=1)
        color  = colors.get(label, "gray")
        name   = labels_nice.get(label, label)

        axes[0].plot(min_v.index, min_v.values, marker="o", linewidth=2, color=color, label=name)
        axes[1].plot(max_l.index, max_l.values, marker="s", linewidth=2, color=color, label=name)

    axes[0].axhline(0.95, color="orange", linestyle="--", linewidth=1, label="ANSI Lower Limit (0.95)")
    axes[0].axhline(0.90, color="red",    linestyle="--", linewidth=1, label="Critical Limit (0.90)")
    axes[0].set_ylabel("Min Bus Voltage (p.u.)")
    axes[0].set_title("Minimum Bus Voltage Across All Scenarios")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0.85, 1.06)

    axes[1].axhline(80,  color="orange", linestyle="--", linewidth=1, label="Warning (80%)")
    axes[1].axhline(100, color="red",    linestyle="--", linewidth=1, label="Thermal Limit (100%)")
    axes[1].set_ylabel("Max Line Loading (%)")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_title("Worst-Case Line Loading Across All Scenarios")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(range(24))

    plt.tight_layout()
    plt.savefig("results/scenario_comparison.png", dpi=150)
    plt.show()
    print("Comparison plot saved: results/scenario_comparison.png")


# ─── Violation report ─────────────────────────────────────────────────────────
def print_violation_report(results_dict):
    print("\n" + "="*60)
    print("  VIOLATION REPORT — EV PENETRATION SCENARIOS")
    print("="*60)
    print(f"{'Scenario':<20} {'Min V (pu)':<15} {'V Violations':<18} {'Max Line %':<15} {'Line Viol.'}")
    print("-"*60)
    for label, (voltages, line_loadings) in results_dict.items():
        v_cols = [c for c in voltages.columns if c.startswith("bus_")]
        l_cols = [c for c in line_loadings.columns if c.startswith("line_")]
        v_data = voltages[v_cols]
        l_data = line_loadings[l_cols]
        min_v  = v_data.values.min()
        v_viol = (v_data.min(axis=1) < 0.95).sum()
        max_l  = l_data.values.max()
        l_viol = (l_data.max(axis=1) > 80).sum()
        print(f"{label:<20} {min_v:<15.4f} {v_viol:<18} {max_l:<15.2f} {l_viol}")
    print("="*60)
    print("\nNote: 'Violations' = hours where any bus/line exceeds the threshold.\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    net          = pn.case33bw()
    base_profile = get_base_load_profile()

    scenarios = {
        "baseline" : 0.00,
        "ev_10pct" : 0.10,
        "ev_25pct" : 0.25,
        "ev_50pct" : 0.50,
    }

    results = {}
    for label, penetration in scenarios.items():
        print(f"\nRunning scenario: {label} ({int(penetration*100)}% EV penetration)...")
        ev_profile = get_ev_charging_profile(penetration)
        v, l = run_ev_simulation(net, base_profile, ev_profile, label=label)
        results[label] = (v, l)

    print_violation_report(results)

    print("\nGenerating comparison plots...")
    plot_scenario_comparison(results)

    print("\nDone! Next step: run week3_smart_charging.py to see how we fix this.")