"""
EV Charging Load Impact Simulator
Week 1-2: Base Grid Setup & Baseline Power Flow
Author: Leo
Project: EV Charging Impact on a Distribution Feeder (Long Island Case Study)

What this script does:
- Loads the standard IEEE 33-bus distribution test network
- Attaches a realistic residential load profile (24-hour day)
- Runs a power flow simulation for each hour
- Saves voltage and line loading results to CSV
- Plots voltage profiles across the network
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ─── Output folder ────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True) # creates a folder called results to save to output files
                                      # exist_ok=True means it wont crash when if it already exists

# ─── 1. Build the IEEE 33-bus network ─────────────────────────────────────────
def build_base_network():
    """Load the standard IEEE 33-bus radial distribution test feeder."""
    net = pn.case33bw()  # Built into pandapower — no download needed + loads the IEEE 33 bus test network
                         # 33 nodes connected by 32 lines in a radial layout
    print(f"Network loaded: {len(net.bus)} buses, {len(net.line)} lines") # prints confirmation that the network loaded + 
    print(f"Total base load: {net.load.p_mw.sum():.3f} MW, {net.load.q_mvar.sum():.3f} MVAr")
    return net
# line 32-33 - net.bus is a table for all buses, len() counts them, p_mw is the active power load in MW, q_mvar is the reactive power 
# .sum() adds the whole column together, :.3f formats the number to 3 decimal places

# ─── 2. Realistic 24-hour residential load profile ────────────────────────────
def get_load_profile():
    """
    Normalized hourly load multipliers for a typical residential day.
    Based on typical U.S. residential demand curves (EIA data patterns).
    1.0 = base load from the network file.
    """
    # Hour:  0     1     2     3     4     5     6     7     8     9    10    11
    profile = [
        0.60, 0.55, 0.52, 0.50, 0.51, 0.55, 0.65, 0.80, 0.85, 0.82, 0.78, 0.75,
    #  12    13    14    15    16    17    18    19    20    21    22    23
        0.73, 0.72, 0.74, 0.78, 0.85, 0.95, 1.00, 0.98, 0.95, 0.88, 0.78, 0.68
    ]
    return np.array(profile)
# get_load_profile() - 24 numbers, one per hour. Each is a multiplier representing how much of the maximum load the grid carries at that hour
# for ex, at 3am, the load is 50% of the base load or at 6pm its 100%. 
# Then I convert the list to a numpy array so I can do math on it later (multiply by the base load to get actual load at that hour)


# ─── 3. Run hourly power flow ─────────────────────────────────────────────────
def run_hourly_simulation(net, load_profile, label="baseline"):
    """
    Runs a power flow for each of the 24 hours.
    Stores per-bus voltages and per-line loadings.
    Returns two DataFrames: voltages and line_loadings.
    """
    hours = range(24)
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()
# 65-66 - saves the original load values before modifying them each hour, so we can restore them later. 
# .values.copy() makes a copy of the array so we don't accidentally modify the original data in net.load.p_mw and net.load.q_mvar
    voltage_records   = []   # one row per hour
    line_load_records = []

    for h in hours:
        multiplier = load_profile[h]

        # Scale all loads by the hourly multiplier
        net.load.p_mw   = base_p * multiplier
        net.load.q_mvar = base_q * multiplier
        # loops through all 24 hours, scaling every load on every bus by the multiplier for that hour. 
        # For example, if the base load is 1 MW and the multiplier is 0.5, the load becomes 0.5 MW.
        
        # Run Newton-Raphson power flow
        pp.runpp(net, algorithm="nr", numba=False)
        # runs the power flow calculation using the Newton-Raphson method (algorithm="nr" means Newton-Raphson - adjusts all voltages simultaneously each iteration, converges in 3-5 steps).
        # numba=False turns off a speed optimization thast causes issues on some machines.

        # --- voltages (per-unit) for every bus ---
        v_row = {"hour": h, "multiplier": multiplier}
        for bus_idx in net.res_bus.index:
            v_row[f"bus_{bus_idx}_pu"] = net.res_bus.at[bus_idx, "vm_pu"] # after the power flow runs, net.res_bus contains the results. vm_pu is voltage magnitude in per-unit. 1.0 is perfect, 0.95 is the ANSI lower limit, below 0.90 is critical
        voltage_records.append(v_row) # records every bus voltage for this hour

        # --- line loading (%) for every line ---
        l_row = {"hour": h, "multiplier": multiplier}
        for line_idx in net.res_line.index:
            l_row[f"line_{line_idx}_pct"] = net.res_line.at[line_idx, "loading_percent"] # does the same thing but for lines. loading_percent is how loaded each wire is as a percentage of its thermal limit. Above 80% is a warning, above 100% means the wire is overheating.
        line_load_records.append(l_row)

    # Restore original loads
    net.load.p_mw   = base_p
    net.load.q_mvar = base_q

    voltages     = pd.DataFrame(voltage_records).set_index("hour")
    line_loadings = pd.DataFrame(line_load_records).set_index("hour")

    voltages.to_csv(f"results/{label}_voltages.csv")
    line_loadings.to_csv(f"results/{label}_line_loadings.csv")
    print(f"\n[{label}] Results saved to results/ folder.")
    return voltages, line_loadings


# ─── 4. Plotting ──────────────────────────────────────────────────────────────
def plot_voltage_heatmap(voltages, label="baseline"):
    """Heatmap: x=bus, y=hour, color=voltage (pu)"""
    # Extract just the voltage columns
    v_cols = [c for c in voltages.columns if c.startswith("bus_")]
    v_data = voltages[v_cols].values  # shape (24, n_buses)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(v_data, aspect="auto", cmap="RdYlGn",
                   vmin=0.90, vmax=1.05, origin="upper")
    plt.colorbar(im, ax=ax, label="Voltage (p.u.)")
    ax.set_xlabel("Bus Index")
    ax.set_ylabel("Hour of Day")
    ax.set_title(f"Bus Voltage Profile — {label.upper()} Scenario")
    ax.set_xticks(range(len(v_cols)))
    ax.set_xticklabels([c.replace("bus_","").replace("_pu","") for c in v_cols],
                       rotation=90, fontsize=7)
    # ANSI lower voltage limit
    ax.axhline(y=17, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/{label}_voltage_heatmap.png", dpi=150)
    plt.show()
    print(f"Heatmap saved: results/{label}_voltage_heatmap.png")


def plot_min_voltage_over_day(voltages, label="baseline"):
    """Line plot of the minimum bus voltage across the network each hour."""
    v_cols = [c for c in voltages.columns if c.startswith("bus_")]
    min_v = voltages[v_cols].min(axis=1)   # worst-case bus each hour

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(min_v.index, min_v.values, marker="o", linewidth=2,
            color="#e63946", label="Min bus voltage")
    ax.axhline(y=0.95, color="orange", linestyle="--", linewidth=1.2,
               label="ANSI lower limit (0.95 p.u.)")
    ax.axhline(y=0.90, color="red", linestyle="--", linewidth=1.2,
               label="Critical lower limit (0.90 p.u.)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Worst-Case Bus Voltage — {label.upper()} Scenario")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{label}_min_voltage.png", dpi=150)
    plt.show()
    print(f"Min voltage plot saved: results/{label}_min_voltage.png")


def plot_max_line_loading(line_loadings, label="baseline"):
    """Line plot of the worst-case line loading each hour."""
    l_cols = [c for c in line_loadings.columns if c.startswith("line_")]
    max_l = line_loadings[l_cols].max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(max_l.index, max_l.values, marker="s", linewidth=2,
            color="#457b9d", label="Max line loading")
    ax.axhline(y=80, color="orange", linestyle="--", linewidth=1.2,
               label="Warning threshold (80%)")
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1.2,
               label="Thermal limit (100%)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Line Loading (%)")
    ax.set_title(f"Worst-Case Line Loading — {label.upper()} Scenario")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{label}_max_line_loading.png", dpi=150)
    plt.show()
    print(f"Line loading plot saved: results/{label}_max_line_loading.png")


# ─── 5. Summary stats ─────────────────────────────────────────────────────────
def print_summary(voltages, line_loadings, label="baseline"):
    v_cols = [c for c in voltages.columns if c.startswith("bus_")]
    l_cols = [c for c in line_loadings.columns if c.startswith("line_")]

    v_data = voltages[v_cols]
    l_data = line_loadings[l_cols]

    print(f"\n{'='*50}")
    print(f"  SUMMARY — {label.upper()} SCENARIO")
    print(f"{'='*50}")
    print(f"  Min voltage across all buses & hours : {v_data.values.min():.4f} p.u.")
    print(f"  Max voltage across all buses & hours : {v_data.values.max():.4f} p.u.")
    print(f"  Hours with any bus below 0.95 p.u.  : "
          f"{(v_data.min(axis=1) < 0.95).sum()}")
    print(f"  Max line loading at any hour         : {l_data.values.max():.2f} %")
    print(f"  Hours any line exceeds 80%           : "
          f"{(l_data.max(axis=1) > 80).sum()}")
    print(f"{'='*50}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building IEEE 33-bus network...")
    net = build_base_network()

    print("\nLoading 24-hour residential load profile...")
    profile = get_load_profile()

    print("\nRunning hourly power flow simulation (baseline — no EVs)...")
    voltages, line_loadings = run_hourly_simulation(net, profile, label="baseline")

    print_summary(voltages, line_loadings, label="baseline")

    print("Generating plots...")
    plot_voltage_heatmap(voltages, label="baseline")
    plot_min_voltage_over_day(voltages, label="baseline")
    plot_max_line_loading(line_loadings, label="baseline")

    print("\nDone! Check the results/ folder for CSVs and plots.")
    print("Next step: run ev_load.py to add EV charging demand.")
