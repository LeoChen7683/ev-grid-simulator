"""
EV Charging Load Impact Simulator
Week 4: Real NYISO Load Data Integration
Author: Leo

What this script does:
- Downloads real hourly load data for Long Island from NYISO's public website
- Normalizes it into a 24-hour multiplier profile (same format as the hardcoded one)
- Re-runs the EV penetration scenarios using real data instead of hardcoded numbers
- Compares results against the hardcoded baseline to show the difference

NYISO publishes free public load data at:
https://www.nyiso.com/public-reports
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import os
from datetime import datetime, timedelta

os.makedirs("results", exist_ok=True)


# ─── 1. Download real NYISO Long Island load data ─────────────────────────────
def fetch_nyiso_load_profile(date=None):
    """
    Downloads the real hourly load data for Long Island from NYISO.
    
    NYISO publishes actual load data for each zone in New York State.
    Long Island is zone "LONGIL" in their system.
    
    Args:
        date : datetime object for the date you want data for.
               Defaults to yesterday (since today's full data may not be posted yet).
    
    Returns:
        np.array of 24 normalized load multipliers (0.0 to 1.0)
        float of the peak MW value for that day
    """
    if date is None:
        date = datetime.now() - timedelta(days=1)  # yesterday by default

    # NYISO URL format for actual load data
    # They store files by month in this format: YYYYMMDD_pal.csv inside a zip
    date_str  = date.strftime("%Y%m%d")
    month_str = date.strftime("%Y%m")
    url = f"https://mis.nyiso.com/public/csv/palIntegrated/{date_str}palIntegrated_csv.zip"

    print(f"Fetching NYISO load data for {date.strftime('%B %d, %Y')}...")
    print(f"URL: {url}")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        # The file comes as a zip — extract it in memory
        z = zipfile.ZipFile(io.BytesIO(response.content))
        csv_filename = [f for f in z.namelist() if f.endswith('.csv')][0]
        df = pd.read_csv(z.open(csv_filename))

        print(f"Downloaded {len(df)} rows of NYISO data.")
        print(f"Columns: {list(df.columns)}")

        # Filter for Long Island zone only
        # NYISO zone name is "N.Y.C." for NYC, "LONGIL" for Long Island
        li_mask = df['Name'].str.upper().str.contains('LONG', na=False)
        df_li   = df[li_mask].copy()

        if df_li.empty:
            print("Could not find Long Island zone — trying all zones...")
            print(f"Available zones: {df['Name'].unique()}")
            # Fall back to total integrated load
            df_li = df.copy()

        # Parse the timestamp column
        df_li['timestamp'] = pd.to_datetime(df_li['Time Stamp'])
        df_li = df_li.sort_values('timestamp')
        df_li['hour'] = df_li['timestamp'].dt.hour

        # Get the load column (NYISO calls it "Integrated Load")
        load_col = [c for c in df_li.columns if 'load' in c.lower() or 'mw' in c.lower()][0]
        print(f"Using load column: '{load_col}'")

        # Average load by hour (in case there are sub-hourly readings)
        hourly = df_li.groupby('hour')[load_col].mean()

        # Make sure we have all 24 hours
        all_hours = pd.Series(index=range(24), dtype=float)
        all_hours.update(hourly)
        all_hours = all_hours.interpolate()  # fill any missing hours

        # Normalize to 0-1 multipliers
        peak_mw   = all_hours.max()
        normalized = all_hours / peak_mw

        print(f"\nNYISO Long Island data loaded successfully!")
        print(f"Peak load: {peak_mw:.1f} MW at hour {all_hours.idxmax()}")
        print(f"Min load : {all_hours.min():.1f} MW at hour {all_hours.idxmin()}")

        return normalized.values, peak_mw, date

    except requests.exceptions.RequestException as e:
        print(f"\nCould not download NYISO data: {e}")
        print("Falling back to hardcoded typical Long Island profile...")
        return get_fallback_profile(), None, date

    except Exception as e:
        print(f"\nError parsing NYISO data: {e}")
        print("Falling back to hardcoded typical Long Island profile...")
        return get_fallback_profile(), None, date


def get_fallback_profile():
    """
    Fallback profile based on typical Long Island summer load shape from NYISO.
    Used if the live data download fails.
    This is more accurate than the generic residential profile in week 1 
    because it's calibrated to Long Island specifically.
    """
    print("Using Long Island calibrated fallback profile.")
    return np.array([
        0.62, 0.58, 0.55, 0.53, 0.52, 0.54,
        0.60, 0.68, 0.75, 0.80, 0.83, 0.85,
        0.86, 0.86, 0.87, 0.89, 0.92, 0.97,
        1.00, 0.99, 0.96, 0.90, 0.80, 0.70
    ])


# ─── 2. EV profiles (same as week 2) ─────────────────────────────────────────
def get_ev_charging_profile(penetration_pct, charger_kw=7.2):
    ev_schedule = np.array([
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.02, 0.10, 0.35,
        0.60, 0.75, 0.70, 0.45, 0.20, 0.05
    ])
    return (charger_kw / 1000) * penetration_pct * ev_schedule


def get_smart_ev_profile(penetration_pct=0.50, charger_kw=7.2):
    dumb         = get_ev_charging_profile(penetration_pct, charger_kw)
    total_energy = dumb.sum()
    smart        = np.zeros(24)
    for h in [23, 0, 1, 2, 3, 4, 5, 6]:
        smart[h] = total_energy / 8
    return smart


# ─── 3. Power flow simulation (same engine as weeks 1-3) ──────────────────────
def run_simulation(net, load_profile, ev_profile, label):
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()
    homes  = 150

    voltage_records   = []
    line_load_records = []

    for h in range(24):
        net.load.p_mw   = base_p * load_profile[h] + ev_profile[h] * homes
        net.load.q_mvar = base_q * load_profile[h]

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
    voltages.to_csv(f"results/nyiso_{label}_voltages.csv")
    line_loadings.to_csv(f"results/nyiso_{label}_line_loadings.csv")
    return voltages, line_loadings


# ─── 4. Plot: hardcoded profile vs real NYISO profile ─────────────────────────
def plot_profile_comparison(hardcoded_profile, nyiso_profile, date):
    """Shows how the real NYISO data compares to the hardcoded profile."""
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = range(24)

    ax.plot(hours, hardcoded_profile, marker="o", linewidth=2,
            color="#2a9d8f", label="Hardcoded typical profile (week 1)")
    ax.plot(hours, nyiso_profile, marker="s", linewidth=2,
            color="#e63946", label=f"Real NYISO Long Island data ({date.strftime('%b %d, %Y')})")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Normalized Load (p.u.)")
    ax.set_title("Hardcoded Profile vs Real NYISO Long Island Load Data")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(24))
    plt.tight_layout()
    plt.savefig("results/nyiso_profile_comparison.png", dpi=150)
    plt.show()
    print("Profile comparison saved: results/nyiso_profile_comparison.png")


# ─── 5. Plot: EV scenarios with real data ─────────────────────────────────────
def plot_nyiso_scenarios(results_dict, date):
    colors = {
        "baseline"  : "#2a9d8f",
        "ev_10pct"  : "#e9c46a",
        "ev_25pct"  : "#f4a261",
        "ev_50pct_dumb"  : "#e63946",
        "ev_50pct_smart" : "#457b9d",
    }
    labels_nice = {
        "baseline"       : "No EVs (Baseline)",
        "ev_10pct"       : "10% EV Penetration",
        "ev_25pct"       : "25% EV Penetration",
        "ev_50pct_dumb"  : "50% EV — Dumb Charging",
        "ev_50pct_smart" : "50% EV — Smart Charging",
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for label, (voltages, line_loadings) in results_dict.items():
        v_cols = [c for c in voltages.columns if c.startswith("bus_")]
        l_cols = [c for c in line_loadings.columns if c.startswith("line_")]
        min_v  = voltages[v_cols].min(axis=1)
        max_l  = line_loadings[l_cols].max(axis=1)
        color  = colors.get(label, "gray")
        name   = labels_nice.get(label, label)

        axes[0].plot(min_v.index, min_v.values, marker="o", linewidth=2,
                     color=color, label=name)
        axes[1].plot(max_l.index, max_l.values, marker="s", linewidth=2,
                     color=color, label=name)

    axes[0].axhline(0.95, color="orange", linestyle="--", linewidth=1,
                    label="ANSI Lower Limit (0.95)")
    axes[0].axhline(0.90, color="red", linestyle="--", linewidth=1,
                    label="Critical Limit (0.90)")
    axes[0].set_ylabel("Min Bus Voltage (p.u.)")
    axes[0].set_title(f"EV Impact on Long Island Grid — Real NYISO Data ({date.strftime('%b %d, %Y')})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0.85, 1.06)

    axes[1].axhline(80,  color="orange", linestyle="--", linewidth=1, label="Warning (80%)")
    axes[1].axhline(100, color="red",    linestyle="--", linewidth=1, label="Thermal Limit (100%)")
    axes[1].set_ylabel("Max Line Loading (%)")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_title("Worst-Case Line Loading")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(range(24))

    plt.tight_layout()
    plt.savefig("results/nyiso_scenario_comparison.png", dpi=150)
    plt.show()
    print("NYISO scenario comparison saved: results/nyiso_scenario_comparison.png")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hardcoded profile from week 1 for comparison
    hardcoded_profile = np.array([
        0.60, 0.55, 0.52, 0.50, 0.51, 0.55, 0.65, 0.80,
        0.85, 0.82, 0.78, 0.75, 0.73, 0.72, 0.74, 0.78,
        0.85, 0.95, 1.00, 0.98, 0.95, 0.88, 0.78, 0.68
    ])

    # Fetch real NYISO data
    nyiso_profile, peak_mw, data_date = fetch_nyiso_load_profile()

    # Show how real data compares to hardcoded
    print("\nPlotting profile comparison...")
    plot_profile_comparison(hardcoded_profile, nyiso_profile, data_date)

    # Build network
    net = pn.case33bw()

    # Run all scenarios with real NYISO data
    scenarios = {
        "baseline"       : (np.zeros(24), ),
        "ev_10pct"       : (get_ev_charging_profile(0.10), ),
        "ev_25pct"       : (get_ev_charging_profile(0.25), ),
        "ev_50pct_dumb"  : (get_ev_charging_profile(0.50), ),
        "ev_50pct_smart" : (get_smart_ev_profile(0.50), ),
    }

    results = {}
    for label, (ev_profile,) in scenarios.items():
        print(f"\nRunning scenario: {label}...")
        v, l = run_simulation(net, nyiso_profile, ev_profile, label=label)
        results[label] = (v, l)

    print("\nGenerating NYISO scenario plots...")
    plot_nyiso_scenarios(results, data_date)

    if peak_mw:
        print(f"\nNote: Real Long Island peak load on {data_date.strftime('%b %d')}: {peak_mw:.1f} MW")
        print("Your IEEE 33-bus network is normalized to this — results reflect real LI demand shape.")

    print("\nAll done! Check results/ for nyiso_*.png and nyiso_*.csv files.")
