# EV Charging Load Impact Simulator
### A Distribution Grid Power Flow Analysis | Long Island Case Study

**Author:** Leo  
**Affiliation:** Stony Brook University — Electrical Engineering (Power & Energy Systems)  
**Tools:** Python · pandapower · pandas · NumPy · Matplotlib

---

## Overview

This project simulates the impact of increasing electric vehicle (EV) adoption on a residential distribution feeder, modeled after Long Island's infrastructure. Using the standard IEEE 33-bus test network and real-world load data patterns, the simulation quantifies voltage violations and thermal overloads under different EV penetration scenarios, and evaluates smart charging as a mitigation strategy.

This is the kind of analysis actively being conducted by utilities like **LIPA** and grid operators like **NYISO** as EV adoption accelerates across New York State.

---

## What This Project Covers

| Phase | Script | Description |
|---|---|---|
| 1 | `week1_base_grid.py` | Build IEEE 33-bus network, run 24-hour baseline power flow |
| 2 | `week2_ev_load.py` | Model EV demand at 10%, 25%, 50% penetration — identify violations |
| 3 | `week3_smart_charging.py` | Implement valley-fill smart charging, compare to dumb charging |

---

## Key Findings

- At **50% EV penetration** with unmanaged (dumb) charging, the feeder experiences voltage violations and line overloads during the evening peak (6–9 PM)
- **Smart charging** (shifting demand to overnight off-peak hours) eliminates voltage violations and reduces peak line loading — with **zero reduction in EV owner convenience**
- Results align with the core argument behind demand response programs and time-of-use pricing

*(See `results/` folder for plots and CSV data)*

---

## How to Run

### 1. Install dependencies
```bash
pip install pandapower pandas numpy matplotlib jupyter
```

### 2. Run in order
```bash
python week1_base_grid.py      # Baseline — no EVs
python week2_ev_load.py        # EV penetration scenarios
python week3_smart_charging.py # Dumb vs smart charging comparison
```

All plots and CSV files are saved to the `results/` folder automatically.

---

## Network: IEEE 33-Bus Distribution Feeder

The IEEE 33-bus radial distribution test system is a standard benchmark widely used in power systems research. It represents a realistic medium-voltage residential/commercial feeder.

- **33 buses**, **32 branches**
- Radial topology (single source at bus 0)
- Total base load: ~3.7 MW, ~2.3 MVAr
- Built into pandapower as `pandapower.networks.case33bw()`

---

## EV Modeling Assumptions

| Parameter | Value | Source |
|---|---|---|
| Charger type | Level 2 (7.2 kW) | DOE EV Handbook |
| Homes per bus | 150 | Long Island residential feeder estimate |
| Dumb charging window | 5 PM – 11 PM | Commuter plug-in behavior (DOE) |
| Smart charging window | 11 PM – 6 AM | Off-peak valley-fill |
| Voltage lower limit | 0.95 p.u. | ANSI C84.1 Standard |
| Line warning threshold | 80% | Typical utility operating limit |

---

## Results Preview

| Scenario | Min Voltage (p.u.) | Voltage Violations | Max Line Loading |
|---|---|---|---|
| Baseline (no EVs) | ~1.00 | 0 hours | ~35% |
| 10% EV penetration | ~0.98 | 0 hours | ~40% |
| 25% EV penetration | ~0.96 | Low | ~55% |
| 50% EV (dumb) | **<0.95** | Multiple hours | **>80%** |
| 50% EV (smart) | ~0.97 | 0 hours | ~50% |

---

## Skills Demonstrated

- Power flow analysis (Newton-Raphson method)
- Distribution system modeling
- Load profile construction from real-world data patterns
- EV demand modeling and scenario analysis
- Optimization-based demand response (valley-fill algorithm)
- Data visualization and engineering reporting

---

## References

- IEEE 33-bus test system: Baran & Wu (1989)
- EV charging behavior: U.S. DOE Vehicle Technologies Office
- Long Island load data: NYISO Public Reports (nyiso.com/public-reports)
- ANSI voltage standards: ANSI C84.1-2020
- pandapower: Thurner et al. (2018), IEEE Trans. Power Systems
