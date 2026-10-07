import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
NUM_PUMPS = 100
INTERVAL_SECONDS = 5
DAY_SECONDS = 24 * 60 * 60
ROWS_PER_PUMP = DAY_SECONDS // INTERVAL_SECONDS

OUTPUT_FILE = "simulation.csv"

# ------------------------------------------------------------
# BASELINE METRICS
# ------------------------------------------------------------
def generate_baseline_metrics():
    return {
        "pressure": np.random.normal(250, 5),
        "flow_rate": np.random.normal(80, 2),
        "temperature": np.random.normal(65, 1),
        "vibration": np.random.normal(2.0, 0.2),
        "power_draw": np.random.normal(15, 0.5)
    }

# ------------------------------------------------------------
# DEGRADATION MODEL
# ------------------------------------------------------------
def apply_degradation(metrics, factor):
    return {
        "pressure": metrics["pressure"] - (40 * factor),
        "flow_rate": metrics["flow_rate"] - (20 * factor),
        "temperature": metrics["temperature"] + (20 * factor),
        "vibration": metrics["vibration"] + (4 * factor),
        "power_draw": metrics["power_draw"] + (5 * factor)
    }

# ------------------------------------------------------------
# RECOVERY MODEL
# ------------------------------------------------------------
def apply_recovery(metrics, factor):
    # factor goes 1 → 0 (reverse degradation)
    return {
        "pressure": metrics["pressure"] + (40 * factor),
        "flow_rate": metrics["flow_rate"] + (20 * factor),
        "temperature": metrics["temperature"] - (20 * factor),
        "vibration": metrics["vibration"] - (4 * factor),
        "power_draw": metrics["power_draw"] - (5 * factor)
    }

# ------------------------------------------------------------
# GENERATE DATA
# ------------------------------------------------------------
def simulate_pump_data():
    pumps = [f"PMP{str(i+1).zfill(3)}" for i in range(NUM_PUMPS)]

    # Select 20% of pumps to degrade
    degrading_pumps = set(random.sample(pumps, int(NUM_PUMPS * 0.2)))

    # Assign staggered degradation start times
    degradation_start = {
        pump: random.randint(0, ROWS_PER_PUMP - 1)
        for pump in degrading_pumps
    }

    # Assign recovery times for half of degrading pumps
    recovering_pumps = set(random.sample(list(degrading_pumps), int(len(degrading_pumps) * 0.5)))

    recovery_start = {}
    for pump in recovering_pumps:
        start_i = degradation_start[pump]
        # Recovery begins sometime AFTER degradation starts
        recovery_start[pump] = random.randint(start_i + 2000, ROWS_PER_PUMP - 1)

    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    all_rows = []

    for pump in pumps:
        for i in range(ROWS_PER_PUMP):
            timestamp = start_time + timedelta(seconds=i * INTERVAL_SECONDS)
            base = generate_baseline_metrics()

            # Default state
            status = "NORMAL"
            metrics = base

            # Pump degrades?
            if pump in degrading_pumps:
                start_i = degradation_start[pump]

                if i >= start_i:
                    # If pump is recovering
                    if pump in recovering_pumps and i >= recovery_start[pump]:
                        # Recovery factor goes from 1 → 0
                        total_recovery = ROWS_PER_PUMP - recovery_start[pump]
                        factor = 1 - ((i - recovery_start[pump]) / total_recovery)
                        factor = max(0, min(1, factor))
                        metrics = apply_recovery(base, factor)
                        status = "RECOVERING"
                    else:
                        # Degradation factor goes 0 → 1
                        factor = (i - start_i) / (ROWS_PER_PUMP - start_i)
                        factor = max(0, min(1, factor))
                        metrics = apply_degradation(base, factor)
                        status = "DEGRADING"

            row = {
                "timestamp": timestamp,
                "pump_id": pump,
                "pressure_psi": round(metrics["pressure"], 2),
                "flow_rate_ls": round(metrics["flow_rate"], 2),
                "temperature_c": round(metrics["temperature"], 2),
                "vibration_mms": round(metrics["vibration"], 2),
                "power_kw": round(metrics["power_draw"], 2),
                "status": status
            }

            all_rows.append(row)

    return pd.DataFrame(all_rows)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Generating simulation with staggered degradation + recovery...")
    df = simulate_pump_data()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Simulation complete. File saved as: {OUTPUT_FILE}")
