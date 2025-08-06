from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import logging
import os
import sys
import traceback
from datetime import datetime

# Configuration
BASE_LIMIT = 600
LARGE_LIMIT = 1200
FOCUS_SIZES = [100, 200, 500]
ALL_SIZES = [10, 20, 30, 40, 50, 70, 100, 200, 500]  # Removed 750 and 1000

# Paths
ROOT = Path(__file__).parent
INPUT_CSV = ROOT / 'input_files' / 'tsp_locations_1000.csv'
OUT_DIR = ROOT / 'output_files'
PLOT_DIR = OUT_DIR / 'plots'
CSV_FILE = OUT_DIR / 'results.csv'
COMPARISON_FILE = OUT_DIR / 'comparison.csv'
LOG_DIR = ROOT / 'logs'

# Create directories
for p in (OUT_DIR, PLOT_DIR, LOG_DIR):
    p.mkdir(exist_ok=True)

# Logging setup
log_file = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_tsp.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info("Working directory: %s", Path.cwd())

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * 6371 * math.asin(math.sqrt(a))

def dist_matrix(df, n):
    sub = df.head(n).reset_index(drop=True)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = haversine(
                sub.at[i, 'Latitude'], sub.at[i, 'Longitude'],
                sub.at[j, 'Latitude'], sub.at[j, 'Longitude']
            )
    return D, sub

def extract_gap(results):
    for attr in ('mip_gap', 'gap'):
        if hasattr(results.solver, attr):
            v = getattr(results.solver, attr)
            if v is not None:
                return v
    return getattr(results.problem, 'MIPGap', None)

def solve_tsp(D, limit, mip_focus=1):
    n = len(D)
    m = pyo.ConcreteModel()
    m.N = pyo.RangeSet(0, n - 1)
    m.d = pyo.Param(m.N, m.N, initialize=lambda _, i, j: float(D[i][j]))
    m.x = pyo.Var(m.N, m.N, within=pyo.Binary)
    m.u = pyo.Var(m.N, bounds=(0, n - 1))

    m.obj = pyo.Objective(expr=sum(m.d[i, j] * m.x[i, j] for i in m.N for j in m.N if i != j))
    m.out = pyo.Constraint(m.N, rule=lambda m, i: sum(m.x[i, j] for j in m.N if j != i) == 1)
    m._in = pyo.Constraint(m.N, rule=lambda m, j: sum(m.x[i, j] for i in m.N if i != j) == 1)

    m.mtz = pyo.ConstraintList()
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.mtz.add(m.u[i] - m.u[j] + n * m.x[i, j] <= n - 1)

    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = limit
    solver.options['MIPGap'] = 0.0001
    solver.options['MIPFocus'] = mip_focus

    start = datetime.now()
    try:
        res = solver.solve(m, tee=False)
    except Exception as e:
        logger.error("Gurobi crashed: %s", e)
        logger.debug(traceback.format_exc())
        return None

    runtime = (datetime.now() - start).total_seconds()
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j and pyo.value(m.x[i, j]) > 0.5]
    dist = sum(D[i][j] for i, j in arcs)
    gap = extract_gap(res)
    status = res.solver.termination_condition.name

    return {
        'distance': dist,
        'runtime': runtime,
        'mip_gap': gap,
        'status': status,
        'tour': arcs
    }

def plot_route(sub, tour, n):
    if not tour:
        return

    plt.figure(figsize=(10, 8))
    plt.scatter(sub['Longitude'], sub['Latitude'], c='red', s=50)

    for idx, row in sub.iterrows():
        plt.annotate(idx, (row['Longitude'], row['Latitude']),
                     textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8)

    for i, j in tour:
        plt.plot(
            [sub.at[i, 'Longitude'], sub.at[j, 'Longitude']],
            [sub.at[i, 'Latitude'], sub.at[j, 'Latitude']],
            'b-', alpha=0.5
        )

    title = f'TSP Solution: {n} Cities (Gurobi, MIPFocus=1)'
    plt.title(title)
    fname = PLOT_DIR / f'tour_gurobi_{n}_focus1.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info('Saved plot: %s', fname)

def run_comparison(df):
    comparison = []

    for n in FOCUS_SIZES:
        logger.info("\n=== Running Gurobi MIPFocus=1 for %d cities ===", n)
        D, sub = dist_matrix(df, n)
        limit = LARGE_LIMIT if n >= 500 else BASE_LIMIT

        res = solve_tsp(D, limit, mip_focus=1)
        if res:
            plot_route(sub, res['tour'], n)
            comparison.append({
                'Num_cities': n,
                'Strategy': 'MIPFocus=1',
                'Distance': res['distance'],
                'Runtime': res['runtime'],
                'MIPGap': res['mip_gap'],
                'Status': res['status']
            })

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(COMPARISON_FILE, index=False)
    logger.info("Saved comparison results to %s", COMPARISON_FILE)

    print("\n=== Gurobi MIPFocus=1 COMPARISON RESULTS ===")
    print(comp_df.to_string(index=False))
    return comp_df

def main():
    try:
        df = pd.read_csv(INPUT_CSV)
        logger.info("Loaded input data with %d cities", len(df))
    except Exception as e:
        logger.error('Failed to load input CSV: %s', e)
        return

    results = []
    time_limit = lambda n: LARGE_LIMIT if n >= 500 else BASE_LIMIT

    for n in ALL_SIZES:
        logger.info("\nSolving TSP for %d cities with Gurobi...", n)
        D, sub = dist_matrix(df, n)
        limit = time_limit(n)

        res = solve_tsp(D, limit, mip_focus=1)
        if res:
            plot_route(sub, res['tour'], n)
            results.append({
                'Num_cities': n,
                'Distance': res['distance'],
                'Runtime': res['runtime'],
                'MIPGap': res['mip_gap'],
                'Status': res['status']
            })

        pd.DataFrame(results).to_csv(CSV_FILE, index=False)

    final_results = pd.DataFrame(results)
    final_results.to_csv(CSV_FILE, index=False)
    logger.info("Saved main results to %s", CSV_FILE)

    print("\n=== MAIN RESULTS (Gurobi MIPFocus=1) ===")
    print(final_results.to_string(index=False))

    run_comparison(df)
    logger.info("All tasks completed")

if __name__ == '__main__':
    main()
