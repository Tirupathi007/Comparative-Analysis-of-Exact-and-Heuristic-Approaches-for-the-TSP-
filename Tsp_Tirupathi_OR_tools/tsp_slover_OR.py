from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration - reduced problem sizes to prevent overflow
CITY_SIZES = [50, 100,200,500,1000]  # Reduced from [50, 100, 200]
TIME_LIMIT = 600
WARM_START_ITERATIONS = 1000

# File paths
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / 'input_files' / 'tsp_locations_1000.csv'
OUTPUT_DIR = BASE_DIR / 'output_files'
OUTPUT_DIR.mkdir(exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two geographic points"""
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def create_distance_matrix(df, n_cities):
    """Create N x N distance matrix with scaling"""
    cities = df.head(n_cities)
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # Scale distances to prevent overflow
            dist = haversine(
                cities.iloc[i]['Latitude'], cities.iloc[i]['Longitude'],
                cities.iloc[j]['Latitude'], cities.iloc[j]['Longitude']
            )
            dist_matrix[i, j] = dist_matrix[j, i] = int(dist * 1000)  # Scale to integers
    return dist_matrix, cities

def solve_with_ortools(dist_matrix, use_warm_start=True):
    """Solve TSP using OR-Tools with error handling"""
    try:
        num_locations = len(dist_matrix)
        manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Scaled distance callback"""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(dist_matrix[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = TIME_LIMIT
        
        if use_warm_start:
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.solution_limit = WARM_START_ITERATIONS

        start_time = time.time()
        solution = routing.SolveWithParameters(search_parameters)
        runtime = time.time() - start_time

        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))

            # Convert back to original distance scale
            total_distance = sum(dist_matrix[route[i]][route[i+1]]/1000 
                               for i in range(len(route)-1))

            return {
                'distance': total_distance,
                'runtime': runtime,
                'route': route,
                'status': 'Optimal' if solution.ObjectiveValue()/1000 == total_distance else 'Heuristic'
            }
        return None
    except Exception as e:
        print(f"OR-Tools failed with error: {str(e)}")
        return None

def solve_with_gurobi(dist_matrix, initial_solution=None):
    """Solve TSP using Gurobi with error handling"""
    try:
        n = len(dist_matrix)
        model = pyo.ConcreteModel()
        model.N = pyo.RangeSet(0, n-1)
        
        # Model components (using scaled distances)
        model.d = pyo.Param(model.N, model.N, 
                           initialize=lambda m, i, j: dist_matrix[i][j])
        model.x = pyo.Var(model.N, model.N, within=pyo.Binary)
        
        # Objective and constraints
        model.obj = pyo.Objective(
            expr=sum(model.d[i,j] * model.x[i,j] for i in model.N for j in model.N if i != j),
            sense=pyo.minimize
        )
        model.out_degree = pyo.Constraint(
            model.N, rule=lambda m, i: sum(m.x[i,j] for j in model.N if j != i) == 1
        )
        model.in_degree = pyo.Constraint(
            model.N, rule=lambda m, j: sum(m.x[i,j] for i in model.N if i != j) == 1
        )
        
        # Warm start if provided
        if initial_solution:
            for i, j in initial_solution:
                model.x[i,j].value = 1
        
        # Solver configuration
        solver = SolverFactory('gurobi')
        solver.options = {
            'TimeLimit': TIME_LIMIT,
            'MIPGap': 0.0001,
            'MIPFocus': 1,
            'NumericFocus': 1  # Helps with numerical stability
        }
        
        # Solve
        start_time = time.time()
        results = solver.solve(model, tee=True)
        runtime = time.time() - start_time
        
        # Extract solution
        solution = [(i,j) for i in range(n) for j in range(n) 
                   if i != j and pyo.value(model.x[i,j]) > 0.5]
        total_distance = sum(dist_matrix[i][j]/1000 for i,j in solution)  # Convert back
        
        return {
            'distance': total_distance,
            'runtime': runtime,
            'solution': solution,
            'status': str(results.solver.termination_condition)
        }
    except Exception as e:
        print(f"Gurobi failed with error: {str(e)}")
        return None

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded data with {len(df)} cities")
    except Exception as e:
        print(f"Failed to load input file: {str(e)}")
        return
    
    results = []
    
    for n in CITY_SIZES:
        print(f"\n=== Processing {n} cities ===")
        dist_matrix, cities = create_distance_matrix(df, n)
        
        # 1. Solve with OR-Tools
        print("Running OR-Tools...")
        ortools_result = solve_with_ortools(dist_matrix, use_warm_start=True)
        
        if ortools_result is None:
            print("OR-Tools failed to find solution")
            continue
            
        # 2. Solve with Gurobi using OR-Tools solution as warm start
        print("Running Gurobi with warm start...")
        warm_start = [(ortools_result['route'][i], ortools_result['route'][i+1]) 
                     for i in range(len(ortools_result['route'])-1)]
        gurobi_result = solve_with_gurobi(dist_matrix, warm_start)
        
        if gurobi_result is None:
            print("Gurobi failed to find solution")
            continue
            
        # Store results
        results.append({
            'Cities': n,
            'OR-Tools Distance': ortools_result['distance'],
            'OR-Tools Runtime': ortools_result['runtime'],
            'OR-Tools Status': ortools_result['status'],
            'Gurobi Distance': gurobi_result['distance'],
            'Gurobi Runtime': gurobi_result['runtime'],
            'Gurobi Status': gurobi_result['status'],
            'Improvement %': round(100*(ortools_result['distance']-gurobi_result['distance'])/ortools_result['distance'], 2) 
                          if ortools_result['distance'] > 0 else 0
        })
        
        # Save intermediate results
        pd.DataFrame(results).to_csv(OUTPUT_DIR / 'solver_comparison.csv', index=False)
    
    # Final output
    if results:
        results_df = pd.DataFrame(results)
        results_file = OUTPUT_DIR / 'solver_comparison.csv'
        results_df.to_csv(results_file, index=False)
        
        print("\n=== Final Results ===")
        print(results_df.to_string(index=False))
        print(f"\nResults saved to {results_file}")
    else:
        print("No successful results to report")

if __name__ == '__main__':
    main()