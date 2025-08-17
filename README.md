Travelling Salesman Problem - Project Analysis

This project focuses on solving the Travelling Salesman Problem (TSP) using multiple algorithms and analyzing their performance on instances with up to 1000 cities.

Mandatory Results

Solver Comparison:


Gurobi consistently achieves better solution quality than CBC, with CBC failing for 500+ cities.

Average performance gap: CBC solutions are ~8.6% worse than Gurobi, with a max of ~27.3%.


Warm Start:


OR-Tools + Gurobi warm start improves distances by ~24.2% on average, with up to 26.9% improvement.

Average runtime speedup: ~74x compared to OR-Tools alone.


Clustering:


Optimal at 20 clusters, yielding up to 38% (100 cities), 42.5% (200 cities), and 50.8% (500 cities) improvement in distance.

Average clustering runtime: ~0.886s, with solutions as fast as 0.013s.


Recommendations
<50 Cities: Use pure Gurobi for optimal solutions.

50–500 Cities: Apply OR-Tools warm start with Gurobi refinement for balanced quality and speed.

>500 Cities: Opt for clustering heuristic for fast results or warm start for higher quality.
