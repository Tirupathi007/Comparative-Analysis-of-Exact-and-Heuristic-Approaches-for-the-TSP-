Travelling Salesman Problem - Project Analysis
This project focuses on solving the Travelling Salesman Problem (TSP) using multiple algorithms and analyzing their performance on instances with up to 1000 cities.

Project Structure

Tsp_ChinnalaTirupathi_2/
│
├── Tsp_Tirupathi_solvers/          # Custom heuristic and exact solvers
├── Tsp_Tirupathi_mipfocus/         # Solvers with different MIP focus settings
├── Tsp_Tirupathi_clustering/       # Clustering-based TSP approach
├── Tsp_Tirupathi_OR_tools/         # Google OR-Tools implementation
├── Travelling-Sales-man-problem-for-1000-cities/  # Full-scale TSP instance
│
├── TSP_analysis.ipynb              # Main notebook with analysis and plots
├── comparision.csv                 # Performance comparison of solvers
├── README.md                       # This file

Problem Definition
Given a list of cities and the distances between each pair, the goal is to find the shortest possible route that visits each city exactly once and returns to the origin city.

✅ Objectives
Implement and evaluate various TSP solvers.

Analyze solution time, accuracy, and scalability.

Visualize TSP paths and performance metrics.

Apply clustering techniques to simplify large TSP instances.

⚙️ Methods and Solvers Used
🔢 Heuristic Solvers (e.g., Nearest Neighbor, 2-opt)

🧮 MIP Models (Pyomo + Gurobi/GLPK with MIPFocus tuning)

🧠 Clustering + Local Solvers (Decomposition for large problems)

⚙️ Google OR-Tools (Routing Solver with metaheuristics)

📊 Analysis Highlights
All solvers tested on benchmark instances up to 1000 cities.

Clustering approaches greatly reduce computation time.

Google OR-Tools provided good balance between speed and quality.

Trade-offs analyzed using comparision.csv and visualized in TSP_analysis.ipynb.

📦 Dependencies
Python ≥ 3.8

numpy, matplotlib, pandas, scikit-learn

pyomo, gurobipy or glpk

ortools

Jupyter Notebook
