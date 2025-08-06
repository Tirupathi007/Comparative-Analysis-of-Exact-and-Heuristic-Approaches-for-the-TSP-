import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt

# Haversine formula to calculate distance between two points (lat, lon in degrees)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Nearest-neighbor TSP solver
def nearest_neighbor_tsp(distance_matrix):
    n = len(distance_matrix)
    visited = [False] * n
    tour = [0]  # Start at node 0
    visited[0] = True
    total_distance = 0
    
    for _ in range(n-1):
        current = tour[-1]
        nearest = None
        min_dist = float('inf')
        for i in range(n):
            if not visited[i] and distance_matrix[current][i] < min_dist:
                min_dist = distance_matrix[current][i]
                nearest = i
        if nearest is not None:
            tour.append(nearest)
            visited[nearest] = True
            total_distance += min_dist
    
    # Return to start
    total_distance += distance_matrix[tour[-1]][0]
    tour.append(0)
    return tour, total_distance

# Function to solve TSP with clustering
def solve_tsp_with_clustering(coords, k):
    start_time = time.time()
    
    # Step 1: Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_
    
    # Step 2: Compute distance matrix for centroids
    n_centroids = len(centroids)
    distance_matrix = np.zeros((n_centroids, n_centroids))
    for i in range(n_centroids):
        for j in range(n_centroids):
            if i != j:
                distance_matrix[i][j] = haversine(centroids[i][0], centroids[i][1], centroids[j][0], centroids[j][1])
    
    # Step 3: Solve TSP on centroids
    _, tour_length = nearest_neighbor_tsp(distance_matrix)
    
    # Step 4: Approximate intra-cluster distances
    intra_cluster_distance = 0
    for i in range(k):
        cluster_points = coords[labels == i]
        if len(cluster_points) > 0:
            centroid = centroids[i]
            for point in cluster_points:
                intra_cluster_distance += haversine(point[0], point[1], centroid[0], centroid[1])
    
    # Total tour length
    total_tour_length = tour_length + intra_cluster_distance
    runtime = time.time() - start_time
    
    return total_tour_length, runtime

# Main script
# Read input file
input_file = "input_files/tsp_locations_1000.csv"
data = pd.read_csv(input_file)
coords_full = data[["Latitude", "Longitude"]].values

# Problem sizes and clustering sizes
problem_sizes = [100, 200, 500]
cluster_sizes = [5, 10, 15, 20]

# Results storage
results = []

# Solve for each problem size and clustering size
for n in problem_sizes:
    # Select first n cities
    coords = coords_full[:n]
    for k in cluster_sizes:
        if k < n:  # Ensure number of clusters is less than number of cities
            tour_length, runtime = solve_tsp_with_clustering(coords, k)
            results.append({
                "Problem Size": n,
                "Clusters": k,
                "Tour Length (km)": tour_length,
                "Runtime (s)": runtime
            })

# Save results to CSV
output_file = "output_files/solver_comparison.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Generate plot
plt.figure(figsize=(10, 6))
for n in problem_sizes:
    n_results = [r for r in results if r["Problem Size"] == n]
    clusters = [r["Clusters"] for r in n_results]
    tour_lengths = [r["Tour Length (km)"] for r in n_results]
    plt.plot(clusters, tour_lengths, marker='o', label=f'Problem Size {n}')

plt.xlabel("Number of Clusters")
plt.ylabel("Tour Length (km)")
plt.title("Tour Length vs. Number of Clusters")
plt.legend()
plt.grid(True)
plot_file = "output_files/tsp_results_plot.png"
plt.savefig(plot_file)
plt.close()
print(f"Plot saved to {plot_file}")