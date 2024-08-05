import heapq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

def dijkstra(graph, start):
    """Dijkstra's Algorithm for shortest paths from a single source."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            if isinstance(weight, dict):
                weight = weight.get('weight', float('inf'))  # Ensure correct weight extraction
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def heuristic(node, goal):
    """Heuristic function for A* (Manhattan distance for grids)."""
    if isinstance(node, tuple) and isinstance(goal, tuple):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    return 0  # For non-grid graphs, heuristic is zero

def a_star(graph, start, goal, grid=False):
    """A* Search Algorithm for shortest path from start to goal."""
    open_set = [(0, start)]
    heapq.heapify(open_set)
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal) if grid else 0

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, weight in graph[current].items():
            if isinstance(weight, dict):
                weight = weight.get('weight', float('inf'))
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal) if grid else 0
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def bellman_ford(graph, start):
    """Bellman-Ford Algorithm for shortest paths from a single source."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node].items():
                if isinstance(weight, dict):
                    weight = weight.get('weight', float('inf'))
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight

    for node in graph:
        for neighbor, weight in graph[node].items():
            if isinstance(weight, dict):
                weight = weight.get('weight', float('inf'))
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Graph contains a negative-weight cycle")

    return distances

def floyd_warshall(graph):
    """Floyd-Warshall Algorithm for shortest paths between all pairs of nodes."""
    nodes = list(graph.keys())
    dist = {node: {node2: float('inf') for node2 in nodes} for node in nodes}

    for node in nodes:
        dist[node][node] = 0
        for neighbor, weight in graph[node].items():
            if isinstance(weight, dict):
                weight = weight.get('weight', float('inf'))
            dist[node][neighbor] = weight

    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist

def run_experiment(graph, start_node, goal_node, grid=False):
    """Run all algorithms and measure their execution time."""
    times = {}
    
    # Dijkstra's Algorithm
    start_time = time.time()
    dijkstra_result = dijkstra(graph, start_node)
    times['Dijkstra'] = time.time() - start_time

    # A* Algorithm
    start_time = time.time()
    a_star_result = a_star(graph, start_node, goal_node, grid)
    times['A*'] = time.time() - start_time

    # Bellman-Ford Algorithm
    start_time = time.time()
    try:
        bellman_ford_result = bellman_ford(graph, start_node)
        times['Bellman-Ford'] = time.time() - start_time
    except ValueError as e:
        print(e)
        times['Bellman-Ford'] = float('inf')  # Infinite time if negative cycle

    # Floyd-Warshall Algorithm
    start_time = time.time()
    floyd_warshall_result = floyd_warshall(graph)
    times['Floyd-Warshall'] = time.time() - start_time

    return times

def create_random_graph(num_nodes, num_edges, weight_range=(1, 20)):
    """Create a random directed graph with a given number of nodes and edges."""
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(*weight_range)
    return nx.to_dict_of_dicts(G)

def create_grid_graph(size, weight_range=(1, 20)):
    """Create a grid graph for A* testing."""
    G = nx.grid_2d_graph(size, size)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.randint(*weight_range)
    return nx.to_dict_of_dicts(G)

def create_simple_negative_weight_graph():
    """Create a simple graph with negative weights for Bellman-Ford testing."""
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        (0, 1, 1), (1, 2, -2),
        (2, 3, 1), (3, 4, 2),
        (4, 5, -1), (5, 0, 3)
    ])
    return nx.to_dict_of_dicts(G)

def plot_results(results):
    """Plot the execution times of the algorithms."""
    for graph_name, times in results.items():
        labels, data = zip(*times.items())
        plt.figure(figsize=(10, 5))
        plt.bar(labels, data, color=['red', 'green', 'blue', 'purple'])
        plt.title(f'Execution Time for {graph_name}')
        plt.xlabel('Algorithm')
        plt.ylabel('Time (seconds)')
        plt.show()

# Main experiment with different scenarios
results = {}

# Dense graph
print("Test cases for dense graph")
dense_graph = create_random_graph(30, 100)  
results['Dense Graph'] = run_experiment(dense_graph, 0, 29)
for alg, time_taken in results['Dense Graph'].items():
    print(f"{alg} on Dense Graph took {time_taken:.6f} seconds")

# Grid graph
print("\nTest cases for grid graph")
grid_graph = create_grid_graph(8)  
results['Grid Graph'] = run_experiment(grid_graph, (0, 0), (7, 7), grid=True)
for alg, time_taken in results['Grid Graph'].items():
    print(f"{alg} on Grid Graph took {time_taken:.6f} seconds")

# Simple negative weight graph
print("\nTest cases for graph with simple negative weights")
negative_graph = create_simple_negative_weight_graph()
results['Negative Weights'] = run_experiment(negative_graph, 0, 5)
for alg, time_taken in results['Negative Weights'].items():
    print(f"{alg} on Negative Weights Graph took {time_taken:.6f} seconds")

# Complete graph
print("\nTest cases for complete graph")
complete_graph = create_random_graph(10, 45)  
results['Complete Graph'] = run_experiment(complete_graph, 0, 9)
for alg, time_taken in results['Complete Graph'].items():
    print(f"{alg} on Complete Graph took {time_taken:.6f} seconds")

plot_results(results)
