import heapq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

def dijkstra(graph, start):
    """Dijkstra's Algorithm for shortest paths from a single source."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}
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
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors

def reconstruct_path(predecessors, start, goal):
    """Reconstruct the path from start to goal using predecessor information."""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    return path if path[0] == start else []

def heuristic(node, goal):
    """Heuristic function for A* (Manhattan distance for grids)."""
    if isinstance(node, tuple) and isinstance(goal, tuple):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    return 0  # For non-grid graphs, heuristic is zero

def a_star(graph, start, goal, grid=False):
    """A* Search Algorithm for shortest path from start to goal."""
    open_set = [(0, start)]
    heapq.heapify(open_set)
    came_from = {start: None}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal) if grid else 0

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, start, goal), g_score[goal]

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

    return [], float('inf')

def bellman_ford(graph, start):
    """Bellman-Ford Algorithm for shortest paths from a single source."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}

    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node].items():
                if isinstance(weight, dict):
                    weight = weight.get('weight', float('inf'))
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    predecessors[neighbor] = node

    for node in graph:
        for neighbor, weight in graph[node].items():
            if isinstance(weight, dict):
                weight = weight.get('weight', float('inf'))
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Graph contains a negative-weight cycle")

    return distances, predecessors

def floyd_warshall(graph):
    """Floyd-Warshall Algorithm for shortest paths between all pairs of nodes."""
    nodes = list(graph.keys())
    dist = {node: {node2: float('inf') for node2 in nodes} for node in nodes}
    next_node = {node: {node2: None for node2 in nodes} for node in nodes}

    for node in nodes:
        dist[node][node] = 0
        for neighbor, weight in graph[node].items():
            if isinstance(weight, dict):
                weight = weight.get('weight', float('inf'))
            dist[node][neighbor] = weight
            next_node[node][neighbor] = neighbor

    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist, next_node

def reconstruct_fw_path(next_node, start, goal):
    """Reconstruct the path using the Floyd-Warshall next node information."""
    if next_node[start][goal] is None:
        return []
    path = [start]
    while start != goal:
        start = next_node[start][goal]
        path.append(start)
    return path

def run_experiment(graph, start_node, goal_node, grid=False):
    """Run all algorithms and measure their execution time."""
    results = {}
    times = {}
    
    # Dijkstra's Algorithm
    start_time = time.time()
    distances, predecessors = dijkstra(graph, start_node)
    path = reconstruct_path(predecessors, start_node, goal_node)
    times['Dijkstra'] = time.time() - start_time
    results['Dijkstra'] = (path, distances[goal_node])

    # A* Algorithm
    start_time = time.time()
    path, cost = a_star(graph, start_node, goal_node, grid)
    times['A*'] = time.time() - start_time
    results['A*'] = (path, cost)

    # Bellman-Ford Algorithm
    start_time = time.time()
    try:
        distances, predecessors = bellman_ford(graph, start_node)
        path = reconstruct_path(predecessors, start_node, goal_node)
        times['Bellman-Ford'] = time.time() - start_time
        results['Bellman-Ford'] = (path, distances[goal_node])
    except ValueError as e:
        print(e)
        times['Bellman-Ford'] = float('inf')  # Infinite time if negative cycle
        results['Bellman-Ford'] = ([], float('inf'))

    # Floyd-Warshall Algorithm
    start_time = time.time()
    dist, next_node = floyd_warshall(graph)
    path = reconstruct_fw_path(next_node, start_node, goal_node)
    times['Floyd-Warshall'] = time.time() - start_time
    results['Floyd-Warshall'] = (path, dist[start_node][goal_node])

    return results, times

def create_random_graph(num_nodes, num_edges, weight_range=(1, 50)):
    """Create a random directed graph with a given number of nodes and edges."""
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(*weight_range)
    return nx.to_dict_of_dicts(G)

def create_grid_graph(size, weight_range=(1, 50)):
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
times = {}

# Dense graph
print("Test cases for dense graph")
dense_graph = create_random_graph(30, 100)  # 30 nodes, 100 edges
results['Dense Graph'], times['Dense Graph'] = run_experiment(dense_graph, 0, 29)
for alg, (path, cost) in results['Dense Graph'].items():
    print(f"{alg} on Dense Graph found path {path} with cost {cost:.6f} in {times['Dense Graph'][alg]:.6f} seconds")

# Grid graph
print("\nTest cases for grid graph")
grid_graph = create_grid_graph(8)  # 8x8 grid
results['Grid Graph'], times['Grid Graph'] = run_experiment(grid_graph, (0, 0), (7, 7), grid=True)
for alg, (path, cost) in results['Grid Graph'].items():
    print(f"{alg} on Grid Graph found path {path} with cost {cost:.6f} in {times['Grid Graph'][alg]:.6f} seconds")

# Simple negative weight graph
print("\nTest cases for graph with simple negative weights")
negative_graph = create_simple_negative_weight_graph()
results['Negative Weights'], times['Negative Weights'] = run_experiment(negative_graph, 0, 5)
for alg, (path, cost) in results['Negative Weights'].items():
    print(f"{alg} on Negative Weights Graph found path {path} with cost {cost:.6f} in {times['Negative Weights'][alg]:.6f} seconds")

# Complete graph
print("\nTest cases for complete graph")
complete_graph = create_random_graph(10, 45)  # Complete graph with 10 nodes, more edges
results['Complete Graph'], times['Complete Graph'] = run_experiment(complete_graph, 0, 9)
for alg, (path, cost) in results['Complete Graph'].items():
    print(f"{alg} on Complete Graph found path {path} with cost {cost:.6f} in {times['Complete Graph'][alg]:.6f} seconds")

plot_results(times)
