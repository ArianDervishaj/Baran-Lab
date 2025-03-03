import networkx as nx
import numpy as np


def simulate_failures(G, failure_percentage, failure_type="node"):
    """
    Simulate network failures by removing nodes or links.

    Args:
        G: NetworkX graph
        failure_percentage: Percentage to remove (0.0 to 1.0)
        failure_type: Type of failure ("node" or "link")

    Returns:
        Copy of G with elements removed
    """
    G_failed = G.copy()

    if failure_type == "node":
        num_elements = len(G.nodes())
        elements_to_remove = int(failure_percentage * num_elements)

        if elements_to_remove > 0:
            nodes_to_remove = np.random.choice(
                list(G.nodes()), size=elements_to_remove, replace=False
            )
            G_failed.remove_nodes_from(nodes_to_remove)

    elif failure_type == "link":
        num_elements = len(G.edges())
        elements_to_remove = int(failure_percentage * num_elements)

        if elements_to_remove > 0:
            all_links = list(G.edges())
            link_indices = np.random.choice(
                range(len(all_links)), size=elements_to_remove, replace=False
            )
            links_to_remove = [all_links[i] for i in link_indices]
            G_failed.remove_edges_from(links_to_remove)

    return G_failed


def calculate_survival_level(G_original, G_failed):
    """
    Calculate network survival level as percentage of nodes in largest connected component.

    Args:
        G_original: Original NetworkX graph before failures
        G_failed: NetworkX graph after failures

    Returns:
        Survival level as percentage (0-100)
    """
    if not G_original.nodes():
        return 0

    if not G_failed.nodes():
        return 0

    largest_cc = max(nx.connected_components(G_failed), key=len)
    return len(largest_cc) / len(G_original.nodes()) * 100


def run_simulation(
    graph_generator, size, failure_probs, failure_type="node", num_trials=15
):
    """
    Run simulation across failure probabilities and return mean and std deviation.

    Args:
        graph_generator: Function that generates the network topology
        size: Size parameter for the graph
        failure_probs: List of failure probabilities to test
        failure_type: Type of failure to simulate ("node" or "link")
        num_trials: Number of trials to run for each probability

    Returns:
        means: List of mean survival rates for each probability
        stds: List of standard deviations for each probability
    """
    results = []

    for prob in failure_probs:
        trial_results = []

        for _ in range(num_trials):
            G = graph_generator(size)
            G_failed = simulate_failures(G, prob, failure_type)
            survival_level = calculate_survival_level(G, G_failed)
            trial_results.append(survival_level)

        results.append((np.mean(trial_results), np.std(trial_results)))

    means, stds = zip(*results)
    return list(means), list(stds)


def run_failure_simulations(topologies, size, failure_probs, failure_type):
    """
    Run simulations for all topologies with specified failure type.

    Args:
        topologies: List of topology dictionaries with keys: generator, label, format
        size: Size parameter for the graph
        failure_probs: List of failure probabilities to test
        failure_type: Type of failure to simulate ("node" or "link")

    Returns:
        List of tuples (means, stds, label, format) for each topology
    """
    failure_name = "node" if failure_type == "node" else "link"
    print(f"Running {failure_name} failure simulations...")
    results = []

    for topo in topologies:
        print(f"Running {failure_name} failure simulation for {topo['label']}...")
        means, stds = run_simulation(
            topo["generator"], size, failure_probs, failure_type
        )

        # Use dashed lines for link failures to distinguish from node failures
        format_spec = topo["format"]
        if failure_type == "link":
            format_spec = format_spec.replace("-", "--")

        results.append((means, stds, topo["label"], format_spec))

    return results


def calculate_redundancy(G, size):
    """
    Calculate redundancy level R of the graph (edges / minimum edges needed).

    Args:
        G: NetworkX graph
        size: Size parameter for redundancy calculation

    Returns:
        Redundancy ratio (>= 1.0)
    """
    min_links = len(G.nodes()) - 1
    if min_links <= 0:
        return 0
    return len(G.edges()) / min_links


def calculate_link_cost(u, v, pos):
    """
    Calculate cost of a link between nodes using Euclidean distance.

    Args:
        u, v: Node IDs
        pos: Dictionary mapping nodes to positions

    Returns:
        Cost value for the link
    """
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calculate_network_cost(G, pos):
    """
    Calculate total and per-node costs of a network.

    Args:
        G: NetworkX graph
        pos: Dictionary mapping nodes to positions

    Returns:
        Dictionary with total_cost, cost_per_node, cost_per_redundancy
    """
    if not G.edges():
        return {"total_cost": 0, "cost_per_node": 0, "cost_per_redundancy": 0}

    total_cost = sum(calculate_link_cost(u, v, pos) for u, v in G.edges())
    cost_per_node = total_cost / len(G.nodes()) if G.nodes() else 0

    # Estimate size from position dictionary
    size = int(np.sqrt(max(pos.keys()) + 1))
    redundancy = calculate_redundancy(G, size)
    cost_per_redundancy = total_cost / redundancy if redundancy > 0 else float("inf")

    return {
        "total_cost": total_cost,
        "cost_per_node": cost_per_node,
        "cost_per_redundancy": cost_per_redundancy,
    }


def calculate_node_importance(G):
    """
    Calculate importance of each node using centrality metrics.

    Combines degree centrality (40%) and betweenness centrality (60%)
    to identify critical nodes in the network.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping nodes to importance values (0.0 to 1.0)
    """
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)

    importance = {}
    for node in G.nodes():
        importance[node] = 0.4 * degree_cent[node] + 0.6 * betweenness_cent[node]

    return importance
