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


def run_failure_simulations(
    topologies, size, failure_probs, failure_type, num_trials=15
):
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
        print(
            f"Running {failure_name} failure simulation for {topo['label']}...")
        means, stds = run_simulation(
            topo["generator"], size, failure_probs, failure_type, num_trials
        )

        format_spec = topo["format"]

        results.append((means, stds, topo["label"], format_spec))

    return results


def calculate_redundancy(G, size):
    """
    Calculate redundancy level R of the graph (edges / minimum edges needed).
    """
    min_links = len(G.nodes()) - 1
    if min_links <= 0:
        return 0
    return len(G.edges()) / min_links


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
        importance[node] = 0.4 * degree_cent[node] + \
            0.6 * betweenness_cent[node]

    return importance
