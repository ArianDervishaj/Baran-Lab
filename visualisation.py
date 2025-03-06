import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from simulation import calculate_node_importance, calculate_redundancy

COLOR_MAP = {
    "b": "blue",
    "g": "green",
    "r": "red",
    "c": "cyan",
    "m": "magenta",
    "y": "yellow",
    "k": "black",
    "w": "white",
    "v": "darkviolet",
    "i": "indigo",
    "o": "darkorange",
    "d": "gold",
}


def _parse_format(format_spec):
    """
    Parse format string with extended color options.

    Format: 'XYZ' where:
    - X is a color code from COLOR_MAP
    - Y is a marker type
    - Z is a linestyle (can be multiple chars)

    Returns: color, marker, linestyle
    """
    if not format_spec:
        return "black", "o", "-"

    color_code = format_spec[0]
    color = COLOR_MAP.get(color_code, "black")

    marker = format_spec[1] if len(format_spec) > 1 else "o"

    linestyle = format_spec[2:] if len(format_spec) > 2 else "-"
    if not linestyle:
        linestyle = "-"

    return color, marker, linestyle


def plot_both_failures(failure_probs, node_results, link_results, size):
    """
    Plot node failures and link failures side by side with appropriate reference lines.
    """

    fig, (node_ax, link_ax) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot node failures (left subplot)
    _plot_failure_subplot(
        node_ax,
        failure_probs,
        node_results,
        "Défaillance des nœuds",
        "Probabilité de panne des nœuds (P)",
    )

    # Plot link failures (right subplot)
    _plot_failure_subplot(
        link_ax,
        failure_probs,
        link_results,
        "Défaillance des liens",
        "Probabilité de panne des liens (P)",
    )

    _add_link_theoretical_line(link_ax, failure_probs, size)

    plt.suptitle("Comparaison des topologies de réseau", fontsize=16)
    plt.show()


def _plot_failure_subplot(ax, failure_probs, results_data, title, xlabel):
    """Helper function to plot a failure subplot with extended colors"""
    for i, data in enumerate(results_data):
        results, std_dev, label, format_spec = data

        color, marker, linestyle = _parse_format(format_spec)

        # Lower error can't go below 0, upper error can't go above 100
        yerr_lower = np.minimum(std_dev, results)
        yerr_upper = np.minimum(std_dev, 100 - np.array(results))
        yerr = [yerr_lower, yerr_upper]

        ax.errorbar(
            failure_probs,
            results,
            yerr=yerr,
            marker=marker,
            linestyle=linestyle,
            color=color,
            capsize=4,
            label=label,
        )

    # Optimal line for node failures
    if "liens" not in xlabel:
        reference_line = [100 * (1 - p) for p in failure_probs]
        ax.plot(failure_probs, reference_line, "k--", label="Meilleure ligne possible")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Niveau de survie (S) %")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def _add_link_theoretical_line(ax, failure_probs, size):
    """Add theoretical optimum line for link failures"""
    node_count = size * size

    # Full mesh has n*(n-1)/2 edges
    total_edges = node_count * (node_count - 1) // 2

    # Minimum edges for connectivity
    min_edges_required = node_count - 1

    # Calculate theoretical maximum survival
    theoretical_line = []
    for prob in failure_probs:
        remaining_edges = total_edges * (1 - prob)
        if remaining_edges >= min_edges_required:
            theoretical_line.append(100)
        else:
            # Approximate scaling as edges are removed below minimum threshold
            theoretical_line.append(100 * (remaining_edges / min_edges_required))

    ax.plot(
        failure_probs,
        theoretical_line,
        "g:",
        label="Limite théorique optimale",
        linewidth=2,
    )


def get_layout_algorithm(topology_name, graph, seed=42):
    """
    Selects the appropriate layout algorithm based on the topology name.
    """
    if "ligne" in topology_name.lower():
        pos = nx.spiral_layout(graph, equidistant=True)

    elif "arbre" in topology_name.lower():
        pos = nx.kamada_kawai_layout(graph)

    elif "étoile" in topology_name.lower() or "star" in topology_name.lower():

        # Identify the center node (highest degree)
        center_node = max(graph.degree, key=lambda x: x[1])[0]
        shells = [[center_node], [n for n in graph.nodes() if n != center_node]]
        pos = nx.shell_layout(graph, shells, scale=10)

    elif "anneau" in topology_name.lower() or "ring" in topology_name.lower():
        pos = nx.circular_layout(graph, scale=10)

    elif "grille" in topology_name.lower() or "grid" in topology_name.lower():
        pos = nx.spring_layout(graph, seed=seed, iterations=100, scale=10)

    elif "hybrid" in topology_name.lower():
        pos = nx.spring_layout(graph, seed=seed, iterations=500)

    else:
        pos = nx.spring_layout(graph, seed=seed, iterations=100, scale=10)

    return pos


def visualize_topologies_with_importance(topologies, visual_size=6):
    """
    Create a grid of visualizations for different network topologies with node importance
    """

    topology_data = _prepare_topology_data(topologies, visual_size)

    num_topologies = len(topologies)
    cols = 4
    rows = (num_topologies + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 5, rows * 5))
    grid = plt.GridSpec(rows, cols, figure=fig)
    fig.suptitle("Topologies de réseau avec importance des nœuds", fontsize=12)

    for i, (topo, graph, stats, importance) in enumerate(topology_data):
        _plot_topology(fig, grid, i, cols, topo, graph, stats, importance, visual_size)

    for i in range(num_topologies, rows * cols):
        row, col = i // cols, i % cols
        ax = fig.add_subplot(grid[row, col])
        ax.axis("off")

    plt.subplots_adjust(top=0.85)
    plt.show()


def _prepare_topology_data(topologies, small_size):
    """Prepare data for topology visualization"""
    topology_data = []

    for topo in topologies:
        graph = topo["generator"](small_size)

        importance = calculate_node_importance(graph)
        links = len(graph.edges())
        nodes = len(graph.nodes())
        redundancy = calculate_redundancy(graph, small_size)

        stats = {
            "name": topo["label"],
            "links": links,
            "nodes": nodes,
            "redundancy": redundancy,
        }

        topology_data.append((topo, graph, stats, importance))

    return topology_data


def _plot_topology(fig, grid, index, cols, topo, graph, stats, importance, small_size):
    """Plot a single topology visualization with realistic layout"""

    row, col = index // cols, index % cols
    ax = fig.add_subplot(grid[row, col])

    positions = get_layout_algorithm(topo["name"], graph, seed=index + 42)

    if importance and len(importance) > 0:
        min_importance = min(importance.values())
        max_importance = max(importance.values())
    else:
        min_importance, max_importance = 0, 1

    node_colors = [importance[node] for node in graph.nodes()]

    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=200,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        vmin=min_importance,
        vmax=max_importance,
    )

    nx.draw_networkx_edges(graph, positions, ax=ax, alpha=0.7)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=8)

    cbar = plt.colorbar(nodes, ax=ax, shrink=0.6)
    cbar.set_label("Importance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        f"{topo.get('name', topo['label'])}\n"
        f"Liens: {stats['links']}, Nœuds: {stats['nodes']}\n"
        f"Redondance: {stats['redundancy']:.2f}\n"
    )

    ax.axis("off")


def visualize_results(
    topologies, failure_probs, node_results, link_results, size, visual_size=6
):
    """
    Generate all visualizations.
    """
    visualize_topologies_with_importance(topologies, visual_size)

    print("\nPlotting combined comparison with appropriate best lines...")
    plot_both_failures(failure_probs, node_results, link_results, size)
