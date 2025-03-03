import matplotlib.pyplot as plt
import networkx as nx

from simulation import (calculate_network_cost, calculate_node_importance,
                        calculate_redundancy)

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


def plot_results(failure_probs, *result_data, failure_type="nodes"):
    """
    Plot simulation results with error bars showing standard deviation.

    Args:
        failure_probs: Array of failure probabilities
        *result_data: Tuples of (results, std_dev, label, format)
        failure_type: Type of failures ("nodes" or "links")
    """
    plt.figure(figsize=(12, 7))

    for data in result_data:
        results, std_dev, label, format_spec = data

        color, marker, linestyle = _parse_format(format_spec)

        plt.errorbar(
            failure_probs,
            results,
            yerr=std_dev,
            marker=marker,
            linestyle=linestyle,
            color=color,
            capsize=4,
            label=label,
        )

    # Add reference line
    best_line = [100 * (1 - p) for p in failure_probs]
    plt.plot(failure_probs, best_line, "k--", label="Meilleure ligne possible")

    plt.xlabel(f"Probabilité de panne des {failure_type} (P)")
    plt.ylabel("Niveau de survie (S) %")
    plt.title("Comparaison des topologies de réseau")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_both_failures(failure_probs, node_results, link_results):
    """
    Plot node failures and link failures side by side with appropriate reference lines.

    Args:
        failure_probs: Array of failure probabilities
        node_results: List of tuples (results, std_dev, label, format) for node failures
        link_results: List of tuples (results, std_dev, label, format) for link failures
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

    _add_link_theoretical_line(link_ax, failure_probs)

    plt.suptitle("Comparaison des topologies de réseau", fontsize=16)
    plt.show()


def _plot_failure_subplot(ax, failure_probs, results_data, title, xlabel):
    """Helper function to plot a failure subplot with extended colors"""
    for i, data in enumerate(results_data):
        results, std_dev, label, format_spec = data

        color, marker, linestyle = _parse_format(format_spec)

        ax.errorbar(
            failure_probs,
            results,
            yerr=std_dev,
            marker=marker,
            linestyle=linestyle,
            color=color,
            capsize=4,
            label=label,
        )

    reference_line = [100 * (1 - p) for p in failure_probs]
    ax.plot(failure_probs, reference_line, "k--", label="Meilleure ligne possible")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Niveau de survie (S) %")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def _add_link_theoretical_line(ax, failure_probs):
    """Add theoretical optimum line for link failures"""
    SIZE = 18
    node_count = SIZE * SIZE

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


def visualize_topologies_with_importance(topologies, small_size=6):
    """
    Create a grid of visualizations for different network topologies with node importance.

    Args:
        topologies: List of topology dictionaries with keys: generator, name, label, color
        small_size: Size for visualization (default: 6)
    """
    topology_data = _prepare_topology_data(topologies, small_size)

    num_topologies = len(topologies)
    cols = 4
    rows = (num_topologies + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 5, rows * 5))
    grid = plt.GridSpec(rows, cols, figure=fig)
    fig.suptitle("Topologies de réseau avec importance des nœuds", fontsize=16)

    for i, (topo, graph, stats, importance) in enumerate(topology_data):
        _plot_topology(fig, grid, i, cols, topo, graph, stats, importance, small_size)

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

        positions = {i: (i % small_size, i // small_size) for i in graph.nodes()}

        importance = calculate_node_importance(graph)
        links = len(graph.edges())
        nodes = len(graph.nodes())
        redundancy = calculate_redundancy(graph, small_size)
        cost_info = calculate_network_cost(graph, positions)

        stats = {
            "name": topo["label"],
            "links": links,
            "nodes": nodes,
            "redundancy": redundancy,
            "cost": cost_info,
        }

        topology_data.append((topo, graph, stats, importance))

    return topology_data


def _plot_topology(fig, grid, index, cols, topo, graph, stats, importance, small_size):
    """Plot a single topology visualization"""
    row, col = index // cols, index % cols
    ax = fig.add_subplot(grid[row, col])

    # Create node positions for visualization
    positions = {i: (i % small_size, -(i // small_size)) for i in graph.nodes()}

    # Set color scale based on importance
    if importance and len(importance) > 0:
        min_importance = min(importance.values())
        max_importance = max(importance.values())
    else:
        min_importance, max_importance = 0, 1

    # Get node colors from importance values
    node_colors = [importance[node] for node in graph.nodes()]

    # Draw the network
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

    # Add colorbar for importance
    cbar = plt.colorbar(nodes, ax=ax, shrink=0.6)
    cbar.set_label("Importance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Set title with statistics
    ax.set_title(
        f"{topo.get('name', topo['label'])}\n"
        f"Liens: {stats['links']}, Nœuds: {stats['nodes']}\n"
        f"Redondance: {stats['redundancy']:.2f}\n"
        f"Coût: {stats['cost']['total_cost']:.0f}"
    )

    ax.axis("off")


def visualize_results(
    topologies, failure_probs, node_results, link_results, small_size=6
):
    """
    Generate all visualizations.

    Args:
        topologies: List of topology dictionaries
        failure_probs: Array of failure probabilities
        node_results: Results from node failure simulations
        link_results: Results from link failure simulations
        small_size: Size for topology visualization (default: 6)
    """
    # Visualize topology structures
    visualize_topologies_with_importance(topologies, small_size)

    # Optional: Plot individual node failure results
    # plot_results(failure_probs, *node_results, failure_type="nodes")

    # Optional: Plot individual link failure results
    # plot_results(failure_probs, *link_results, failure_type="liens")

    # Plot combined node and link failure comparison
    print("\nPlotting combined comparison with appropriate best lines...")
    plot_both_failures(failure_probs, node_results, link_results)
