import argparse
import sys
from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.widgets import Button

from simulation import (calculate_network_cost, calculate_node_importance,
                        calculate_redundancy)
from topologies import (create_grid_plus_plus_topology,
                        create_grid_plus_topology, create_grid_topology,
                        create_hybrid_topology, create_line_topology,
                        create_ring_topology, create_star_topology,
                        create_tree_topology)

TOPOLOGY_MAP = {
    "line": (create_line_topology, "Line"),
    "grid": (create_grid_topology, "Grid"),
    "grid+": (create_grid_plus_topology, "Grid+"),
    "grid++": (create_grid_plus_plus_topology, "Grid++"),
    "tree": (create_tree_topology, "Tree"),
    "star": (create_star_topology, "Star"),
    "ring": (create_ring_topology, "Ring"),
    "hybrid": (create_hybrid_topology, "Hybrid"),
}


def get_topology_generator(topology_name):
    """
    Get the appropriate topology generator function based on the name.

    Args:
        topology_name: Name of the topology

    Returns:
        Tuple of (generator_function, display_name)

    Raises:
        SystemExit: If topology name is not found
    """
    # Try to find the topology in our map
    if topology_name in TOPOLOGY_MAP:
        return TOPOLOGY_MAP[topology_name]

    # Print available topologies if not found
    print(f"Topology '{topology_name}' not found.")
    print("Available topologies:")
    for key in TOPOLOGY_MAP:
        print(f"  - {key}")
    sys.exit(1)


def calculate_component_statistics(graph):
    """
    Calculate statistics about connected components in the graph.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary containing connected component statistics
    """
    # Get all connected components
    components = list(nx.connected_components(graph))

    # Count the number of components
    num_components = len(components)

    # Find the largest component size
    largest_component_size = len(max(components, key=len)) if components else 0

    # Calculate percentage of nodes in largest component
    total_nodes = len(graph.nodes())
    largest_component_percentage = (
        (largest_component_size / total_nodes) * 100 if total_nodes > 0 else 0
    )

    return {
        "num_components": num_components,
        "largest_component_size": largest_component_size,
        "largest_component_percentage": largest_component_percentage,
    }


def calculate_relative_importance(graph, node_importance):
    """
    Calculate node importance relative to the largest connected component.

    Args:
        graph: NetworkX graph
        node_importance: Dictionary mapping nodes to their importance values

    Returns:
        Dictionary mapping nodes to their relative importance values
    """
    # Get all connected components
    components = list(nx.connected_components(graph))

    if not components:
        return node_importance

    # Find the largest component
    largest_component = max(components, key=len)

    # Calculate the max importance in the largest component
    largest_component_nodes = {
        node: node_importance[node] for node in largest_component
    }
    max_importance = (
        max(largest_component_nodes.values()) if largest_component_nodes else 1.0
    )

    # Normalize importance values relative to max in largest component
    relative_importance = {}
    for node, importance in node_importance.items():
        if node in largest_component:
            # Nodes in largest component keep their relative importance
            relative_importance[node] = (
                importance / max_importance if max_importance > 0 else 0
            )
        else:
            # Nodes outside the largest component get lower importance values
            component_size_factor = (
                0.5  # Nodes in smaller components have reduced importance
            )
            relative_importance[node] = (
                (importance / max_importance) * component_size_factor
                if max_importance > 0
                else 0
            )

    return relative_importance


def calculate_statistics(graph, positions, size):
    """
    Calculate network statistics.

    Args:
        graph: NetworkX graph
        positions: Dictionary mapping nodes to positions
        size: Size parameter of the graph

    Returns:
        Dictionary of statistics including links, nodes, redundancy, and cost
    """
    links = len(graph.edges())
    nodes = len(graph.nodes())
    redundancy = calculate_redundancy(graph, size)
    cost_info = calculate_network_cost(graph, positions)

    # Add connected component statistics
    component_stats = calculate_component_statistics(graph)

    stats = {
        "links": links,
        "nodes": nodes,
        "redundancy": redundancy,
        "cost": cost_info,
    }

    # Merge component statistics into main stats
    stats.update(component_stats)

    return stats


def create_visualization(topology_generator, topology_name, size=6):
    """
    Create and set up the interactive visualization.

    Args:
        topology_generator: Function to generate the topology
        topology_name: Display name for the topology
        size: Size parameter for the visualization

    Returns:
        Tuple of (fig, state_dict) containing figure and visualization state
    """
    # Generate the network graph
    graph = topology_generator(size)

    # Initialize visualization state
    state = {
        "graph": graph,
        "deleted_nodes": set(),
        "size": size,
        "topology_name": topology_name,
        "cbar": None,
    }

    # Create a new figure
    plt.close("all")
    fig = plt.figure(figsize=(12, 10))

    # Add title with proper spacing
    fig.suptitle(
        f"Interactive {topology_name} Topology Visualization", fontsize=16, y=0.98
    )

    # Create a reset button
    reset_ax = plt.axes([0.45, 0.05, 0.15, 0.05])
    reset_button = Button(reset_ax, "Reset")
    reset_button.on_clicked(partial(reset_topology, state=state, fig=fig))

    # Add instructions text
    plt.figtext(
        0.5,
        0.02,
        "Click on a node to delete it and recalculate importance",
        ha="center",
        fontsize=12,
    )

    # Connect click event handler for node deletion
    fig.canvas.mpl_connect(
        "button_press_event", partial(on_click, state=state, fig=fig)
    )

    return fig, state


def draw_topology(state, fig, first_draw=False):
    """
    Draw the topology with node importance visualization.

    Args:
        state: Dictionary containing visualization state
        fig: Matplotlib figure
        first_draw: Whether this is the first time drawing
    """
    # Clear existing plot
    plt.clf()

    # Create a new axis with more space for visualization
    ax = plt.axes([0.05, 0.15, 0.9, 0.75])
    ax.set_aspect("equal")

    # Remove old colorbar if it exists
    if state["cbar"] is not None:
        try:
            state["cbar"].remove()
            state["cbar"] = None
        except Exception:
            pass

    # Create a working copy of the graph with deleted nodes removed
    working_graph = state["graph"].copy()
    working_graph.remove_nodes_from(state["deleted_nodes"])

    # Calculate node positions on a grid
    positions = {
        i: (i % state["size"], -(i // state["size"])) for i in working_graph.nodes()
    }

    # Set consistent axis limits with padding for better centering
    padding = 0.5
    plt.xlim(-padding, state["size"] - 1 + padding)
    plt.ylim(-(state["size"] - 1 + padding), padding)

    # Draw the network if there are nodes
    if working_graph.nodes():
        # Calculate importance metrics
        node_importance = calculate_node_importance(working_graph)

        # Calculate importance relative to largest connected component
        relative_importance = calculate_relative_importance(
            working_graph, node_importance
        )

        # Get color scale range
        min_importance = min(relative_importance.values()) if relative_importance else 0
        max_importance = max(relative_importance.values()) if relative_importance else 1

        # Get node colors based on relative importance
        node_colors = [relative_importance[node] for node in working_graph.nodes()]

        # Draw edges first
        nx.draw_networkx_edges(
            working_graph, positions, width=1.5, alpha=0.7, edge_color="gray"
        )

        # Draw nodes with importance-based coloring
        nodes = nx.draw_networkx_nodes(
            working_graph,
            positions,
            node_size=250,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            vmin=min_importance,
            vmax=max_importance,
        )

        # Draw node labels
        nx.draw_networkx_labels(
            working_graph, positions, font_size=10, font_color="white"
        )

        # Add colorbar for importance visualization
        if nodes:
            state["cbar"] = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
            state["cbar"].set_label("Relative Node Importance", fontsize=10)

    # Calculate and display statistics
    stats = calculate_statistics(working_graph, positions, state["size"])

    # Calculate how many nodes have been deleted
    original_nodes = len(state["graph"].nodes())
    current_nodes = len(working_graph.nodes())
    node_reduction = original_nodes - current_nodes

    # Set title with network statistics including component info
    plt.title(
        f"{state['topology_name']} Topology\n"
        f"Links: {stats['links']}, Nodes: {current_nodes} ({node_reduction} deleted)\n"
        f"Connected Components: {stats['num_components']}, "
        f"Largest: {stats['largest_component_size']} nodes "
        f"({stats['largest_component_percentage']:.1f}%)\n"
        f"Redundancy: {stats['redundancy']:.2f}, Cost: {stats['cost']['total_cost']:.0f}",
        pad=20,
    )

    # Hide the axis elements (ticks, labels, etc.)
    plt.axis("off")

    # Force canvas update
    fig.canvas.draw()
    plt.pause(0.001)


def on_click(event, state, fig):
    """
    Handle mouse click events to delete nodes.

    Args:
        event: The matplotlib event
        state: Visualization state dictionary
        fig: The matplotlib figure
    """
    # Check if click is within the visualization area
    ax = plt.gca()
    if event.inaxes != ax:
        return

    # Get the working graph with already deleted nodes removed
    working_graph = state["graph"].copy()
    working_graph.remove_nodes_from(state["deleted_nodes"])

    # Skip if no nodes left
    if not working_graph.nodes():
        return

    # Calculate node positions
    positions = {
        i: (i % state["size"], -(i // state["size"])) for i in working_graph.nodes()
    }

    # Find the node closest to the click point
    click_pos = (event.xdata, event.ydata)
    closest_node = None
    min_distance = float("inf")

    # Check all nodes to find the closest one
    for node, pos in positions.items():
        distance = np.sqrt((click_pos[0] - pos[0]) ** 2 + (click_pos[1] - pos[1]) ** 2)
        # Only consider nodes within certain distance
        if distance < min_distance and distance < 0.2:
            min_distance = distance
            closest_node = node

    # Delete the node if one was found
    if closest_node is not None:
        state["deleted_nodes"].add(closest_node)
        draw_topology(state, fig)


def reset_topology(event, state, fig):
    """
    Reset the topology by clearing deleted nodes.

    Args:
        event: The matplotlib event
        state: Visualization state dictionary
        fig: The matplotlib figure
    """
    state["deleted_nodes"] = set()
    draw_topology(state, fig)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Interactive Network Topology Visualization"
    )
    parser.add_argument(
        "topology", help='Topology to visualize (e.g., "tree", "grid", "line")'
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=6,
        help="Size parameter for the visualization (default: 6)",
    )
    return parser.parse_args()


def main():
    """
    Main function to run the visualization.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Get topology generator based on name
    topology_generator, display_name = get_topology_generator(args.topology)

    # Create the visualization
    fig, state = create_visualization(topology_generator, display_name, args.size)

    # Draw the initial topology
    draw_topology(state, fig, first_draw=True)

    # Show the visualization
    plt.show()


if __name__ == "__main__":
    main()
