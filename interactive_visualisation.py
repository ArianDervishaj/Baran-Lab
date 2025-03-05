import argparse
import sys
from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.widgets import Button

from simulation import calculate_node_importance, calculate_redundancy
from topologies import (create_grid_plus_plus_topology,
                        create_grid_plus_topology, create_grid_topology,
                        create_hybrid_topology, create_line_topology,
                        create_ring_topology, create_star_topology,
                        create_tree_topology)
from visualisation import get_layout_algorithm

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
    """Get the appropriate topology generator function and display name."""
    if topology_name in TOPOLOGY_MAP:
        return TOPOLOGY_MAP[topology_name]

    print(f"Topology '{topology_name}' not found.")
    print("Available topologies:")
    for key in TOPOLOGY_MAP:
        print(f"  - {key}")
    sys.exit(1)


def calculate_component_statistics(graph):
    """Calculate statistics about connected components."""
    components = list(nx.connected_components(graph))
    num_components = len(components)
    largest_component_size = len(max(components, key=len)) if components else 0
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
    """Calculate node importance relative to the largest connected component."""
    components = list(nx.connected_components(graph))
    if not components:
        return node_importance

    largest_component = max(components, key=len)
    largest_component_nodes = {
        node: node_importance[node] for node in largest_component
    }
    max_importance = (
        max(largest_component_nodes.values()) if largest_component_nodes else 1.0
    )

    relative_importance = {}
    for node, importance in node_importance.items():
        if node in largest_component:
            relative_importance[node] = (
                importance / max_importance if max_importance > 0 else 0
            )
        else:
            component_size_factor = 0.5
            relative_importance[node] = (
                (importance / max_importance) * component_size_factor
                if max_importance > 0
                else 0
            )

    return relative_importance


def calculate_statistics(graph, positions, size):
    """Calculate network statistics including links, nodes, redundancy."""
    links = len(graph.edges())
    nodes = len(graph.nodes())
    redundancy = calculate_redundancy(graph, size)
    component_stats = calculate_component_statistics(graph)

    stats = {
        "links": links,
        "nodes": nodes,
        "redundancy": redundancy,
    }
    stats.update(component_stats)

    return stats


def create_visualization(topology_generator, topology_name, size=6):
    """Create and set up the interactive visualization."""
    graph = topology_generator(size)

    state = {
        "graph": graph,
        "deleted_nodes": set(),
        "size": size,
        "topology_name": topology_name,
        "cbar": None,
        "positions": None,
        "original_positions": None,
        "seed": 42,
    }

    plt.close("all")
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(
        f"Interactive {topology_name} Topology Visualization", fontsize=12, y=0.98
    )

    reset_ax = plt.axes([0.45, 0.05, 0.15, 0.05])
    reset_button = Button(reset_ax, "Reset")
    reset_button.on_clicked(partial(reset_topology, state=state, fig=fig))

    plt.figtext(
        0.5,
        0.02,
        "Click on a node to delete it and recalculate importance",
        ha="center",
        fontsize=10,
    )

    fig.canvas.mpl_connect(
        "button_press_event", partial(on_click, state=state, fig=fig)
    )

    return fig, state


def draw_topology(state, fig, first_draw=False):
    """Draw the topology with node importance visualization."""
    plt.clf()
    ax = plt.axes([0.05, 0.15, 0.9, 0.75])
    ax.set_aspect("equal")

    if state["cbar"] is not None:
        try:
            state["cbar"].remove()
            state["cbar"] = None
        except Exception:
            pass

    working_graph = state["graph"].copy()
    working_graph.remove_nodes_from(state["deleted_nodes"])

    # Calculate positions once and reuse for stability
    if first_draw or state["original_positions"] is None:
        state["original_positions"] = get_layout_algorithm(
            state["topology_name"], state["graph"], seed=state["seed"]
        )

    # Use subset of original positions for visible nodes
    state["positions"] = {
        node: state["original_positions"][node]
        for node in working_graph.nodes()
        if node in state["original_positions"]
    }

    positions = state["positions"]

    # Keep view consistent based on original layout
    if positions:
        x_coords = [pos[0] for pos in state["original_positions"].values()]
        y_coords = [pos[1] for pos in state["original_positions"].values()]

        if x_coords and y_coords:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            padding = 2.0
            plt.xlim(x_min - padding, x_max + padding)
            plt.ylim(y_min - padding, y_max + padding)
    else:
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

    if working_graph.nodes():
        node_importance = calculate_node_importance(working_graph)
        relative_importance = calculate_relative_importance(
            working_graph, node_importance
        )

        min_importance = min(relative_importance.values()
                             ) if relative_importance else 0
        max_importance = max(relative_importance.values()
                             ) if relative_importance else 1
        node_colors = [relative_importance[node]
                       for node in working_graph.nodes()]

        nx.draw_networkx_edges(
            working_graph, positions, width=1.5, alpha=0.7, edge_color="gray"
        )

        nodes = nx.draw_networkx_nodes(
            working_graph,
            positions,
            node_size=250,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            vmin=min_importance,
            vmax=max_importance,
        )

        nx.draw_networkx_labels(
            working_graph, positions, font_size=10, font_color="white"
        )

        if nodes:
            state["cbar"] = plt.colorbar(
                nodes, ax=ax, fraction=0.046, pad=0.04)
            state["cbar"].set_label("Relative Node Importance", fontsize=10)

    stats = calculate_statistics(working_graph, positions, state["size"])
    original_nodes = len(state["graph"].nodes())
    current_nodes = len(working_graph.nodes())
    node_reduction = original_nodes - current_nodes

    plt.title(
        f"{state['topology_name']} Topology\n"
        f"Links: {stats['links']}, Nodes: {current_nodes} ({node_reduction} deleted)\n"
        f"Connected Components: {stats['num_components']}, "
        f"Largest: {stats['largest_component_size']} nodes "
        f"({stats['largest_component_percentage']:.1f}%)\n"
        f"Redundancy: {stats['redundancy']:.2f}",
        pad=20,
    )

    plt.axis("off")
    fig.canvas.draw()
    plt.pause(0.001)


def on_click(event, state, fig):
    """Handle mouse click events to delete nodes."""
    ax = plt.gca()
    if event.inaxes != ax:
        return

    working_graph = state["graph"].copy()
    working_graph.remove_nodes_from(state["deleted_nodes"])

    if not working_graph.nodes():
        return

    positions = state["positions"]
    if not positions:
        return

    # Find node closest to click
    click_pos = (event.xdata, event.ydata)
    closest_node = None
    min_distance = float("inf")

    for node, pos in positions.items():
        distance = np.sqrt((click_pos[0] - pos[0])
                           ** 2 + (click_pos[1] - pos[1]) ** 2)
        if distance < min_distance and distance < 1.0:
            min_distance = distance
            closest_node = node

    if closest_node is not None:
        state["deleted_nodes"].add(closest_node)
        draw_topology(state, fig)


def reset_topology(event, state, fig):
    """Reset the topology by clearing deleted nodes."""
    state["deleted_nodes"] = set()
    draw_topology(state, fig)


def parse_arguments():
    """Parse command line arguments."""
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
    """Main function to run the visualization."""
    args = parse_arguments()
    topology_generator, display_name = get_topology_generator(args.topology)
    fig, state = create_visualization(
        topology_generator, display_name, args.size)
    draw_topology(state, fig, first_draw=True)
    plt.show()


if __name__ == "__main__":
    main()
