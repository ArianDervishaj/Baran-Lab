import time

import numpy as np

from simulation import run_simulation
from topologies import (create_grid_plus_plus_topology,
                        create_grid_plus_topology, create_grid_topology,
                        create_hybrid_topology, create_line_topology,
                        create_ring_topology, create_star_topology,
                        create_tree_topology)
from visualisation import (plot_both_failures,
                           visualize_topologies_with_importance)

SIZE = 18


def define_topologies():
    """Define the network topologies to be simulated"""
    return [
        {
            "generator": create_line_topology,
            "name": "R=1: Topologie en ligne",
            "label": "R=1 (Ligne)",
            "format": "bo-",
            "color": "skyblue",
        },
        {
            "generator": create_grid_topology,
            "name": "R=2: Topologie en grille",
            "label": "R=2 (Grille)",
            "format": "go-",
            "color": "lightgreen",
        },
        {
            "generator": create_grid_plus_topology,
            "name": "R=3: Topologie en grille+",
            "label": "R=3 (Grille+)",
            "format": "ro-",
            "color": "salmon",
        },
        {
            "generator": create_grid_plus_plus_topology,
            "name": "R=4: Topologie en grille++",
            "label": "R=4 (Grille++)",
            "format": "mo-",
            "color": "plum",
        },
        {
            "generator": create_tree_topology,
            "name": "Topologie en arbre",
            "label": "Arbre",
            "format": "co-",
            "color": "yellow",
        },
        {
            "generator": create_star_topology,
            "name": "Topologie en étoile",
            "label": "Etoile",
            "format": "yo-",
            "color": "green",
        },
        {
            "generator": create_ring_topology,
            "name": "Topologie en Anneau",
            "label": "Anneau",
            "format": "mo-",
            "color": "red",
        },
        {
            "generator": create_hybrid_topology,
            "name": "Topologie hybrid",
            "label": "Hybrid",
            "format": "ko-",
            "color": "lightcyan",
        },
    ]


def run_failure_simulations(topologies, size, failure_probs, failure_type):
    """Run simulations for all topologies with specified failure type"""
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


def visualize_results(
    topologies, failure_probs, node_results, link_results, size=SIZE, small_size=6
):
    """Generate all visualizations"""
    # Visualize topology structures
    visualize_topologies_with_importance(size, topologies, small_size)

    # Optional: Plot individual node failure results
    # plot_results(failure_probs, *node_results, failure_type="nodes")

    # Optional: Plot individual link failure results
    # plot_results(failure_probs, *link_results, failure_type="liens")

    # Plot combined node and link failure comparison
    print("\nPlotting combined comparison with appropriate best lines...")
    plot_both_failures(failure_probs, node_results, link_results)


def main():
    """Main execution function for network resilience simulation"""
    start_time = time.time()

    # Generate failure probabilities
    failure_probs = np.linspace(0, 1, 21)

    # Define topologies to simulate
    topologies = define_topologies()

    # Run simulations
    node_results = run_failure_simulations(topologies, SIZE, failure_probs, "node")
    link_results = run_failure_simulations(topologies, SIZE, failure_probs, "link")

    # Generate visualizations
    visualize_results(topologies, failure_probs, node_results, link_results)

    # Report execution time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
