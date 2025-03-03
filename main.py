import time

import numpy as np

from simulation import run_failure_simulations
from topologies import (create_grid_plus_plus_topology,
                        create_grid_plus_topology, create_grid_topology,
                        create_hybrid_topology, create_line_topology,
                        create_ring_topology, create_star_topology,
                        create_tree_topology)
from visualisation import visualize_results

SIZE = 18


def define_topologies():
    """Define the network topologies to be simulated with expanded color options"""
    return [
        {
            "generator": create_line_topology,
            "name": "R=1: Topologie en ligne",
            "label": "R=1 (Ligne)",
            "format": "bo-",  # Dodgerblue circles
        },
        {
            "generator": create_grid_topology,
            "name": "R=2: Topologie en grille",
            "label": "R=2 (Grille)",
            "format": "go-",  # Limegreen circles
        },
        {
            "generator": create_grid_plus_topology,
            "name": "R=3: Topologie en grille+",
            "label": "R=3 (Grille+)",
            "format": "ro-",  # Crimson circles
        },
        {
            "generator": create_grid_plus_plus_topology,
            "name": "R=4: Topologie en grille++",
            "label": "R=4 (Grille++)",
            "format": "do-",  # Darkviolet circles
        },
        {
            "generator": create_tree_topology,
            "name": "Topologie en arbre",
            "label": "Arbre",
            "format": "mo-",  # Darkorange circles
        },
        {
            "generator": create_star_topology,
            "name": "Topologie en étoile",
            "label": "Etoile",
            "format": "vo-",  # Deeppink circles
        },
        {
            "generator": create_ring_topology,
            "name": "Topologie en Anneau",
            "label": "Anneau",
            "format": "io-",  # Teal circles
        },
        {
            "generator": create_hybrid_topology,
            "name": "Topologie hybrid",
            "label": "Hybrid",
            "format": "oo-",  # Navy circles
        },
    ]


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
