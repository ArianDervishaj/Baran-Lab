import argparse
import time

import numpy as np

from simulation import run_failure_simulations
from topologies import (create_grid_plus_plus_topology,
                        create_grid_plus_topology, create_grid_topology,
                        create_hybrid_topology, create_line_topology,
                        create_ring_topology, create_star_topology,
                        create_tree_topology)
from visualisation import visualize_results


def define_topologies():
    """Define the network topologies to be simulated with expanded color options"""
    return [
        {
            "generator": create_line_topology,
            "name": "R=1: Topologie en ligne",
            "label": "R=1 (Ligne)",
            "format": "bo-",
        },
        {
            "generator": create_grid_topology,
            "name": "R=2: Topologie en grille",
            "label": "R=2 (Grille)",
            "format": "go-",
        },
        {
            "generator": create_grid_plus_topology,
            "name": "R=3: Topologie en grille+",
            "label": "R=3 (Grille+)",
            "format": "ro-",
        },
        {
            "generator": create_grid_plus_plus_topology,
            "name": "R=4: Topologie en grille++",
            "label": "R=4 (Grille++)",
            "format": "do-",
        },
        {
            "generator": create_tree_topology,
            "name": "Topologie en arbre",
            "label": "Arbre",
            "format": "mo-",
        },
        {
            "generator": create_star_topology,
            "name": "Topologie en étoile",
            "label": "Etoile",
            "format": "vo-",
        },
        {
            "generator": create_ring_topology,
            "name": "Topologie en Anneau",
            "label": "Anneau",
            "format": "io-",
        },
        {
            "generator": create_hybrid_topology,
            "name": "Topologie hybrid",
            "label": "Hybrid",
            "format": "oo-",
        },
    ]


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Network Resilience Simulation")
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=18,
        help="Size parameter for the simulation (default: 18)",
    )
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        default=25,
        help="Number of probability points (default: 25)",
    )
    parser.add_argument(
        "-n",
        "--trials",
        type=int,
        default=15,
        help="Number of trials per probability point (default: 15)",
    )
    parser.add_argument(
        "-v",
        "--visual",
        type=int,
        default=6,
        help="Number of node in the visualisation",
    )
    return parser.parse_args()


def main():
    """Main execution function for network resilience simulation"""
    args = parse_arguments()

    print(
        f"Running simulation with size={args.size}, points={args.points}, trials={args.trials}, visualisation size={args.visual}"
    )

    start_time = time.time()

    failure_probs = np.linspace(0, 1, args.points)

    topologies = define_topologies()

    node_results = run_failure_simulations(
        topologies, args.size, failure_probs, "node", args.trials
    )

    link_results = run_failure_simulations(
        topologies, args.size, failure_probs, "link", args.trials
    )

    visualize_results(
        topologies, failure_probs, node_results, link_results, args.size, args.visual
    )

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
