import random

import networkx as nx

# Grid Coordinate Utilities


def init_graph(size):
    """
    Initialize a graph with size*size nodes.

    Returns:
        Empty NetworkX graph with nodes indexed from 0 to size²-1
    """
    G = nx.Graph()
    G.add_nodes_from(range(size * size))
    return G


def coords_to_node(i, j, size):
    """
    Convert grid coordinates to node ID.

    Returns:
        Node ID in the flattened grid
    """
    return i * size + j


def node_to_coords(node, size):
    """
    Convert node ID to grid coordinates.

    Returns:
        Tuple (row, column) of coordinates
    """
    return node // size, node % size


def create_line_topology(size):
    """
    Create a line topology.
    """
    G = nx.path_graph(size * size)
    return G


def create_ring_topology(size):
    """
    Create a ring topology.
    """
    G = nx.cycle_graph(size * size)
    return G


def create_grid_topology(size):
    """
    Create a standard grid topology with vertical and horizontal connections.
    """

    # G = nx.grid_2d_graph(size, size)
    # return G

    grid = nx.grid_2d_graph(size, size)

    # Remap node IDs to match our convention
    mapping = {
        (i, j): coords_to_node(i, j, size) for i in range(size) for j in range(size)
    }
    G = nx.relabel_nodes(grid, mapping)
    return G


def create_grid_plus_topology(size):
    """
    Create a grid+ topology with main diagonal connections added.

    Returns:
        NetworkX graph with a grid+ topology
    """
    G = create_grid_topology(size)

    # Add diagonal connections (top-left to bottom-right)
    for i in range(size - 1):
        for j in range(size - 1):
            G.add_edge(coords_to_node(i, j, size),
                       coords_to_node(i + 1, j + 1, size))

    return G


def create_grid_plus_plus_topology(size):
    """
    Create a grid++ topology with both diagonal directions added.

    Returns:
        NetworkX graph with a grid++ topology
    """
    G = create_grid_plus_topology(size)

    # Add the other diagonal (top-right to bottom-left)
    for i in range(size - 1):
        for j in range(1, size):
            G.add_edge(coords_to_node(i, j, size),
                       coords_to_node(i + 1, j - 1, size))

    return G


def create_full_mesh_topology(size):
    """
    Create a fully connected mesh using NetworkX's complete_graph.

    Returns:
        NetworkX graph with a fully-connected topology
    """
    # Use NetworkX's built-in complete_graph
    G = nx.complete_graph(size * size)
    return G


def create_star_topology(size):
    """
    Create a star topology using NetworkX's star_graph.

    Returns:
        NetworkX graph with a star topology
    """
    # Use NetworkX's built-in star_graph
    # Note: star_graph uses 0 as the hub node by default
    G = nx.star_graph(size * size - 1)

    # Remap nodes if center is not the default center of our grid
    if size > 2 and (size * size - 1) // 2 != 0:
        # Map center from 0 to grid center
        center = coords_to_node(size // 2, size // 2, size)

        # Create a mapping that swaps 0 with the center
        mapping = {i: i for i in range(size * size)}
        mapping[0] = center
        mapping[center] = 0

        G = nx.relabel_nodes(G, mapping)

    return G


def create_tree_topology(size, branching_factor=3, max_depth=None):
    """
    Create a tree topology with specified branching factor.

    Args:
        size: Size of the grid
        branching_factor: Maximum number of children per node
        max_depth: Maximum depth of the tree (None for unlimited)

    Returns:
        NetworkX graph with a tree topology
    """
    # Use NetworkX's balanced_tree if appropriate
    total_nodes = size * size

    # Try to approximate with a balanced tree
    if branching_factor > 1:
        # Calculate the depth needed for this branching factor and total nodes
        # n = 1 + r + r^2 + ... + r^d for a tree with depth d and branching r
        depth = 0
        nodes_at_depth = 1
        total = 1

        while total < total_nodes and (max_depth is None or depth < max_depth):
            depth += 1
            nodes_at_depth *= branching_factor
            total += nodes_at_depth

        # Create a balanced tree with calculated depth
        G = nx.balanced_tree(branching_factor, depth)

        # If we have more nodes than needed, remove some leaf nodes
        if G.number_of_nodes() > total_nodes:
            # Get leaf nodes (nodes with degree 1)
            leaf_nodes = [n for n, d in G.degree() if d == 1]

            # Remove excess leaf nodes
            excess = G.number_of_nodes() - total_nodes
            G.remove_nodes_from(leaf_nodes[:excess])

        # Relabel nodes to match our convention (0 to total_nodes-1)
        mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)

        # Center the root node
        root = 0  # Assuming 0 is the root in the balanced tree
        center = coords_to_node(size // 2, size // 2, size)

        if root != center:
            # Swap root with center node
            mapping = {i: i for i in range(total_nodes)}
            mapping[root] = center
            mapping[center] = root
            G = nx.relabel_nodes(G, mapping)

        return G

    # Fall back to the original implementation if balanced_tree isn't suitable
    G = init_graph(size)

    # Define the root node at the center of the grid
    root_node = coords_to_node(size // 2, size // 2, size)

    # BFS tree construction
    current_level = [root_node]
    next_level = []
    visited = {root_node}

    # Direction priorities: cardinal directions first, then diagonals
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),  # Cardinal directions
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),  # Diagonal directions
    ]

    depth = 0
    while current_level and (max_depth is None or depth < max_depth):
        for parent in current_level:
            parent_i, parent_j = node_to_coords(parent, size)
            children_candidates = []

            # Check all possible directions for children
            for di, dj in directions:
                child_i, child_j = parent_i + di, parent_j + dj

                # Check if within grid bounds
                if 0 <= child_i < size and 0 <= child_j < size:
                    child = coords_to_node(child_i, child_j, size)
                    if child not in visited:
                        children_candidates.append(child)

            # Take only up to branching_factor children
            children = children_candidates[:branching_factor]

            # Connect parent to children
            for child in children:
                G.add_edge(parent, child)
                next_level.append(child)
                visited.add(child)

        current_level = next_level
        next_level = []
        depth += 1

    return G


def create_realistic_hybrid_topology(size, seed=None):
    """
    Create a realistic hybrid network topology resembling real-world networks.

    This creates a network with:
    - A core/backbone component (mesh-like)
    - Regional distribution networks (tree-like)
    - Metropolitan areas (grid-like)
    - Edge access networks (star-like)

    Args:
        size: Size of the grid
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph with a realistic hybrid topology
    """
    if seed is not None:
        random.seed(seed)

    G = init_graph(size)
    total_nodes = size * size

    # 1. Create the backbone/core network (small dense mesh at center)
    core_size = max(3, size // 4)
    core_start = (size - core_size) // 2
    core_end = core_start + core_size

    core_nodes = []
    for i in range(core_start, core_end):
        for j in range(core_start, core_end):
            node = coords_to_node(i, j, size)
            core_nodes.append(node)

    # Create a dense (but not complete) core network
    for i, u in enumerate(core_nodes):
        for j, v in enumerate(core_nodes[i + 1:], i + 1):
            # Create edges with high probability
            if random.random() < 0.7:
                G.add_edge(u, v)

    # 2. Create regional networks (tree-like structures branching from core)
    region_centers = [
        coords_to_node(core_start - 1, core_start - 1, size),  # Upper left
        coords_to_node(core_start - 1, core_end, size),  # Upper right
        coords_to_node(core_end, core_start - 1, size),  # Lower left
        coords_to_node(core_end, core_end, size),  # Lower right
    ]

    # Connect region centers to closest core nodes
    for region_center in region_centers:
        # Find closest core node
        rc_i, rc_j = node_to_coords(region_center, size)
        closest_core = min(
            core_nodes,
            key=lambda n: abs(rc_i - node_to_coords(n, size)[0])
            + abs(rc_j - node_to_coords(n, size)[1]),
        )
        G.add_edge(region_center, closest_core)

        # Create tree-like branches from each region center
        # These simulate backbone connections to regional distribution
        unvisited = [
            n
            for n in range(total_nodes)
            if n not in core_nodes and n not in region_centers
        ]

        # Get nodes close to this region center
        rc_i, rc_j = node_to_coords(region_center, size)
        region_quadrant = []

        # Determine which quadrant this region is in
        is_upper = rc_i < size // 2
        is_left = rc_j < size // 2

        # Get nodes in the same quadrant
        for node in unvisited:
            n_i, n_j = node_to_coords(node, size)
            if (is_upper == (n_i < size // 2)) and (is_left == (n_j < size // 2)):
                if random.random() < 0.8:  # Not all nodes in the quadrant
                    region_quadrant.append(node)

        # Create a tree/hub structure in each region
        if region_quadrant:
            # Create some sub-regional hubs
            num_hubs = min(4, max(1, len(region_quadrant) // 5))
            # Ensure we don't try to sample more elements than exist in region_quadrant
            num_hubs = min(num_hubs, len(region_quadrant))
            hubs = random.sample(
                region_quadrant, num_hubs) if num_hubs > 0 else []

            # Connect hubs to region center
            for hub in hubs:
                G.add_edge(region_center, hub)

            # Remaining nodes in region
            remaining = [n for n in region_quadrant if n not in hubs]

            # Connect nodes to closest hub with high probability
            for node in remaining:
                # Only try to find closest hub if there are hubs
                if hubs:
                    n_i, n_j = node_to_coords(node, size)
                    closest_hub = min(
                        hubs,
                        key=lambda h: abs(n_i - node_to_coords(h, size)[0])
                        + abs(n_j - node_to_coords(h, size)[1]),
                    )

                    # Connect to hub
                    if random.random() < 0.9:
                        G.add_edge(node, closest_hub)

    # 3. Create metropolitan areas (small grid-like structures)
    # Find areas that need more connectivity
    components = list(nx.connected_components(G))

    # If there are disconnected components, connect them
    if len(components) > 1:
        # Sort components by size (largest first)
        components = sorted(components, key=len, reverse=True)

        # Connect smaller components to the largest
        largest = components[0]
        for comp in components[1:]:
            # Find closest nodes between components
            closest_pair = None
            min_dist = float("inf")

            for u in comp:
                u_i, u_j = node_to_coords(u, size)
                for v in largest:
                    v_i, v_j = node_to_coords(v, size)
                    dist = abs(u_i - v_i) + abs(u_j - v_j)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (u, v)

            if closest_pair:
                G.add_edge(*closest_pair)

    # 4. Create edge access networks (star-like structures)
    # Identify good locations for access networks
    low_degree_nodes = [n for n, d in G.degree() if 0 < d <= 2]

    # Create small star networks around some low-degree nodes
    num_access_networks = min(size, len(low_degree_nodes) // 3)

    if num_access_networks > 0:
        access_centers = random.sample(low_degree_nodes, num_access_networks)

        # For each access center, find nearby unconnected nodes
        for center in access_centers:
            c_i, c_j = node_to_coords(center, size)

            # Find nodes close to this center that have no connections
            nearby_unconnected = []
            for node in range(total_nodes):
                if G.degree(node) == 0:
                    n_i, n_j = node_to_coords(node, size)
                    dist = abs(c_i - n_i) + abs(c_j - n_j)
                    if dist <= 3:  # Close enough to be "nearby"
                        nearby_unconnected.append(node)

            # Connect some nearby nodes to this center
            for node in nearby_unconnected:
                if random.random() < 0.8:
                    G.add_edge(center, node)

    # 5. Ensure the graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))

        # Connect all components to the largest one
        largest = max(components, key=len)
        other_components = [c for c in components if c != largest]

        for component in other_components:
            # Pick a random node from each component and connect to largest
            node_from_component = random.choice(list(component))
            node_from_largest = random.choice(list(largest))
            G.add_edge(node_from_component, node_from_largest)

    return G


def create_hybrid_topology(size):
    """
    Maintain original hybrid topology for compatibility, but use the
    more sophisticated realistic hybrid topology underneath.

    Returns:
        NetworkX graph with a hybrid topology
    """
    return create_realistic_hybrid_topology(size, seed=42)
