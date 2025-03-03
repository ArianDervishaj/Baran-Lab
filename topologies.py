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


# Basic Topology Generators


def create_line_topology(size):
    """
    Create a line topology that follows a snake-like path through the grid.

    Returns:
        NetworkX graph with a line topology
    """
    G = init_graph(size)

    # Create a snake-like path through the grid
    path = []
    for i in range(size):
        # For even rows, go left to right
        if i % 2 == 0:
            row_nodes = [coords_to_node(i, j, size) for j in range(size)]
        # For odd rows, go right to left
        else:
            row_nodes = [coords_to_node(i, j, size) for j in range(size - 1, -1, -1)]
        path.extend(row_nodes)

    # Connect nodes along the path
    for i in range(len(path) - 1):
        G.add_edge(path[i], path[i + 1])

    return G


def create_ring_topology(size):
    """
    Create a ring topology that connects all nodes in a cycle.

    Returns:
        NetworkX graph with a ring topology
    """
    G = create_line_topology(size)

    # Connect the last node back to the first
    first_node = 0

    # Determine last node based on grid size
    if size % 2 == 0:
        # For even sized grids, we end at the bottom-left
        last_node = coords_to_node(size - 1, 0, size)
    else:
        # For odd sized grids, we end at the bottom-right
        last_node = coords_to_node(size - 1, size - 1, size)

    G.add_edge(last_node, first_node)
    return G


def create_grid_topology(size):
    """
    Create a standard grid topology with vertical and horizontal connections.

    Returns:
        NetworkX graph with a grid topology
    """
    # Create a grid using NetworkX's built-in function
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
            G.add_edge(coords_to_node(i, j, size), coords_to_node(i + 1, j + 1, size))

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
            G.add_edge(coords_to_node(i, j, size), coords_to_node(i + 1, j - 1, size))

    return G


def create_full_mesh_topology(size):
    """
    Create a fully connected mesh where every node connects to every other node.

    Returns:
        NetworkX graph with a fully-connected topology
    """
    G = init_graph(size)

    # Connect every node to every other node
    total_nodes = size * size
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            G.add_edge(i, j)

    return G


def create_star_topology(size):
    """
    Create a star topology with a center node connected to all other nodes.

    Returns:
        NetworkX graph with a star topology
    """
    G = init_graph(size)

    # Use the center of the grid as the central node
    center_node = coords_to_node(size // 2, size // 2, size)

    # Connect center to all other nodes
    for node in range(size * size):
        if node != center_node:
            G.add_edge(center_node, node)

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


def create_hybrid_topology(size):
    """
    Create a hybrid topology with four quadrants of different topologies.

    The quadrants are:
    - Upper Left: Tree topology
    - Upper Right: Full mesh topology
    - Lower Left: Grid topology
    - Lower Right: Star topology

    !!!! Size should be even

    Returns:
        NetworkX graph with a hybrid topology
    """
    G = init_graph(size)
    half_size = size // 2

    # Create mapping functions for the quadrants
    def _map_to_quadrant(local_i, local_j, quadrant):
        """Map local coordinates to global coordinates based on quadrant"""
        if quadrant == 0:  # Upper left
            return coords_to_node(local_i, local_j, size)
        elif quadrant == 1:  # Upper right
            return coords_to_node(local_i, local_j + half_size, size)
        elif quadrant == 2:  # Lower right
            return coords_to_node(local_i + half_size, local_j + half_size, size)
        elif quadrant == 3:  # Lower left
            return coords_to_node(local_i + half_size, local_j, size)

    def _map_from_id(local_id, quadrant):
        """Map local node ID to global node ID based on quadrant"""
        local_i = local_id // half_size
        local_j = local_id % half_size
        return _map_to_quadrant(local_i, local_j, quadrant)

    # Generate topologies for each quadrant
    quadrant_topologies = [
        # Upper Left: Tree
        create_tree_topology(half_size, branching_factor=3),
        # Upper Right: Mesh
        create_full_mesh_topology(half_size),
        # Lower Right: Star
        create_star_topology(half_size),
        # Lower Left: Grid
        create_grid_topology(half_size),
    ]

    # Add edges from each quadrant to the main graph
    for quadrant, topology in enumerate(quadrant_topologies):
        for u, v in topology.edges():
            G.add_edge(_map_from_id(u, quadrant), _map_from_id(v, quadrant))

    # Define key connection points between quadrants
    tree_root = _map_to_quadrant(half_size // 2, half_size // 2, 0)
    mesh_node_near_tree = _map_to_quadrant(half_size // 2, 0, 1)
    mesh_node_near_grid = _map_to_quadrant(half_size - 1, 0, 1)
    mesh_node_near_star = _map_to_quadrant(half_size - 1, half_size - 1, 1)
    grid_node = _map_to_quadrant(0, half_size // 2, 3)
    star_center = _map_from_id(
        coords_to_node(half_size // 2, half_size // 2, half_size), 2
    )

    # Connect the quadrants
    G.add_edge(tree_root, mesh_node_near_tree)
    G.add_edge(grid_node, mesh_node_near_grid)
    G.add_edge(star_center, mesh_node_near_star)

    return G
