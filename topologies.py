import networkx as nx


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
    """
    total_nodes = size * size

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

        G = nx.balanced_tree(branching_factor, depth)

        # If we have more nodes than needed, remove some leaf nodes
        if G.number_of_nodes() > total_nodes:
            # Get leaf nodes (nodes with degree 1)
            leaf_nodes = [n for n, d in G.degree() if d == 1]

            # Remove excess leaf nodes
            excess = G.number_of_nodes() - total_nodes
            G.remove_nodes_from(leaf_nodes[:excess])

        mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)

        root = 0
        center = coords_to_node(size // 2, size // 2, size)

        if root != center:
            # Swap root with center node
            mapping = {i: i for i in range(total_nodes)}
            mapping[root] = center
            mapping[center] = root
            G = nx.relabel_nodes(G, mapping)

        return G


def create_hybrid_topology(size):
    """
    Maintain original hybrid topology for compatibility, but use the
    more sophisticated realistic hybrid topology underneath.
    """
    full_size = size**2
    if full_size < 50:
        m = 2
    elif full_size < 200:
        m = 3
    else:
        m = 4

    p = 0.5

    G = nx.powerlaw_cluster_graph(full_size, m, p, seed=42)
    return G
