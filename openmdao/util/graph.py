import networkx as nx

def collapse_nodes(graph, node_map, copy=False):
    """
    Args
    ----
    graph : nx.DiGraph
        Graph with nodes we want to collapse.

    node_map : dict
        A map of existing node names to collapsed names.

    copy : bool(False)
        If True, copy the graph before collapsing the nodes.

    Returns
    -------
    nx.DiGraph
        The graph with the nodes collapsed.

    """
    graph = nx.relabel_nodes(graph, node_map, copy=copy)

    # remove any self edges created by the relabeling
    graph.remove_edges_from([(u, v) for u, v in graph.edges_iter()
                             if u == v])

    return graph
