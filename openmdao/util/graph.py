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

# plain_bfs is taken from networkx, but it isn't present in all versions,
# so putting it here to make sure it's available.
#
#    Copyright (C) 2004-2015 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
def plain_bfs(G, source):
    """A fast BFS node generator

    The direction of the edge between nodes is ignored.

    For directed graphs only.

    """
    Gsucc = G.succ
    Gpred = G.pred

    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(Gsucc[v])
                nextlevel.update(Gpred[v])
