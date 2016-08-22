import networkx as nx
from collections import OrderedDict


class OrderedDigraph(nx.DiGraph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict
    edge_attr_dict_factory = OrderedDict


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


def break_strongly_connected(parent, broken_edges, scc):
    """
    Breaks strongly connected components. Called recursively until all such
    cycles are broken.
    
    Args
    ----
    parent : nx.DiGraph
        Directed graph from which the SCCs are drawn.

    broken_edges : list
        List to which broken edges are appended.

    scc : list
        List of nodes that make up a single SCC.
    """
    sgraph = parent.subgraph(scc)
    max_node = None
    max_score = -1
    in_smaller = False

    # Greedy Heuristic: look for the most asymmetrical (in terms of inputs vs
    # outputs) node and break the smallest set of connections for that node.
    for node in scc:
        din = sgraph.in_degree(node, weight='weight')
        dout = sgraph.out_degree(node, weight='weight')
        score = abs(din - dout)
        # Break ties lexicographically
        if max_node is None or score > max_score or \
                (score == max_score and node < max_node):
            max_node = node
            max_score = score
            in_smaller = din <= dout

    if in_smaller:
        for p in sgraph.predecessors(max_node):
            sgraph.remove_edge(p, max_node)
            parent.remove_edge(p, max_node)
            broken_edges.append((p, max_node))
    else:
        for s in sgraph.successors(max_node):
            sgraph.remove_edge(max_node, s)
            parent.remove_edge(max_node, s)
            broken_edges.append((max_node, s))

    # This subgraph is no longer strongly connected, but there may be such
    # components remaining.
    remaining_sccs = (s for s in nx.strongly_connected_components(sgraph)
                      if len(s) > 1)
    for sccs in remaining_sccs:
        break_strongly_connected(parent, broken_edges, sccs)
