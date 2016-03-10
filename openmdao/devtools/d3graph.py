
import os
import sys
import json

import webbrowser
from itertools import chain
from six import iteritems

import networkx as nx

from openmdao.core.component import Component

# default options for different viewers
viewer_options = {
    'collapse_tree': {
        'expand_level': 1,
    },
    'partition_tree': {
        'size_1': True,
    }
}

def _system_tree_dict(system, size_1=True, expand_level=9999):
    """
    Returns a dict representation of the system hierarchy with
    the given System as root.
    """

    def _tree_dict(ss, level):
        dct = { 'name': ss.name }
        children = [_tree_dict(s, level+1) for s in ss.subsystems()]

        if isinstance(ss, Component):
            for vname, meta in ss.unknowns.items():
                size = meta['size'] if meta['size'] and not size_1 else 1
                children.append({'name': vname, 'size': size })

            for vname, meta in ss.params.items():
                size = meta['size'] if meta['size'] and not size_1 else 1
                children.append({'name': vname, 'size': size })

        if level > expand_level:
            dct['_children'] = children
            dct['children'] = None
        else:
            dct['children'] = children
            dct['_children'] = None

        return dct

    tree = _tree_dict(system, 1)
    if not tree['name']:
        tree['name'] = 'root'

    return tree

def _create_vgraph(system):
    vgraph = nx.DiGraph()  # var graph

    compins = {}  # maps input vars to components
    compouts = {} # maps output vars to components

    for param, meta in iteritems(system._params_dict):
        tcomp = param.rsplit('.', 1)[0]
        compins.setdefault(tcomp, []).append(param)
        vgraph.add_node(param, name=param, key=param.rsplit('.',1)[1],
                        io='in')

    for unknown, meta in iteritems(system._unknowns_dict):
        scomp = unknown.rsplit('.', 1)[0]
        compouts.setdefault(scomp, []).append(unknown)
        vgraph.add_node(unknown, name=unknown,
                        key=unknown.rsplit('.',1)[1],
                        io='out')

    for target, (source, idxs) in iteritems(system.connections):
        vgraph.add_edge(source, target)

    # connect inputs to outputs on same component in order to fully
    # connect the variable graph.
    for comp, inputs in iteritems(compins):
        for inp in inputs:
            for out in compouts.get(comp, ()):
                vgraph.add_edge(inp, out, ignore=True)

    return vgraph

def _graph_tree_dict(system, recurse=False):
    """
    Returns a dict representation of the system hierarchy with
    the given System as root.
    """
    vgraph = _create_vgraph(system)

    visited = set()

    # dct = { '': { 'name': '', 'key': '', 'parent': None, 'children': [] }}
    # links = []
    srcs = {}

    for u, v, data in vgraph.edges_iter(data=True):
        if 'ignore' not in data:
            if u not in srcs:
                srcs[u] = {'name':u, 'size':1, 'imports':set([v])}
            else:
                srcs[u]['imports'].add(v)
            if v not in srcs:
                srcs[v] = {'name':v, 'size':1, 'imports':set()}

    classes = []
    for src, data in iteritems(srcs):
        classes.append({'name':data['name'], 'size':data['size'],
                         'imports': list(data['imports'])})

    return classes

    #for node, data in vgraph.nodes_iter(data=True):
        # dct[node] = data.copy()
        # del dct[node]['io']
        # parent = node.rsplit('.',1)[0]
        # if parent not in dct:
        #     key = parent.rsplit('.',1)
        #     key = key[1] if len(key) > 1 else key[0]
        #     dct[parent] = {
        #         'name': parent,
        #         'key': key,
        #         'children': [dct[node]]
        #     }
        # else:
        #     dct[parent]['children'].append(dct[node])
        #
        # if dct[parent] not in dct['']['children']:
        #     dct['']['children'].append(dct[parent])

    # for name, data in iteritems(dct):
    #     if name:
    #         if name in vgraph:
    #             data['parent'] = dct[name.rsplit('.',1)[0]]
    #         else:
    #             data['parent'] = dct['']
    #
    # for node, data in vgraph.nodes_iter(data=True):
    #     if recurse:
    #         if len(vgraph.pred[node]) == 0: # 0 in_degree
    #             for u, v in nx.dfs_edges(node):
    #                 ucomp = u.rsplit('.', 1)[0]
    #                 vcomp = v.rsplit('.', 1)[0]
    #                 if ucomp != vcomp:
    #                     pass
    #     else:
    #         for u,v,data in vgraph.edges_iter(node, data=True):
    #             if 'ignore' not in data:
    #                 links.append({'source':dct[u], 'target':dct[v]})
    #
    # return dct[''], links

def view_graph(system, viewer='edge_map',
               outfile='graph.html', show_browser=True):
    """
    Generates a self-contained html file containing a graph viewer
    of the specified type.  Optionally pops up a web browser to
    view the file.

    Args
    ----
    system : system
        The root system for the desired tree.

    viewer : str, optional
        The type of web viewer used to view the graph. Options are:
        hier_edge_bundling.

    outfile : str, optional
        The name of the output html file.  Defaults to 'graph.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.
    """
    connections = system._probdata.connections
    conns = [{'source':s, 'target':t} for t,(s,_) in iteritems(connections)]

    viewer += '.html'

    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    graphjson = json.dumps(conns)
    with open(outfile, 'w') as f:
        f.write(template % graphjson)

    if show_browser:
        webview(outfile)

def view_tree(system, viewer='collapse_tree', expand_level=9999,
              outfile='tree.html', show_browser=True):
    """
    Generates a self-contained html file containing a tree viewer
    of the specified type.  Optionally pops up a web browser to
    view the file.

    Args
    ----
    system : system
        The root system for the desired tree.

    viewer : str, optional
        The type of web viewer used to view the tree. Options are:
        collapse_tree and partition_tree.

    expand_level : int, optional
        Optionally set the level that the tree will initially be expanded to.
        This option currently only works with collapse_tree. If not set,
        the entire tree will be expanded.

    outfile : str, optional
        The name of the output html file.  Defaults to 'tree.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.
    """
    options = viewer_options[viewer]
    if 'expand_level' in options:
        options['expand_level'] = expand_level

    tree = _system_tree_dict(system, **options)
    viewer += '.template'

    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    treejson = json.dumps(tree)
    with open(outfile, 'w') as f:
        f.write(template % treejson)

    if show_browser:
        webview(outfile)


def webview(outfile):
    """pop up a web browser for the given file"""
    if sys.platform == 'darwin':
        os.system('open %s' % outfile)
    else:
        webbrowser.get().open(outfile)

def webview_argv():
    """This is tied to a console script called webview.  It just provides
    a convenient way to pop up a browser to view a specified html file(s).
    """
    for name in sys.argv[1:]:
        if os.path.isfile(name):
            webview(name)
