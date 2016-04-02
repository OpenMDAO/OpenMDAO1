
import os
import sys
import json
from itertools import chain

from six import iteritems

import webbrowser

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

def view_connections(system, viewer='connect_table',
                     outfile='connections.html', show_browser=True):
    """
    Generates a self-contained html file containing a connection viewer
    of the specified type.  Optionally pops up a web browser to
    view the file.

    Args
    ----
    system : system
        The root system for the desired tree.

    viewer : str, optional
        The type of web viewer used to view the connections.

    outfile : str, optional
        The name of the output html file.  Defaults to 'connections.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.
    """
    connections = system._probdata.connections
    to_prom = system._sysdata.to_prom_name
    src2tgts = {}
    units = {n: m.get('units','') for n,m in chain(iteritems(system._unknowns_dict),
                                                   iteritems(system._params_dict))}

    sizes = {}
    for t, (s, idxs) in iteritems(connections):
        if idxs is not None:
            sizes[t] = len(idxs)
        else:
            sizes[t] = system._params_dict[t]['size']
        if s not in src2tgts:
            src2tgts[s] = [t]
        else:
            src2tgts[s].append(t)

    src_groups = set()
    tgt_groups = set()
    for s in src2tgts:
        parts = s.split('.')
        for i in range(len(parts)):
            src_groups.add('.'.join(parts[:i]))

    for t in connections:
        parts = t.split('.')
        for i in range(len(parts)):
            tgt_groups.add('.'.join(parts[:i]))

    # reverse sort so that "NO CONNECTION" shows up at the bottom
    src2tgts['NO CONNECTION'] = sorted([t for t in to_prom
                                    if t not in system._unknowns_dict and
                                       t not in connections], reverse=True)

    src_groups = [{'name':n} for n in sorted(src_groups)]
    src_groups.insert(1, {'name': "NO CONNECTION"})
    tgt_groups = [{'name':n} for n in sorted(tgt_groups)]
    tgt_groups.insert(1, {'name': "NO CONNECTION"})

    data = {
        'src2tgts': [(s,ts) for s,ts in sorted(iteritems(src2tgts), reverse=True)],
        'proms': to_prom,
        'units': units,
        'sizes': sizes,
        'src_groups': src_groups,
        'tgt_groups': tgt_groups,
        'noconn_srcs': sorted((n for n in system._unknowns_dict
                               if n not in src2tgts), reverse=True),
    }

    viewer += '.html'

    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    graphjson = json.dumps(data)

    with open(outfile, 'w') as f:
        s = template.replace("<connection_data>", graphjson)
        f.write(s)

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
