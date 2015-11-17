
import os
import sys
import webbrowser

from six import itertools

import networkx as nx

from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.group import Group
from openmdao.util.string_util import nearest_child
from six import itervalues

def plot_sys_tree(system, outfile=None, fmt='pdf'):
    """
    Generate a plot of the system tree and bring it up in a browser.
    (requires graphviz).

    Args
    ----
    system : `System`
        Starting node of the System tree to be plotted.

    outfile : str, optional
        Name of the output file.  Default is 'system_tree.<fmt>'

    fmt : str, optional
        Format for the plot file. Any format accepted by dot should work.
        Default is 'pdf'.
    """
    if outfile is None:
        outfile = 'system_tree.'+fmt

    dotfile = os.path.splitext(outfile)[0]+'.dot'

    _write_system_dot(system, dotfile)

    os.system("dot -T%s -o %s %s" % (fmt, outfile, dotfile))

    if sys.platform == 'darwin':
        os.system('open %s' % outfile)
    else:
        webbrowser.get().open(outfile)

    try:
        os.remove(dotfile)
    except:
        pass

def plot_vgraph(group, outfile=None, fmt='pdf'):
    """
    Generate a plot of the variable graph and bring it up in a browser.
    (requires graphviz).

    Args
    ----
    group : `Group`
        Only the part of the overall variable graph belonging to this group
        will be plotted.

    outfile : str, optional
        Name of the output file.  Default is 'graph.<fmt>'

    fmt : str, optional
        Format for the plot file. Any format accepted by dot should work.
        Default is 'pdf'.

    """
    _plot_graph(group._probdata.relevance._vgraph, outfile=outfile, fmt=fmt)

def plot_sgraph(group, outfile=None, fmt='pdf'):
    """
    Generate a plot of the system graph at a particular group level
    and bring it up in a browser.
    (requires graphviz).

    Args
    ----
    group : `Group`
        Only the part of the overall system graph belonging to this group
        will be plotted.

    outfile : str, optional
        Name of the output file.  Default is 'graph.<fmt>'

    fmt : str, optional
        Format for the plot file. Any format accepted by dot should work.
        Default is 'pdf'.

    """
    _plot_graph(group._get_sys_graph(), outfile=outfile, fmt=fmt)


def _write_node(f, meta, node, indent):
    assigns = ['%s=%s' % (k,v) for k,v in meta.items()]
    f.write('%s"%s" [%s];\n' % (' '*indent, node, ','.join(assigns)))

def _write_system_dot(system, dotfile):
    # first, create a mapping of unique names to each system

    with open(dotfile, 'w') as f:
        indent = 3

        f.write("strict digraph {\n")

        meta = {
            'shape': _dot_shape(system),
            'label': '"' + system.name + '"'
        }
        _write_node(f, meta, system.pathname, indent)

        _sys_dot(system, indent, f)

        f.write("}\n")

def _dot_shape(system):
    if isinstance(system, ParallelGroup):
        return "parallelogram"
    elif isinstance(system, Group):
        return "rectangle"
    return "ellipse"

def _sys_dot(system, indent, f):

    for i, s in enumerate(itervalues(system._subsystems)):
        meta = {
            'shape': _dot_shape(s),
            'label': '"' + s.name + '"'
        }
        _write_node(f, meta, s.pathname, indent)
        f.write('%s"%s" -> "%s" [label="%d"];\n' %
                        (' '*indent, system.pathname, s.pathname, i))
        _sys_dot(s, indent+3, f)

def _plot_graph(G, outfile=None, fmt='pdf'):
    """Create a plot of the given graph"""

    if outfile is None:
        outfile = 'graph.'+fmt

    dotfile = os.path.splitext(outfile)[0]+'.dot'

    nx.write_dot(G, dotfile)

    os.system("dot -T%s -o %s %s" % (fmt, outfile, dotfile))

    if sys.platform == 'darwin':
        os.system('open %s' % outfile)
    else:
        webbrowser.get().open(outfile)

    os.remove(dotfile)
