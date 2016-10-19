
import os
import pickle
import json
from six import iteritems
import networkx as nx
from collections import OrderedDict

from sqlitedict import SqliteDict

try:
    import h5py
except ImportError:
    # Necessary for the file to parse
    h5py = None

from openmdao.core.component import Component
from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.core.mpi_wrap import MPI
from openmdao.util.record_util import is_valid_sqlite3_db
import base64


def _system_tree_dict(system, component_execution_orders):
    """
    Returns a dict representation of the system hierarchy with
    the given System as root.
    """

    def _tree_dict(ss, component_execution_orders, component_execution_index):
        subsystem_type = 'group'
        if isinstance(ss, Component):
            subsystem_type = 'component'
            component_execution_orders[ss.pathname] = component_execution_index[0]
            component_execution_index[0] += 1
        dct = OrderedDict([ ('name', ss.name), ('type', 'subsystem'), ('subsystem_type', subsystem_type)])

        #local_prom_dict = {}
        #for output_path, output_prom_name in iteritems(ss._sysdata.to_prom_name):
        #    if "." in output_prom_name:
        #        local_prom_dict[output_path] = output_prom_name
        #if(len(local_prom_dict) > 0):
        #    dct['promotions'] = local_prom_dict

        children = [_tree_dict(s, component_execution_orders, component_execution_index) for s in ss.subsystems()]

        if isinstance(ss, Component):
            if ss.is_active():
                my_chlist = []
                for vname, meta in ss.params.items():
                    dtype=type(meta['val']).__name__
                    my_chlist.append(OrderedDict([('name', vname), ('type', 'param'), ('dtype', dtype)]))

                for vname, meta in ss.unknowns.items():
                    dtype=type(meta['val']).__name__
                    implicit = False
                    if meta.get('state'):
                        implicit = True
                    my_chlist.append(OrderedDict([('name', vname), ('type', 'unknown'), ('implicit', implicit), ('dtype', dtype)]))
            else:
                my_chlist = []   # just make an empty list

            if MPI:
                chlist = system.comm.gather(my_chlist, root=0)

                # now in rank 0, just use the first non-empty entry in the list
                if system.comm.rank == 0 :
                    for vars_on_rank in chlist:
                        if vars_on_rank:
                            children.extend(vars_on_rank)
                            break
            else:
                children.extend(my_chlist)

        dct['children'] = children

        return dct

    component_execution_idx = [0] #list so pass by ref
    tree = _tree_dict(system, component_execution_orders, component_execution_idx)
    if not tree['name']:
        tree['name'] = 'root'
        tree['type'] = 'root'

    return tree

def get_model_viewer_data(problem_or_rootgroup):
    """Gets the data needed for generating the n2 partition tree diagram.

    Args
    ----
    problem_or_rootgroup : Problem or Group
        Problem or root Group used to get graph and connections info

    """

    if isinstance(problem_or_rootgroup, Problem):
        root_group = problem_or_rootgroup.root
    elif isinstance(problem_or_rootgroup, Group):
        if not problem_or_rootgroup.pathname: # root group
            root_group = problem_or_rootgroup
        else:
            # this function only makes sense when it is at the root
            return {}
    else:
        raise TypeError('get_model_viewer_data only accepts Problems or Groups')

    data_dict = {}
    component_execution_orders = {}
    data_dict['tree'] = _system_tree_dict(root_group, component_execution_orders)

    connections_list = []
    G = root_group._probdata.relevance._sgraph
    scc = nx.strongly_connected_components(G)
    scc_list = [s for s in scc if len(s)>1] #list(scc)

    for tgt, (src, idxs) in iteritems(root_group._probdata.connections):
        src_subsystem = src.rsplit('.', 1)[0]
        tgt_subsystem = tgt.rsplit('.', 1)[0]

        count = 0
        edges_list = []
        for li in scc_list:
            if src_subsystem in li and tgt_subsystem in li:
                count = count+1
                if(count > 1):
                    raise ValueError('Count greater than 1')

                exe_tgt = component_execution_orders[tgt_subsystem]
                exe_src = component_execution_orders[src_subsystem]
                exe_low = min(exe_tgt,exe_src)
                exe_high = max(exe_tgt,exe_src)
                subg = G.subgraph(li)
                for n in subg.nodes():
                    exe_order = component_execution_orders[n]
                    if(exe_order < exe_low or exe_order > exe_high):
                        subg.remove_node(n)


                src_to_tgt_str = src_subsystem + ' ' + tgt_subsystem
                for tup in subg.edges():
                    edge_str = tup[0] + ' ' + tup[1]
                    if edge_str != src_to_tgt_str:
                        edges_list.append(edge_str)


        if(len(edges_list) > 0):
            edges_list.sort() # make deterministic so same .html file will be produced each run
            connections_list.append(OrderedDict([('src', src), ('tgt', tgt), ('cycle_arrows', edges_list)]))
        else:
            connections_list.append(OrderedDict([('src', src), ('tgt', tgt)]))

    data_dict['connections_list'] = connections_list

    return data_dict


def view_tree(*args, **kwargs):
    """
    view_tree was renamed to view_model, but left here for backwards compatibility
    """
    import warnings
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn("view_tree is deprecated. Please switch to view_model.", DeprecationWarning, stacklevel=2)
    warnings.simplefilter('ignore', DeprecationWarning)
    view_model(*args, **kwargs)

def view_model(problem_or_filename, outfile='partition_tree_n2.html', show_browser=True, offline=True, embed=False):
    """
    Generates a self-contained html file containing a tree viewer
    of the specified type.  Optionally pops up a web browser to
    view the file.

    Args
    ----
    problem_or_filename : Either a Problem() or a string
        Problem() : The Problem (after problem.setup()) for the desired tree.
        string : The filename of the case recorder file containing the data required to build the tree.

    outfile : str, optional
        The name of the output html file.  Defaults to 'partition_tree_n2.html'.

    show_browser : bool, optional
        If True, pop up the system default web browser to view the generated html file.
        Defaults to True.

    offline : bool, optional
        If True, embed the javascript d3 library into the generated html file so that the tree can be viewed
        offline without an internet connection.  Otherwise if False, have the html request the latest d3 file
        from https://d3js.org/d3.v4.min.js when opening the html file.
        Defaults to True.

    embed : bool, optional
        If True, export only the innerHTML that is between the body tags, used for embedding the viewer into another html file.
        If False, create a standalone HTML file that has the DOCTYPE, html, head, meta, and body tags.
        Defaults to False.
    """
    viewer = 'partition_tree_n2.template'

    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    html_begin_tags = ("<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "    <meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">\n"
        "</head>\n"
        "<body>\n")

    html_end_tags = ("</body>\n"
        "</html>\n")

    display_none_attr = ""

    if embed:
        html_begin_tags = html_end_tags = ""
        display_none_attr = " style=\"display:none\""

    d3_library = "<script src=\"https://d3js.org/d3.v4.min.js\" charset=\"utf-8\"></script>"
    if offline:
        with open(os.path.join(code_dir, 'd3.v4.min.js'), "r") as f:
            d3_library = "<script type=\"text/javascript\"> %s </script>" % (f.read())

    with open(os.path.join(code_dir, "fontello.woff"), "rb") as f:
        encoded_font = str(base64.b64encode(f.read()).decode("ascii"))

    with open(os.path.join(code_dir, 'awesomplete.css'), "r") as f:
            awesomplete_css = "%s" % (f.read())

    with open(os.path.join(code_dir, 'awesomplete.js'), "r") as f:
            awesomplete_js = "%s" % (f.read())

    if isinstance(problem_or_filename, Problem):
        model_viewer_data = get_model_viewer_data(problem_or_filename)
    else:
        # Do not know file type. Try opening to see what works
        file_type = None
        if is_valid_sqlite3_db(problem_or_filename):
            db = SqliteDict(filename=problem_or_filename, flag='r', tablename='metadata')
            file_type = "sqlite"
        else:
            try:
                hdf = h5py.File(problem_or_filename, 'r')
                file_type = 'hdf5'
            except:
                raise ValueError("The given filename is not one of the supported file formats: sqlite or hdf5")

        if file_type == "sqlite":
            model_viewer_data = db['model_viewer_data']
        elif file_type == "hdf5":
            metadata = hdf.get('metadata', None)
            model_viewer_data = pickle.loads(metadata.get('model_viewer_data').value)


    tree_json = json.dumps(model_viewer_data['tree'])
    conns_json = json.dumps(model_viewer_data['connections_list'])

    with open(outfile, 'w') as f:
        f.write(template % (html_begin_tags, awesomplete_css, encoded_font, display_none_attr, d3_library, awesomplete_js, tree_json, conns_json, html_end_tags))

    if show_browser:
        from openmdao.devtools.webview import webview
        webview(outfile)
