""" Defines the base class for a Group in OpenMDAO."""

from collections import OrderedDict

from openmdao.core.system import System
from openmdao.core.varmanager import VarManager, VarViewManager

class Group(System):
    """A system that contains other systems"""

    def __init__(self):
        super(Group, self).__init__()

        self._subsystems = OrderedDict()
        self._local_subsystems = OrderedDict()
        self._src = {}

        # These point to (du,df) or (df,du) depending on mode.
        self.sol_vec = None
        self.rhs_vec = None

    def __contains__(self, name):
        return name in self._subsystems

    def add(self, name, system, promotes=None):
        if promotes is not None:
            system.promotes = promotes
        self._subsystems[name] = system
        system.name = name
        return system

    def connect(self, src, target):
        self._src[target] = src

    def subsystems(self):
        """ returns iterator over subsystems """
        return self._subsystems.items()

    def subgroups(self):
        """ returns iterator over subgroups """
        for name, subsystem in self._subsystems.items():
            if isinstance(subsystem, Group):
                yield name, subsystem

    def _setup_variables(self):
        """Return params and unknowns for all subsystems and stores them
        as attributes of the group"""
        # TODO: check for the same var appearing more than once in unknowns

        for name, sub in self.subsystems():
            subparams, subunknowns = sub._setup_variables()
            for p, meta in subparams.items():
                meta = meta.copy()
                meta['relative_name'] = self.var_pathname(meta['relative_name'], sub)
                self._params[p] = meta

            for u, meta in subunknowns.items():
                meta = meta.copy()
                meta['relative_name'] = self.var_pathname(meta['relative_name'], sub)
                self._unknowns[u] = meta

        return self._params, self._unknowns

    def var_pathname(self, name, subsystem):
        if subsystem.promoted(name):
            return name
        if len(subsystem.name) > 0:
            return subsystem.name+':'+name
        else:
            return name

    def _setup_vectors(self, param_owners, connections, parent_vm=None):
        my_params = param_owners.get(self.pathname, [])
        if parent_vm is None:
            self.varmanager = VarManager(self._params, self._unknowns,
                                         my_params, connections)
        else:
            self.varmanager = VarViewManager(parent_vm,
                                             self.pathname,
                                             self._params,
                                             self._unknowns,
                                             my_params,
                                             connections)

        for name, sub in self.subgroups():
            sub._setup_vectors(param_owners, connections, parent_vm=self.varmanager)

    def _setup_paths(self, parent_path):
        """Set the absolute pathname of each System in the
        tree.
        """
        super(Group, self)._setup_paths(parent_path)
        for name, sub in self.subsystems():
            sub._setup_paths(self.pathname)

    def _get_explicit_connections(self):
        """ Get all explicit connections stated with absolute pathnames
        """
        connections = {}
        for _, sub in self.subgroups():
            connections.update(sub._get_explicit_connections())

        for tgt, src in self._src.items():
            src_pathname = get_absvarpathname(src, self._unknowns)
            tgt_pathname = get_absvarpathname(tgt, self._params)
            connections[tgt_pathname] = src_pathname

        return connections

def _get_implicit_connections(params, unknowns):
    """Finds all matches between relative names of params and
    unknowns.  Any matches imply an implicit connection.

    This should only be called using params and unknowns from the
    top level Group in the system tree.
    """

    # collect all absolute names that map to each relative name
    abs_unknowns = {}
    for abs_name, u in unknowns.items():
        abs_unknowns.setdefault(u['relative_name'], []).append(abs_name)

    abs_params = {}
    for abs_name, p in params.items():
        abs_params.setdefault(p['relative_name'], []).append(abs_name)

    # check if any relative names correspond to mutiple unknowns
    for name, lst in abs_unknowns.items():
        if len(lst) > 1:
            raise RuntimeError("Promoted name %s matches multiple unknowns: %s" %
                               (name, lst))

    connections = {}
    for uname, uabs in abs_unknowns.items():
        pabs = abs_params.get(uname, ())
        for p in pabs:
            connections[p] = uabs[0]

    return connections

def get_absvarpathname(var_name, var_dict):
    """Returns the absolute pathname for the given relative variable
    name in the variable dictionary
    """
    for pathname, meta in var_dict.items():
        if meta['relative_name'] == var_name:
            return pathname
    raise RuntimeError("Absolute pathname not found for %s" % var_name)
