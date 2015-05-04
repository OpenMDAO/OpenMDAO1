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

    def setup_variables(self):
        """Return params and unknowns for all subsystems and stores them
        as attributes of the group"""
        # TODO: check for the same var appearing more than once in unknowns

        comps = {}
        for name, sub in self.subsystems():
            subparams, subunknowns = sub.setup_variables()
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

    def setup_vectors(self, param_owners, connections, parent_vm=None):
        my_params = param_owners.get(self.pathname, [])
        if parent_vm is None:
            self.varmanager = VarManager(self._params, self._unknowns,
                                         my_params, connections)
        else:
            self.varmanager = VarViewManager(parent_vm,
                                             self.pathname,
                                             self.promotes,
                                             self._params,
                                             self._unknowns,
                                             my_params,
                                             connections)

        for name, sub in self.subgroups():
            sub.setup_vectors(param_owners, connections, parent_vm=self.varmanager)

    def setup_paths(self, parent_path):
        """Set the absolute pathname of each System in the
        tree.
        """
        super(Group, self).setup_paths(parent_path)
        for name, sub in self.subsystems():
            sub.setup_paths(self.pathname)

    def get_connections(self):
        """ Get all explicit connections stated with absolute pathnames
        """
        connections = {}
        for _, sub in self.subgroups():
            connections.update(sub.get_connections())

        for tgt, src in self._src.items():
            src_pathname = get_varpathname(src, self._unknowns)
            tgt_pathname = get_varpathname(tgt, self._params)
            connections[tgt_pathname] = src_pathname

        return connections

def get_varpathname(var_name, var_dict):
    """Returns the absolute pathname for the given relative variable
    name in the variable dictionary"""
    for pathname, meta in var_dict.items():
        if meta['relative_name'] == var_name:
            return pathname
    raise RuntimeError("Absolute pathname not found for %s" % var_name)

