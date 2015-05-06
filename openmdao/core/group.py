""" Defines the base class for a Group in OpenMDAO."""

from collections import OrderedDict

from openmdao.core.system import System
from openmdao.core.varmanager import VarManager, VarViewManager
from openmdao.solvers.nl_gauss_seidel import NL_Gauss_Seidel
from openmdao.solvers.scipy_gmres import ScipyGMRES

class Group(System):
    """A system that contains other systems"""

    def __init__(self):
        super(Group, self).__init__()

        self._subsystems = OrderedDict()
        self._local_subsystems = OrderedDict()
        self._src = {}

        # These solvers are the default
        self.ln_solver = ScipyGMRES()
        self.nl_solver = NL_Gauss_Seidel()

        # These point to (du,df) or (df,du) depending on mode.
        self.sol_vec = None
        self.rhs_vec = None

    def add(self, name, system, promotes=None):
        """Add a subsystem to this group, specifying its name and any variables
        that it promotes to the parent level.
        """
        if promotes is not None:
            system._promotes = promotes
        self._subsystems[name] = system
        system.name = name
        return system

    def connect(self, src, target):
        """Connect the given source variable to the given target
        variable.
        """
        self._src[target] = src

    def subsystems(self, local=False):
        """ Returns an iterator over subsystems.

        local: bool
            Set to True to return only systems that are local.
        """
        if local == True:
            return self._local_subsystems.items()
        return self._subsystems.items()

    def subgroups(self):
        """ Returns an iterator over subgroups. """
        for name, subsystem in self._subsystems.items():
            if isinstance(subsystem, Group):
                yield name, subsystem

    def _setup_variables(self):
        """Return params and unknowns for all subsystems and stores them
        as attributes of the group
        """

        for name, sub in self.subsystems():
            subparams, subunknowns = sub._setup_variables()
            for p, meta in subparams.items():
                meta = meta.copy()
                meta['relative_name'] = self._var_pathname(meta['relative_name'], sub)
                self._params_dict[p] = meta

            for u, meta in subunknowns.items():
                meta = meta.copy()
                meta['relative_name'] = self._var_pathname(meta['relative_name'], sub)
                self._unknowns_dict[u] = meta

        return self._params_dict, self._unknowns_dict

    def _var_pathname(self, name, subsystem):
        """Return the name of the given variable, based on its
        promotion status.
        """
        if subsystem.promoted(name):
            return name
        if len(subsystem.name) > 0:
            return subsystem.name+':'+name
        else:
            return name

    def _setup_vectors(self, param_owners, connections, parent_vm=None):
        """Create a VarManager for this Group and all below it in the System
        tree, along with their internal VecWrappers.
        """
        my_params = param_owners.get(self.pathname, [])
        if parent_vm is None:
            self.varmanager = VarManager(self._params_dict, self._unknowns_dict,
                                         my_params, connections)
        else:
            self.varmanager = VarViewManager(parent_vm,
                                             self.pathname,
                                             self._params_dict,
                                             self._unknowns_dict,
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
            src_pathname = get_absvarpathname(src, self._unknowns_dict)
            tgt_pathname = get_absvarpathname(tgt, self._params_dict)
            connections[tgt_pathname] = src_pathname

        return connections

    def solve_nonlinear(self, params, unknowns, resids):
        """Solves the group using the slotted nl_solver.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)

        resids: vecwrapper
            VecWrapper containing residuals. (r)
        """
        self.nl_solver.solve(params, unknowns, resids, self)

    def children_solve_nonlinear(self):
        """Loops over our children systems and asks them to solve."""

        varmanager = self.varmanager

        # TODO: Should be local subs only, but local dict isn't filled yet
        for system in self.subsystems():

            # Local scatter
            varmanager._scatter('u', 'p', system.name)

            # TODO: We need subviews of the vecwrappers
            params = varmanager.params
            unknowns = varmanager.unknowns
            resids = varmanager.resids

            system.solve_nonlinear(params, unknowns, resids)

def _get_implicit_connections(params, unknowns):
    """Finds all matches between relative names of params and
    unknowns.  Any matches imply an implicit connection.  All
    connections are expressed using absolute pathnames.

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
