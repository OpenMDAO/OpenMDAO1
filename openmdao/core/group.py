""" Defines the base class for a Group in OpenMDAO."""

from collections import OrderedDict

from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.core.varmanager import VarManager, ViewVarManager, create_views, \
                                      ViewTuple, get_relname_map
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
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
        self.nl_solver = NLGaussSeidel()

        # These point to (du,df) or (df,du) depending on mode.
        self.sol_vec = None
        self.rhs_vec = None

    def add(self, name, system, promotes=None):
        """Add a subsystem to this group, specifying its name and any variables
        that it promotes to the parent level.

        Parameters
        ----------
        name : str
            the name by which the subsystem is to be known

        system : `System`
            the subsystem to be added

        promotes : tuple, optional
            the names of variables in the subsystem which are to be promoted
        """
        if promotes is not None:
            system._promotes = promotes

        self._subsystems[name] = system
        system.name = name
        return system

    def connect(self, source, target):
        """Connect the given source variable to the given target
        variable.

        Parameters
        ----------
        source : source
            the name of the source variable

        target : str
            the name of the target variable
        """
        self._src[target] = source

    def subsystems(self, local=False):
        """ Returns an iterator over subsystems.

        local: bool
            Set to True to return only systems that are local.
        """
        if local == True:
            return self._local_subsystems.items()
        return self._subsystems.items()

    def subgroups(self):
        """ Returns
            -------
            iterator
                iterator over subgroups.
        """
        for name, subsystem in self._subsystems.items():
            if isinstance(subsystem, Group):
                yield name, subsystem

    def components(self):
        """ Returns
                -------
                iterator
                    iterator over sub-`Component`s.
            """
        for name, comp in self._subsystems.items():
            if isinstance(comp, Component):
                yield name, comp

    def _setup_variables(self):
        """Create dictionaries of metadata for parameters and for unknowns for
           this `Group` and stores them as attributes of the `Group'. The
           relative name of subsystem variables with respect to this `Group`
           system is included in the metadata.

           Returns
           -------
           tuple
               a dictionary of metadata for parameters and for unknowns
               for all subsystems
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
        """Returns
           -------
           str
               the pathname of the given variable, based on its promotion status.
        """
        if subsystem.promoted(name):
            return name
        if len(subsystem.name) > 0:
            return subsystem.name+':'+name
        else:
            return name

    def _setup_vectors(self, param_owners, connections, parent_vm=None):
        """Create a `VarManager` for this `Group` and all below it in the
        `System` tree.

        Parameters
        ----------
        param_owners : dict
            a dictionary mapping `System` pathnames to the pathnames of parameters
            they are reponsible for propagating

        connections : dict
            a dictionary mapping the pathname of a target variable to the
            pathname of the source variable that it is connected to

        parent_vm : `VarManager`, optional
            the `VarManager` for the parent `Group`, if any, into which this
            `VarManager` will provide a view.

        """
        my_params = param_owners.get(self.pathname, [])
        if parent_vm is None:
            self._varmanager = VarManager(self._params_dict, self._unknowns_dict,
                                         my_params, connections)
        else:
            self._varmanager = ViewVarManager(parent_vm,
                                             self.pathname,
                                             self._params_dict,
                                             self._unknowns_dict,
                                             my_params,
                                             connections)

        self._views = {}
        for name, sub in self.subgroups():
            sub._setup_vectors(param_owners, connections, parent_vm=self._varmanager)
            vm = sub._varmanager
            self._views[name] = ViewTuple(vm.unknowns, vm.dunknowns,
                                          vm.resids, vm.dresids,
                                          vm.params, vm.dparams)

        for name, sub in self.components():
            u, du, r, dr, p, dp = create_views(self._varmanager, sub.pathname,
                                               sub._params_dict, sub._unknowns_dict, [], {})
            relmap = get_relname_map(self._varmanager.params,
                                     sub._params_dict, name)
            self._views[name] = ViewTuple(u, du, r, dr,
                                          self._varmanager.params.get_view(relmap),
                                          self._varmanager.dparams.get_view(relmap))

    def _setup_paths(self, parent_path):
        """Set the absolute pathname of each `System` in the tree.

        Parameter
        ---------
        parent_path : str
            the pathname of the parent `System`, which is to be prepended to the
            name of this child `System` and all subsystems.
        """
        super(Group, self)._setup_paths(parent_path)
        for name, sub in self.subsystems():
            sub._setup_paths(self.pathname)

    def _get_explicit_connections(self):
        """ Returns
            -------
            dict
                explicit connections in this `Group`, represented as a mapping
                from the pathname of the target to the pathname of the source
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

        Parameters
        ----------
        params : `VecWrapper`
            ``VecWrapper` ` containing parameters (p)

        unknowns : `VecWrapper`
            `VecWrapper`  containing outputs and states (u)

        resids : `VecWrapper`
            `VecWrapper`  containing residuals. (r)
        """
        self.nl_solver.solve(params, unknowns, resids, self)

    def children_solve_nonlinear(self):
        """Loops over our children systems and asks them to solve."""

        varmanager = self._varmanager

        # TODO: Should be local subs only, but local dict isn't filled yet
        for name, system in self.subsystems():

            # Local scatter
            varmanager._transfer_data(name)

            view = self._views[system.name]

            params = view.params
            unknowns = view.unknowns
            resids = view.resids

            system.solve_nonlinear(params, unknowns, resids)

    def apply_nonlinear(self, params, unknowns, resids):
        """ Evaluates the residuals of our children systems.

        Parameters
        ----------
        params : `VecWrapper`
            ``VecWrapper` ` containing parameters (p)

        unknowns : `VecWrapper`
            `VecWrapper`  containing outputs and states (u)

        resids : `VecWrapper`
            `VecWrapper`  containing residuals. (r)
        """

        varmanager = self._varmanager

        # TODO: Should be local subs only, but local dict isn't filled yet
        for name, system in self.subsystems():

            # Local scatter
            varmanager._transfer_data(name)

            view = self._views[system.name]

            params = view.params
            unknowns = view.unknowns
            resids = view.resids

            system.apply_nonlinear(params, unknowns, resids)


def _get_implicit_connections(params_dict, unknowns_dict):
    """Finds all matches between relative names of parameters and
    unknowns.  Any matches imply an implicit connection.  All
    connections are expressed using absolute pathnames.

    This should only be called using params and unknowns from the
    top level `Group` in the system tree.

    Parameters
    ----------
    params_dict : dict
        dictionary of metadata for all parameters in this `Group`

    unknowns_dict : dict
        dictionary of metadata for all unknowns in this `Group`

    Returns
    -------
    dict
        implicit connections in this `Group`, represented as a mapping
        from the pathname of the target to the pathname of the source

    Raises
    ------
    RuntimeError
        if a a promoted variable name matches multiple unknowns
    """

    # collect all absolute names that map to each relative name
    abs_unknowns = {}
    for abs_name, u in unknowns_dict.items():
        abs_unknowns.setdefault(u['relative_name'], []).append(abs_name)

    abs_params = {}
    for abs_name, p in params_dict.items():
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
    """
       Parameters
       ----------
       var_name : str
           name of a variable relative to a `System`

       var_dict : dict
           dictionary of variable metadata, keyed on relative name

       Returns
       -------
       str
           the absolute pathname for the given variable in the
           variable dictionary
    """
    for pathname, meta in var_dict.items():
        if meta['relative_name'] == var_name:
            return pathname
    raise RuntimeError("Absolute pathname not found for %s" % var_name)
