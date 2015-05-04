""" Defines the Problem class in OpenMDAO."""

from openmdao.core.component import Component
from openmdao.core.group import _get_implicit_connections

class Problem(Component):
    """ The Problem is always the top object for running an OpenMDAO
    model."""

    def __init__(self, root=None, driver=None):
        super(Problem, self).__init__()
        self.root = root
        self.driver = driver

    def setup(self):
        # Give every system an absolute pathname
        self.root._setup_paths(self.pathname)

        # Give every system a dictionary of parameters and of unknowns
        # that are visible to that system, keyed on absolute pathnames.
        # Metadata for each variable will contain the name of the
        # variable relative to that system.
        # Returns the parameters and unknowns dictionaries for the root.
        params_dict, unknowns_dict = self.root._setup_variables()

        # Get all explicit connections (stated with absolute pathnames)
        connections = self.root._get_explicit_connections()

        # go through relative names of all top level params/unknowns
        # if relative name in unknowns matches relative name in params
        # that indicates an implicit connection
        # make those names absolute and add to connections
        implicit_conns = _get_implicit_connections(params_dict, unknowns_dict)

        # check for conflicting explicit/implicit connections
        for tgt, src in connections.items():
            if tgt in implicit_conns:
                msg = '%s is explicitly connected to %s but implicitly connected to %s' % \
                      (tgt, connections[tgt], implicit_conns[tgt])
                raise RuntimeError(msg)

        connections.update(implicit_conns)

        # check for parameters that are not connected to a source/unknown
        hanging_params = []
        for p in params_dict:
            if p not in connections.keys():
                hanging_params.append(p)

        if hanging_params:
            msg = 'Parameters %s have no associated unknowns.' % hanging_params
            raise RuntimeError(msg)

        # Given connection information, create mapping from system pathname
        # to the parameters that system must perform scatters to
        param_owners = assign_parameters(connections)

        # create VarManagers and vecWrappers for all groups in the system tree.
        self.root._setup_vectors(param_owners, connections)

    def run(self):
        pass

    def calc_gradient(self, params, unknowns, mode='auto',
                      return_format='array'):
        """ Returns the gradient for the system that is slotted in
        self.root. This function is used by the optimizer, but also can be
        used for testing derivatives on your model.

        params: list of strings (optional)
            List of parameter name strings with respect to which derivatives
            are desired. All params must have a paramcomp.

        unknowns: list of strings (optional)
            List of output or state name strings for derivatives to be
            calculated. All must be valid unknowns in OpenMDAO.

        mode: string (optional)
            Deriviative direction, can be 'fwd', 'rev', or 'auto'.
            Default is 'auto', which uses mode specified on the linear solver
            in root.

        return_format: string (optional)
            Format for the derivatives, can be 'array' or 'dict'.
        """

        if mode not in ['auto', 'fwd', 'rev']:
            msg = "mode must be 'auto', 'fwd', or 'rev'"
            raise ValueError(msg)

        if return_format not in ['array', 'dict']:
            msg = "return_format must be 'array' or 'dict'"
            raise ValueError(msg)

        # Here, we will assemble right hand sides and call solve_linear on the
        # system in root for each of them.

        pass

def assign_parameters(connections):
    """Map absolute system names to the absolute names of the
    parameters they control
    """
    param_owners = {}

    for par, unk in connections.items():
        par_parts = par.split(':')
        unk_parts = unk.split(':')

        common_parts = []
        i = 0
        while(par_parts[i] == unk_parts[i]):
            common_parts.append(par_parts[i])
            i = i+1
        owner = ':'.join(common_parts)

        if owner in param_owners:
            param_owners[owner].append(par)
        else:
            param_owners[owner] = [par]

    return param_owners
