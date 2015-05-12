""" OpenMDAO Problem class defintion."""

import numpy as np

from openmdao.core.component import Component
from openmdao.core.driver import Driver
from openmdao.core.group import _get_implicit_connections

from collections import namedtuple

class ConnectError(Exception):
    @classmethod
    def type_mismatch_error(cls, src, target):
        msg = "Type '{src[type]}' of source '{src[relative_name]}' must be the same as type '{target[type]}' of target '{target[relative_name]}'"
        msg = msg.format(src=src, target=target)
        
        return cls(msg)
        
    @classmethod
    def shape_mismatch_error(cls, src, target):
        msg  = "Shape '{src[shape]}' of the source '{src[relative_name]}' must match the shape '{target[shape]}' of the target '{target[relative_name]}'"
        msg = msg.format(src=src, target=target)
        
        return cls(msg)
        
    @classmethod
    def val_and_shape_mismatch_error(cls, src, target):
        msg = "Shape of the initial value '{src[val].shape}' of source '{src[relative_name]}' must match the shape '{target[shape]}' of the target '{target[relative_name]}'"
        msg = msg.format(src=src, target=target)
        
        return cls(msg)
    
def __make_metadata(metadata):
    '''
    Add type field to metadata dict.
    Returns a modified copy of `metadata`.
    '''
    metadata = dict(metadata)
    metadata['type'] = type(metadata['val'])
        
    return metadata

def __get_metadata(paths, metadata_dict):
    metadata = []
    
    for path in paths:
        var_metadata = metadata_dict[path]
        metadata.append(__make_metadata(var_metadata))
        
    return metadata
    
    
def check_types_match(src, target):
    if src['type'] != target['type']:
        raise ConnectError.type_mismatch_error(src, target)

def check_connections(connections, params, unknowns):
    # Get metadata for all sources
    sources = __get_metadata(connections.values(), unknowns)
    
    #Get metadata for all targets
    targets = __get_metadata(connections.keys(), params)
    
    for source, target in zip(sources, targets):
        check_types_match(source, target)
        check_shapes_match(source, target)

def check_shapes_match(source, target):
    #Use the type of the shape of source and target to determine which the #correct function to use for shape checking
    
    check_shape_function = __shape_checks.get((type(source.get('shape')), type(target.get('shape'))), lambda x, y: None)
    
    check_shape_function(source, target)

def __check_shapes_match(src, target):
    if src['shape'] != target['shape']:
        raise ConnectError.shape_mismatch_error(src, target)

def __check_val_and_shape_match(src, target):
    if src['val'].shape != target['shape']:
        raise ConnectError.val_and_shape_mismatch_error(src, target)
        
__shape_checks = {
    (tuple, tuple) : __check_shapes_match,
    (type(None), tuple)  : __check_val_and_shape_match
}

class Problem(Component):
    """ The Problem is always the top object for running an OpenMDAO
    model.
    """

    def __init__(self, root=None, driver=None, impl=None):
        super(Problem, self).__init__()
        self.root = root
        self.impl = impl
        if driver is None:
            self.driver = Driver()
        else:
            self.driver = driver

    def __getitem__(self, name):
        """Retrieve unflattened value of named variable from the root system

        Parameters
        ----------
        name : str   OR   tuple : (name, vector)
             the name of the variable to retrieve from the unknowns vector OR
             a tuple of the name of the variable and the vector to get it's
             value from.

        Returns
        -------
        the unflattened value of the given variable
        """
        return self.root[name]

    def __setitem__(self, name, val):
        """Sets the given value into the appropriate `VecWrapper`.

        Parameters
        ----------
        name : str
             the name of the variable to set into the unknowns vector
        """
        self.root_varmanager.unknowns[name] = val

    def setup(self):
        """Performs all setup of vector storage, data transfer, etc.,
        necessary to perform calculations.
        """
        # Give every system an absolute pathname
        self.root._setup_paths(self.pathname)

        # Give every system a dictionary of parameters and of unknowns
        # that are visible to that system, keyed on absolute pathnames.
        # Metadata for each variable will contain the name of the
        # variable relative to that system as well as size and shape if
        # known.

        # Returns the parameters and unknowns dictionaries for the root.
        params_dict, unknowns_dict = self.root._setup_variables()

        # Get all explicit connections (stated with absolute pathnames)
        connections = self.root._get_explicit_connections()

        # go through relative names of all top level params/unknowns
        # if relative name in unknowns matches relative name in params
        # that indicates an implicit connection. All connections are returned
        # in absolute form.
        implicit_conns = _get_implicit_connections(params_dict, unknowns_dict)

        # check for conflicting explicit/implicit connections
        for tgt, src in connections.items():
            if tgt in implicit_conns:
                msg = '%s is explicitly connected to %s but implicitly connected to %s' % \
                      (tgt, connections[tgt], implicit_conns[tgt])
                raise RuntimeError(msg)

        # combine implicit and explicit connections
        connections.update(implicit_conns)

        check_connections(connections, params_dict, unknowns_dict)

        # check for parameters that are not connected to a source/unknown
        hanging_params = []
        for p in params_dict:
            if p not in connections.keys():
                hanging_params.append(p)

        if hanging_params:
            msg = 'Parameters %s have no associated unknowns.' % hanging_params
            raise RuntimeError(msg)

        # Given connection information, create mapping from system pathname
        # to the parameters that system must transfer data to
        param_owners = assign_parameters(connections)

        # create VarManagers and VecWrappers for all groups in the system tree.
        self.root._setup_vectors(param_owners, connections, impl=self.impl)

    def run(self):
        """ Runs the Driver in self.driver. """
        self.driver.run(self.root)

    def calc_gradient(self, param_list, unknown_list, mode='auto',
                      return_format='array'):
        """ Returns the gradient for the system that is slotted in
        self.root. This function is used by the optimizer, but also can be
        used for testing derivatives on your model.

        Parameters
        ----------
        param_list : list of strings (optional)
            List of parameter name strings with respect to which derivatives
            are desired. All params must have a paramcomp.

        unknown_list : list of strings (optional)
            List of output or state name strings for derivatives to be
            calculated. All must be valid unknowns in OpenMDAO.

        mode : string (optional)
            Deriviative direction, can be 'fwd', 'rev', or 'auto'.
            Default is 'auto', which uses mode specified on the linear solver
            in root.

        return_format : string (optional)
            Format for the derivatives, can be 'array' or 'dict'.

        Returns
        -------
        ndarray or dict
            Jacobian of unknowns with respect to params
        """

        if mode not in ['auto', 'fwd', 'rev']:
            msg = "mode must be 'auto', 'fwd', or 'rev'"
            raise ValueError(msg)

        if return_format not in ['array', 'dict']:
            msg = "return_format must be 'array' or 'dict'"
            raise ValueError(msg)

        # TODO Some of this stuff should go in the linearsolver, and some in
        # Group.

        root = self.root
        varmanager = root._varmanager
        params = varmanager.params
        unknowns = varmanager.unknowns
        n_edge = len(unknowns.vec)
        rhs = np.zeros((n_edge, ))

        # Linearize the model
        root.linearize(params, unknowns)

        # Initialized Jacobian
        if return_format == 'dict':
            J = {}
            for okey in unknown_list:
                J[okey] = {}
                for ikey in param_list:
                    if isinstance(ikey, tuple):
                        ikey = ikey[0]
                    J[okey][ikey] = None
        else:
            # TODO: need these functions
            num_input = system.get_size(param_list)
            num_output = system.get_size(unknown_list)
            J = np.zeros((num_output, num_input))

        # Respect choice of mode based on precedence.
        # Call arg > ln_solver option > auto-detect
        if mode =='auto':
            mode = root.ln_solver.options['mode']
            if mode == 'auto':
                # TODO: Choose based on size
                msg = 'Automatic mode selction not yet implemented.'
                raise NotImplementedError(msg)

        if mode == 'fwd':
            input_list, output_list = param_list, unknown_list
        else:
            input_list, output_list = unknown_list, param_list

        # If Forward mode, solve linear system for each param
        # If Adjoint mode, solve linear system for each unknown
        j = 0
        for param in input_list:

            in_idx = unknowns.get_idxs(param)
            jbase = j

            for irhs in in_idx:

                rhs[irhs] = 1.0

                # Call GMRES to solve the linear system
                dx = root.ln_solver.solve(rhs, root, mode)

                rhs[irhs] = 0.0

                i = 0
                for item in output_list:

                    out_idx = unknowns.get_idxs(item)
                    nk = len(out_idx)

                    if return_format == 'dict':
                        if mode == 'fwd':
                            if J[item][param] is None:
                                J[item][param] = np.zeros((nk, len(in_idx)))
                            J[item][param][:, j-jbase] = dx[out_idx]
                        else:
                            if J[param][item] is None:
                                J[param][item] = np.zeros((len(in_idx), nk))
                            J[param][item][j-jbase, :] = dx[out_idx]

                    else:
                        if mode == 'fwd':
                            J[i:i+nk, j] = dx[out_indices]
                        else:
                            J[j, i:i+nk] = dx[out_indices]
                        i += nk

                j += 1

        #print params, '\n', unknowns, '\n', J
        return J

def assign_parameters(connections):
    """Map absolute system names to the absolute names of the
    parameters they transfer data to.
    """
    param_owners = {}

    for par, unk in connections.items():
        common_parts = []
        for ppart, upart in zip(par.split(':'), unk.split(':')):
            if ppart == upart:
                common_parts.append(ppart)
            else:
                break

        owner = ':'.join(common_parts)
        param_owners.setdefault(owner, []).append(par)

    return param_owners
