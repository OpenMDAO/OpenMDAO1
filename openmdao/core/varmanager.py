
from collections import namedtuple
import numpy
from openmdao.core.vecwrapper import VecWrapper
from openmdao.core.dataxfer import DataXfer

ViewTuple = namedtuple('ViewTuple', 'unknowns, dunknowns, resids, dresids, params, dparams')

class VarManagerBase(object):
    """Base class for a manager of the data transfer of a possibly distributed
    collection of variables.

    Parameters
    ----------
        connections : dict
            a dictionary mapping the pathname of a target variable to the
            pathname of the source variable that it is connected to
    """
    def __init__(self, connections):
        self.connections = connections
        self.params    = None
        self.dparams   = None
        self.unknowns  = None
        self.dunknowns = None
        self.resids    = None
        self.dresids   = None
        self.data_xfer = {}

    def _setup_data_transfer(self, my_params):
        """Create `DataXfer` objects to handle data transfer for all of the
           connections that involve paramaters for which this `VarManager`
           is responsible.

           Parameters
           ----------
           my_params : list
               list of pathnames for parameters that the VarManager is
               responsible for propagating
        """

        # collect all flattenable var sizes from self.unknowns
        flats = [m['size'] for m in self.unknowns.values()
                     if not m.get('noflat')]

        # create a 1x<num_flat_vars> numpy array with the sizes of each var
        self._local_sizes = numpy.array([[flats]])

        # we would do an Allgather of the local_sizes in the distributed case so all
        # processes would know the sizes of all variables (needed to determine distributed
        # indices)

        #TODO: invesigate providing enough system info here to determine what types of scatters
        # are necessary (for example, full scatter isn't needed except when solving using jacobi,
        # so why allocate space for the index arrays?)

        xfer_dict = {}
        for param, unknown in self.connections.items():
            if param in my_params:
                dest_comp = param.split(':',1)[0]
                src_idx_list, dest_idx_list, noflat_conns = xfer_dict.setdefault(dest_comp, ([],[],[]))
                urelname = self.unknowns.get_relative_varname(unknown)
                prelname = self.params.get_relative_varname(param)
                noflat = self.unknowns.metadata(urelname)[0].get('noflat')
                if noflat:
                    noflat_conns.append(prelname, urelname)
                else:
                    src_idx_list.append(self.unknowns.get_idxs(urelname))
                    dest_idx_list.append(self.params.get_idxs(prelname))

        for tgt_comp, (srcs, tgts, noflat_conns) in xfer_dict.items():
            src_idxs, tgt_idxs = self.unknowns.merge_idxs(srcs, tgts)
            self.data_xfer[tgt_comp] = DataXfer(src_idxs, tgt_idxs, noflat_conns)

        #TODO: create a jacobi DataXfer object (if necessary) that combines all of the
        #      individual subsystem src_idxs, tgt_idxs, and noflat_conns

    def _transfer_data(self, target_system, mode='fwd', deriv=False):
        """Transfer data to/from target_system depending on mode.

        Parameters
        ----------
        target_system : str
            Name of the target `System`.

        mode : { 'fwd', 'rev' }, optional
            Specifies forward or reverse data transfer.

        deriv : bool, optional
            If True, perform a data transfer between derivative `VecWrapper`s
        """
        x = self.data_xfer.get(target_system)
        if x is not None:
            if deriv:
                x.transfer(self.dunknowns, self.dparams, mode)
            else:
                x.transfer(self.unknowns, self.params, mode)


class VarManager(VarManagerBase):
    """A manager of the data transfer of a possibly distributed
    collection of variables.

    Parameters
    ----------
    params_dict : dict
        dictionary of metadata for all parameters

    unknowns_dict : dict
        dictionary of metadata for all unknowns

    my_params : list
        list of pathnames for parameters that this `VarManager` is
        responsible for propagating

    connections : dict
        a dictionary mapping the pathname of a target variable to the
        pathname of the source variable that it is connected to
    """
    def __init__(self, params_dict, unknowns_dict, my_params, connections):
        super(VarManager, self).__init__(connections)

        self.unknowns  = VecWrapper.create_source_vector(unknowns_dict, store_noflats=True)
        self.dunknowns = VecWrapper.create_source_vector(unknowns_dict)
        self.resids    = VecWrapper.create_source_vector(unknowns_dict)
        self.dresids   = VecWrapper.create_source_vector(unknowns_dict)
        self.params    = VecWrapper.create_target_vector(params_dict, self.unknowns, my_params, connections, store_noflats=True)
        self.dparams   = VecWrapper.create_target_vector(params_dict, self.unknowns, my_params, connections)

        self._setup_data_transfer(my_params)

class ViewVarManager(VarManagerBase):
    """A manager of the data transfer of a possibly distributed collection of
    variables.  The variables are based on views into an existing VarManager.

    Parameters
    ----------
    parent_vm : `VarManager`
        the `VarManager` which provides the `VecWrapper`s on which to create views

    params_dict : dict
        dictionary of metadata for all parameters

    unknowns_dict : dict
        dictionary of metadata for all unknowns

    my_params : list
        list of pathnames for parameters that this `VarManager` is
        responsible for propagating

    connections : dict
        a dictionary mapping the pathname of a target variable to the
        pathname of the source variable that it is connected to
    """
    def __init__(self, parent_vm, sys_pathname, params_dict, unknowns_dict, my_params, connections):
        super(ViewVarManager, self).__init__(connections)

        self.unknowns, self.dunknowns, self.resids, self.dresids, self.params, self.dparams = \
            create_views(parent_vm, sys_pathname, params_dict, unknowns_dict, my_params, connections)

        self._setup_data_transfer(my_params)


def create_views(parent_vm, sys_pathname, params_dict, unknowns_dict, my_params, connections):
    """A manager of the data transfer of a possibly distributed collection of
    variables.  The variables are based on views into an existing VarManager.

    Parameters
    ----------
    parent_vm : `VarManager`
        the `VarManager` which provides the `VecWrapper`s on which to create views

    sys_pathname : str
        pathname of the system for which the views are being created

    params_dict : dict
        dictionary of metadata for all parameters

    unknowns_dict : dict
        dictionary of metadata for all unknowns

    my_params : list
        list of pathnames for parameters that this `VarManager` is
        responsible for propagating

    connections : dict
        a dictionary mapping the pathname of a target variable to the
        pathname of the source variable that it is connected to

    Returns
    -------
    `ViewTuple`
        a namedtuple of six (6) `VecWrapper`s:
        unknowns, dunknowns, resids, dresids, params, dparams
    """

    # map relative name in parent to corresponding relative name in this view
    umap = get_relname_map(parent_vm.unknowns, unknowns_dict, sys_pathname)

    unknowns  = parent_vm.unknowns.get_view(umap)
    dunknowns = parent_vm.dunknowns.get_view(umap)
    resids    = parent_vm.resids.get_view(umap)
    dresids   = parent_vm.dresids.get_view(umap)
    params    = VecWrapper.create_target_vector(params_dict, unknowns,
                                                     my_params, connections, store_noflats=True)
    dparams   = VecWrapper.create_target_vector(params_dict, unknowns,
                                                     my_params, connections)

    return ViewTuple(unknowns, dunknowns, resids, dresids, params, dparams)


def get_relname_map(unknowns, unknowns_dict, child_name):
    """
    Parameters
    ----------
    unknowns : `VecWrapper`
        A dict-like object containing variables keyed using relative names.

    unknowns_dict : `OrderedDict`
        An ordered mapping of absolute variable name to its metadata.

    child_name : str
        The pathname of the child for which to get relative name

    Returns
    -------
    dict
        Maps relative name in parent (owner of unknowns and unknowns_dict) to
        the corresponding relative name in the child, where relative name may
        include the 'promoted' name of a variable.
    """
    # unknowns is keyed on name relative to the parent system/varmanager
    # unknowns_dict is keyed on absolute pathname
    umap = {}
    for rel, meta in unknowns.items():
        abspath = meta['pathname']
        if abspath.startswith(child_name+':'):
            newmeta = unknowns_dict.get(abspath)
            if newmeta is not None:
                newrel = newmeta['relative_name']
            else:
                newrel = rel
            umap[rel] = newrel

    return umap
