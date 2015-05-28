
from collections import namedtuple, OrderedDict
import numpy

VecTuple = namedtuple('VecTuple', 'unknowns, dunknowns, resids, dresids, params, dparams')

class VarManagerBase(object):
    """Base class for a manager of the data transfer of a possibly distributed
    collection of variables.

    Parameters
    ----------
        connections : dict
            A dictionary mapping the pathname of a target variable to the
            pathname of the source variable that it is connected to.
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

    def __getitem__(self, name):
        """Retrieve unflattened value of named variable.

        Parameters
        ----------
        name : str   OR   tuple : (name, vector)
             The name of the variable to retrieve from the unknowns vector OR
             a tuple of the name of the variable and the vector to get its
             value from.

        Returns
        -------
        The unflattened value of the given variable.
        """
        if isinstance(name, tuple):
            name, vector = name
        else:
            vector = 'unknowns'
        try:
            return getattr(self, vector)[name]
        except KeyError:
            raise KeyError("'%s' is not in the %s vector for this system" %
                           (name, vector))

    def _setup_data_transfer(self, sys_pathname, my_params):
        """Create `DataXfer` objects to handle data transfer for all of the
           connections that involve paramaters for which this `VarManager`
           is responsible.

           Parameters
           ----------
           sys_pathname : str
               Absolute pathname of the `System` that will own this `VarManager`.

           my_params : list
               list of pathnames for parameters that the VarManager is
               responsible for propagating.
        """

        self._local_unknown_sizes = self.unknowns._get_flattened_sizes()
        self._local_param_sizes = self.params._get_flattened_sizes()

        self.app_ordering = self.impl_factory.create_app_ordering(self)

        xfer_dict = {}
        for param, unknown in self.connections.items():
            if param in my_params:
                # remove our system pathname from the abs pathname of the param and
                # get the subsystem name from that
                if sys_pathname:
                    start = len(sys_pathname)+1
                else:
                    start = 0
                tgt_sys = param[start:].split(':', 1)[0]
                src_idx_list, dest_idx_list, flat_conns, noflat_conns = \
                                   xfer_dict.setdefault(tgt_sys, ([],[],[],[]))
                urelname = self.unknowns.get_relative_varname(unknown)
                prelname = self.params.get_relative_varname(param)
                noflat = self.unknowns.metadata(urelname).get('noflat')
                if noflat:
                    noflat_conns.append((prelname, urelname))
                else:
                    flat_conns.append((prelname, urelname))
                    src_idx_list.append(self.unknowns.get_global_idxs(urelname))
                    dest_idx_list.append(self.params.get_global_idxs(prelname))

        for tgt_sys, (srcs, tgts, flat_conns, noflat_conns) in xfer_dict.items():
            src_idxs, tgt_idxs = self.unknowns.merge_idxs(srcs, tgts)
            if flat_conns or noflat_conns:
                self.data_xfer[tgt_sys] = self.impl_factory.create_data_xfer(self, src_idxs, tgt_idxs,
                                                                             flat_conns, noflat_conns)

        # create a DataXfer object that combines all of the
        # individual subsystem src_idxs, tgt_idxs, and noflat_conns, so that a 'full'
        # scatter to all subsystems can be done at the same time.  Store that DataXfer
        # object under the name ''.
        full_srcs = []
        full_tgts = []
        full_flats = []
        full_noflats = []

        for src, tgts, flats, noflats in xfer_dict.values():
            full_srcs.extend(src)
            full_tgts.extend(tgts)
            full_flats.extend(flats)
            full_noflats.extend(noflats)

        src_idxs, tgt_idxs = self.unknowns.merge_idxs(full_srcs, full_tgts)
        self.data_xfer[''] = self.impl_factory.create_data_xfer(self, src_idxs, tgt_idxs,
                                                                full_flats, full_noflats)

    def _transfer_data(self, target_system='', mode='fwd', deriv=False):
        """Transfer data to/from target_system depending on mode.

        Parameters
        ----------
        target_system : str
            Name of the target `System`.  A name of '' indicates that data
            should be transfered to all subsystems at once.

        mode : { 'fwd', 'rev' }, optional
            Specifies forward or reverse data transfer.

        deriv : bool, optional
            If True, perform a data transfer between derivative `VecWrappers`.
        """
        x = self.data_xfer.get(target_system)
        if x is not None:
            if deriv:
                x.transfer(self.dunknowns, self.dparams, mode, deriv=True)
            else:
                x.transfer(self.unknowns, self.params, mode)

    def vectors(self):
        """Return the set of variable vectors being managed by this `VarManager`.

        Returns
        -------
        `VecTuple`
            A namedtuple of six (6) `VecWrappers`:
            unknowns, dunknowns, resids, dresids, params, dparams.
        """
        return VecTuple(self.unknowns, self.dunknowns,
                        self.resids, self.dresids,
                        self.params, self.dparams)


class VarManager(VarManagerBase):
    """A manager of the data transfer of a possibly distributed
    collection of variables.

    Parameters
    ----------
    params_dict : dict
        Dictionary of metadata for all parameters.

    unknowns_dict : dict
        Dictionary of metadata for all unknowns.

    my_params : list
        List of pathnames for parameters that this `VarManager` is
        responsible for propagating.

    connections : dict
        A dictionary mapping the pathname of a target variable to the
        pathname of the source variable that it is connected to.

    impl : an implementation factory, optional
        Specifies the factory object used to create `VecWrapper` and
        `DataXfer` objects.
    """
    def __init__(self, comm, sys_pathname, params_dict, unknowns_dict, my_params,
                 connections, impl):
        super(VarManager, self).__init__(connections)

        self.impl_factory = impl
        self.comm = comm

        # create implementation specific VecWrappers
        self.unknowns  = self.impl_factory.create_src_vecwrapper(sys_pathname, comm)
        self.dunknowns = self.impl_factory.create_src_vecwrapper(sys_pathname, comm)
        self.resids    = self.impl_factory.create_src_vecwrapper(sys_pathname, comm)
        self.dresids   = self.impl_factory.create_src_vecwrapper(sys_pathname, comm)
        self.params    = self.impl_factory.create_tgt_vecwrapper(sys_pathname, comm)
        self.dparams   = self.impl_factory.create_tgt_vecwrapper(sys_pathname, comm)

        # populate the VecWrappers with data
        self.unknowns.setup(unknowns_dict, store_noflats=True)
        self.dunknowns.setup(unknowns_dict)
        self.resids.setup(unknowns_dict)
        self.dresids.setup(unknowns_dict)

        self.params.setup(None, params_dict, self.unknowns,
                                              my_params, connections, store_noflats=True)
        self.dparams.setup(None, params_dict, self.unknowns,
                                               my_params, connections)

        self._setup_data_transfer(sys_pathname, my_params)


class ViewVarManager(VarManagerBase):
    """A manager of the data transfer of a possibly distributed collection of
    variables.  The variables are based on views into an existing VarManager.

    Parameters
    ----------
    top_unknowns : `VecWrapper`
        The `Problem` level unknowns `VecWrapper`.

    parent_vm : `VarManager`
        The `VarManager` which provides the `VecWrappers` on which to create views.

    params_dict : dict
        Dictionary of metadata for all parameters.

    unknowns_dict : dict
        Dictionary of metadata for all unknowns.

    my_params : list
        List of pathnames for parameters that this `VarManager` is
        responsible for propagating.

    """
    def __init__(self, top_unknowns, parent_vm, comm, sys_pathname, params_dict, unknowns_dict,
                 my_params):
        super(ViewVarManager, self).__init__(parent_vm.connections)

        self.impl_factory = parent_vm.impl_factory

        self.unknowns, self.dunknowns, self.resids, self.dresids, self.params, self.dparams = \
            create_views(top_unknowns, parent_vm, comm, sys_pathname, params_dict, unknowns_dict,
                         my_params, parent_vm.connections)

        self._setup_data_transfer(sys_pathname, my_params)


def create_views(top_unknowns, parent_vm, comm, sys_pathname, params_dict, unknowns_dict,
                 my_params, connections):
    """
    A manager of the data transfer of a possibly distributed collection of
    variables.  The variables are based on views into an existing VarManager.

    Parameters
    ----------
    top_unknowns : `VecWrapper`
        The `Problem` level unknowns `VecWrapper`.

    parent_vm : `VarManager`
        The `VarManager` which provides the `VecWrapper` on which to create views.

    comm : an MPI communicator (real or fake)
        Communicator to be used for any distributed operations.

    sys_pathname : str
        Pathname of the system for which the views are being created.

    params_dict : dict
        Dictionary of metadata for all parameters.

    unknowns_dict : dict
        Dictionary of metadata for all unknowns.

    my_params : list
        List of pathnames for parameters that this `VarManager` is
        responsible for propagating.

    connections : dict
        A dictionary mapping the pathname of a target variable to the
        pathname of the source variable that it is connected to.

    Returns
    -------
    `VecTuple`
        A namedtuple of six (6) `VecWrappers`:
        unknowns, dunknowns, resids, dresids, params, dparams.
    """

    # map relative name in parent to corresponding relative name in this view
    umap = get_relname_map(parent_vm.unknowns, unknowns_dict, sys_pathname)

    unknowns  = parent_vm.unknowns.get_view(comm, umap)
    dunknowns = parent_vm.dunknowns.get_view(comm, umap)
    resids    = parent_vm.resids.get_view(comm, umap)
    dresids   = parent_vm.dresids.get_view(comm, umap)

    params  = parent_vm.impl_factory.create_tgt_vecwrapper(sys_pathname, comm)
    dparams = parent_vm.impl_factory.create_tgt_vecwrapper(sys_pathname, comm)

    params.setup(parent_vm.params, params_dict, top_unknowns,
                               my_params, connections, store_noflats=True)
    dparams.setup(parent_vm.dparams, params_dict, top_unknowns,
                                my_params, connections)

    return VecTuple(unknowns, dunknowns, resids, dresids, params, dparams)


def get_relname_map(unknowns, unknowns_dict, child_name):
    """
    Parameters
    ----------
    unknowns : `VecWrapper`
        A dict-like object containing variables keyed using relative names.

    unknowns_dict : `OrderedDict`
        An ordered mapping of absolute variable name to its metadata.

    child_name : str
        The pathname of the child for which to get relative name.

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
