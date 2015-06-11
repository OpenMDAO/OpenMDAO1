import sys
from collections import namedtuple, OrderedDict
import numpy

VecTuple = namedtuple('VecTuple', 'unknowns, dunknowns, resids, dresids, params, dparams')

from openmdao.devtools.debug import debug

class VarManagerBase(object):
    """Base class for a manager of the data transfer of a possibly distributed
    collection of variables.

    Parameters
    ----------
        connections : dict
            A dictionary mapping the pathname of a target variable to the
            pathname of the source variable that it is connected to.

        vardeps : dict
            A dictionary of dictionaries that maps full variable pathnames to all
            of their 'downstream' variables, where 'downstream' depends on mode, which
            can be 'fwd' or 'rev'.

    """
    def __init__(self, connections, vardeps):
        self.connections = connections
        self.params    = None
        self.dparams   = None
        self.unknowns  = None
        self.dunknowns = None
        self.resids    = None
        self.dresids   = None
        self.vardeps   = vardeps
        self.data_xfer = {}
        self.distrib_idxs = {}  # this will be non-empty if some systems have distributed vars

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

    def _get_global_idxs(self, uname, pname):
        """
        Parameters
        ----------
        uname : str
            Name of variable in the unknowns vector.

        pname : str
            Name of the variable in the params vector.

        Returns
        -------
        tuple of (idx_array, idx_array)
            index array into the global unknowns vector and the corresponding
            index array into the global params vector.
        """
        pmeta = self.params.metadata(pname)
        if pmeta.get('remote'):
            # just return empty index arrays for remote vars
            debug("returning empty for",pname)
            return self.params.make_idx_array(0, 0), self.params.make_idx_array(0, 0)

        if pname in self.distrib_idxs:
            raise NotImplementedError("distrib comps not supported yet")
        else:
            arg_idxs = self.params.make_idx_array(0, pmeta['size'])

        # get the offset to the beginning of local storage in the distributed
        # vector
        var_rank = self._get_owning_rank(uname)

        offset = 0
        rank = 0
        while rank < var_rank:
            for vname, size in self._local_unknown_sizes[rank]:
                offset += size
            rank += 1

        # now, we need the offset into the owning rank storage for the variable
        for vname, size in self._local_unknown_sizes[var_rank]:
            if vname == uname:
                break
            offset += size

        debug("slices:",self.unknowns._slices)
        src_idxs = arg_idxs + offset

        myrank = self.unknowns.comm.rank if self.unknowns.comm else 0

        tgt_start = 0
        rank = 0
        while rank < myrank:
            for vname, size in self._local_param_sizes[rank]:
                tgt_start += size
            rank += 1

        for vname, size in self._local_param_sizes[myrank]:
            if vname == pname:
                break
            tgt_start += size

        debug('tgt_start:',tgt_start)
        #tgt_idxs = tgt_start + self.params._slices[pname][0] + \
            #self.params.make_idx_array(0, len(arg_idxs))
        tgt_idxs = tgt_start + \
            self.params.make_idx_array(0, len(arg_idxs))

        return src_idxs, tgt_idxs

    def _setup_data_transfer(self, system, my_params, vardeps):
        """
        Create `DataXfer` objects to handle data transfer for all of the
        connections that involve parameters for which this `VarManager`
        is responsible.

        Parameters
        ----------
        system : `System`
            The `System` that will own this `VarManager`.

        my_params : list
            List of pathnames for parameters that the VarManager is
            responsible for propagating.

        """

        sys_pathname = system.pathname

        self._local_unknown_sizes = self.unknowns._get_flattened_sizes()
        self._local_param_sizes = self.params._get_flattened_sizes()
        debug("local_param_sizes = ",self._local_param_sizes)

        xfer_dict = {}
        for param, unknown in self.connections.items():
            if param in my_params:
                # remove our system pathname from the abs pathname of the param and
                # get the subsystem name from that
                start = len(sys_pathname)+1 if sys_pathname else 0

                tgt_sys = param[start:].split(':', 1)[0]
                src_sys = unknown[start:].split(':', 1)[0]

                src_idx_list, dest_idx_list, vec_conns, byobj_conns = \
                                   xfer_dict.setdefault((tgt_sys, 'fwd'), ([],[],[],[]))

                rev_src_idx_list, rev_dest_idx_list, rev_vec_conns, rev_byobj_conns = \
                                   xfer_dict.setdefault((src_sys, 'rev'), ([],[],[],[]))

                urelname = self.unknowns.get_relative_varname(unknown)
                prelname = self.params.get_relative_varname(param)

                if self.unknowns.metadata(urelname).get('pass_by_obj'):
                    byobj_conns.append((prelname, urelname))
                else: # pass by vector
                    sidxs, didxs = self._get_global_idxs(urelname, prelname)
                    #forward
                    vec_conns.append((prelname, urelname))
                    src_idx_list.append(sidxs)
                    dest_idx_list.append(didxs)

                    # reverse
                    rev_vec_conns.append((prelname, urelname))
                    rev_src_idx_list.append(sidxs)
                    rev_dest_idx_list.append(didxs)

        for (tgt_sys, mode), (srcs, tgts, vec_conns, byobj_conns) in xfer_dict.items():
            src_idxs, tgt_idxs = self.unknowns.merge_idxs(srcs, tgts)
            if vec_conns or byobj_conns:
                self.data_xfer[(tgt_sys, mode)] = \
                    self.impl_factory.create_data_xfer(self, src_idxs, tgt_idxs,
                                                       vec_conns, byobj_conns)

        # create a DataXfer object that combines all of the
        # individual subsystem src_idxs, tgt_idxs, and byobj_conns, so that a 'full'
        # scatter to all subsystems can be done at the same time.  Store that DataXfer
        # object under the name ''.

        for mode in ('fwd', 'rev'):
            full_srcs = []
            full_tgts = []
            full_flats = []
            full_byobjs = []
            for (tgt_sys, direction), (srcs, tgts, flats, byobjs) in xfer_dict.items():
                if mode == direction:
                    full_srcs.extend(srcs)
                    full_tgts.extend(tgts)
                    full_flats.extend(flats)
                    full_byobjs.extend(byobjs)

            src_idxs, tgt_idxs = self.unknowns.merge_idxs(full_srcs, full_tgts)
            self.data_xfer[('', mode)] = \
                self.impl_factory.create_data_xfer(self, src_idxs, tgt_idxs,
                                                   full_flats, full_byobjs)

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
        x = self.data_xfer.get((target_system, mode))
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
                        self.resids,   self.dresids,
                        self.params,   self.dparams)

    def _get_owning_rank(self, name):
        """
        Parameters
        ----------
        name : str
            Name of the variable to find the owning rank for

        Returns
        -------
        int
            The rank of the lowest ranked process that has a local copy
            of the variable.
        """
        if self.comm is None:
            return 0

        vidx = self.unknowns._var_idx(name)
        if self._local_unknown_sizes[self.comm.rank][vidx][1]:
            return self.comm.rank
        else:
            for i in range(self.comm.size):
                if self._local_unknown_sizes[i][vidx][1]:
                    return i
            else:
                raise RuntimeError("Can't find a source for '%s' with a non-zero size" %
                                   name)


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

    vardeps : dict
        A dictionary of dictionaries that maps full variable pathnames to all
        of their 'downstream' variables, where 'downstream' depends on mode, which
        can be 'fwd' or 'rev'.

    impl : an implementation factory, optional
        Specifies the factory object used to create `VecWrapper` and
        `DataXfer` objects.
    """
    def __init__(self, system, my_params, connections, vardeps, impl):
        super(VarManager, self).__init__(connections, vardeps)

        comm = system.comm
        sys_pathname = system.pathname
        params_dict = system._params_dict
        unknowns_dict = system._unknowns_dict

        self.impl_factory = impl
        self.comm = comm

        # create implementation specific VecWrappers
        self.unknowns  = self.impl_factory.create_src_vecwrapper(sys_pathname, comm)
        self.resids    = self.impl_factory.create_src_vecwrapper(sys_pathname, comm)
        self.params    = self.impl_factory.create_tgt_vecwrapper(sys_pathname, comm)

        self.dunknowns = self.impl_factory.create_src_vecwrapper(sys_pathname, comm)
        self.dresids   = self.impl_factory.create_src_vecwrapper(sys_pathname, comm)
        self.dparams   = self.impl_factory.create_tgt_vecwrapper(sys_pathname, comm)

        # populate the VecWrappers with data
        self.unknowns.setup(unknowns_dict, vardeps, store_byobjs=True)
        self.resids.setup(unknowns_dict, vardeps)
        self.params.setup(None, params_dict, self.unknowns,
                                my_params, connections, vardeps, store_byobjs=True)

        self.dunknowns.setup(unknowns_dict, vardeps)
        self.dresids.setup(unknowns_dict, vardeps)
        self.dparams.setup(None, params_dict, self.unknowns,
                                 my_params, connections, vardeps)

        self._setup_data_transfer(system, my_params, vardeps)


class ViewVarManager(VarManagerBase):
    """A manager of the data transfer of a possibly distributed collection of
    variables.  The variables are based on views into an existing VarManager.

    Parameters
    ----------
    top_unknowns : `VecWrapper`
        The `Problem` level unknowns `VecWrapper`.

    system : `System`
        The `System` that owns this VarManager

    my_params : list
        List of pathnames for parameters that this `VarManager` is
        responsible for propagating.

    """
    def __init__(self, top_unknowns, parent_vm, system, my_params):
        super(ViewVarManager, self).__init__(parent_vm.connections, parent_vm.vardeps)

        comm = system.comm

        self.impl_factory = parent_vm.impl_factory
        self.comm = comm

        self.unknowns, self.dunknowns, self.resids, self.dresids, self.params, self.dparams = \
            create_views(top_unknowns, parent_vm, system, my_params)

        self._setup_data_transfer(system, my_params, parent_vm.vardeps)


def create_views(top_unknowns, parent_vm, system, my_params):
    """
    A manager of the data transfer of a possibly distributed collection of
    variables.  The variables are based on views into an existing VarManager.

    Parameters
    ----------
    top_unknowns : `VecWrapper`
        The `Problem` level unknowns `VecWrapper`.

    parent_vm : `VarManager`
        The `VarManager` which provides the `VecWrapper` on which to create views.

    system : `System`
        The `System` that will contain the views.

    my_params : list
        List of pathnames for parameters that this `VarManager` is
        responsible for propagating.

    Returns
    -------
    `VecTuple`
        A namedtuple of six (6) `VecWrappers`:
        unknowns, dunknowns, resids, dresids, params, dparams.
    """

    comm = system.comm
    unknowns_dict = system._unknowns_dict
    params_dict = system._params_dict
    sys_pathname = system.pathname
    connections = parent_vm.connections
    vardeps = parent_vm.vardeps

    # map relative name in parent to corresponding relative name in this view
    umap = get_relname_map(parent_vm.unknowns, unknowns_dict, sys_pathname)

    unknowns  = parent_vm.unknowns.get_view(sys_pathname, comm, umap)
    resids    = parent_vm.resids.get_view(sys_pathname, comm, umap)
    params    = parent_vm.impl_factory.create_tgt_vecwrapper(sys_pathname, comm)
    params.setup(parent_vm.params, params_dict, top_unknowns,
                 my_params, connections, vardeps, store_byobjs=True)

    dunknowns = parent_vm.dunknowns.get_view(sys_pathname, comm, umap)
    dresids   = parent_vm.dresids.get_view(sys_pathname, comm, umap)
    dparams   = parent_vm.impl_factory.create_tgt_vecwrapper(sys_pathname, comm)
    dparams.setup(parent_vm.dparams, params_dict, top_unknowns,
                  my_params, connections, vardeps)

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
