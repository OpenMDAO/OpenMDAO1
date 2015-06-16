import sys
from collections import namedtuple, OrderedDict
import numpy

VecTuple = namedtuple('VecTuple', 'unknowns, dunknowns, resids, dresids, params, dparams')

from openmdao.devtools.debug import debug

class VarManager(object):
    """A manager of the data transfer of a possibly distributed
    collection of variables.

    Parameters
    ----------
    system : `System`
        `System` containing this `VarManager`.

    my_params : list
        List of pathnames for parameters that this `VarManager` is
        responsible for propagating.

    impl : an implementation factory, optional
        Specifies the factory object used to create `VecWrapper` and
        `DataXfer` objects.
    """
    def __init__(self, system, my_params, impl):
        comm = system.comm
        sys_pathname = system.pathname
        params_dict = system._params_dict
        unknowns_dict = system._unknowns_dict

        self.impl_factory = impl
        self.comm = comm
        self.data_xfer = {}
        self.distrib_idxs = {}  # this will be non-empty if some systems have distributed vars

        self._setup_data_transfer(system, my_params, system._relevance, None)

    def _get_global_offset(self, name, var_rank, sizes_table):
        """
        Parameters
        ----------
        name : str
            The variable name.

        var_rank : int
            The rank the the offset is requested for.

        sizes_table : list of OrderDicts mappging var name to size.
            Size information for all vars in all ranks.

        Returns
        -------
        int
            The offset into the distributed vector for the named variable
            in the specified rank (process).
        """
        offset = 0
        rank = 0

        # first get the offset of the distributed storage for var_rank
        while rank < var_rank:
            offset += sum(sizes_table[rank].values())
            rank += 1

        # now, get the offset into the var_rank storage for the variable
        for vname, size in sizes_table[var_rank].items():
            if vname == name:
                break
            offset += size

        return offset

    def _get_global_idxs(self, uname, pname, uvec, pvec, var_of_interest, mode):
        """
        Parameters
        ----------
        uname : str
            Name of variable in the unknowns vector.

        pname : str
            Name of the variable in the params vector.

        uvec : `VecWrapper`
            unknowns/dunknowns vec wrapper.

        pvec : `VecWrapper`
            params/dparams vec wrapper.

        var_of_interest : str or None
            Name of variable of interest used to determine relevance.

        Returns
        -------
        tuple of (idx_array, idx_array)
            index array into the global unknowns vector and the corresponding
            index array into the global params vector.
        """
        umeta = uvec.metadata(uname)
        pmeta = pvec.metadata(pname)

        # FIXME: if we switch to push scatters, this check will flip
        if (mode == 'fwd' and pmeta.get('remote')) or (mode == 'rev' and umeta.get('remote')):
            # just return empty index arrays for remote vars
            return pvec.make_idx_array(0, 0), pvec.make_idx_array(0, 0)

        if pname in self.distrib_idxs:
            raise NotImplementedError("distrib comps not supported yet")
        else:
            arg_idxs = pvec.make_idx_array(0, pmeta['size'])

        var_rank = self._get_owning_rank(uname, self._local_unknown_sizes)
        offset = self._get_global_offset(uname, var_rank, self._local_unknown_sizes)
        src_idxs = arg_idxs + offset

        var_rank = self._get_owning_rank(pname, self._local_param_sizes)
        tgt_start = self._get_global_offset(pname, var_rank, self._local_param_sizes)
        tgt_idxs = tgt_start + pvec.make_idx_array(0, len(arg_idxs))

        return src_idxs, tgt_idxs

    def _setup_data_transfer(self, system, my_params, relevance, var_of_interest):
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

        self._local_unknown_sizes = system.unknowns._get_flattened_sizes()
        self._local_param_sizes = system.params._get_flattened_sizes()

        xfer_dict = {}
        for param, unknown in system.connections.items():
            if not (relevance.is_relevant(var_of_interest, param) or
                      relevance.is_relevant(var_of_interest, unknown)):
                continue

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

                urelname = system.unknowns.get_relative_varname(unknown)
                prelname = system.params.get_relative_varname(param)

                if system.unknowns.metadata(urelname).get('pass_by_obj'):
                    byobj_conns.append((prelname, urelname))
                else: # pass by vector
                    #forward
                    sidxs, didxs = self._get_global_idxs(urelname, prelname,
                                                         system.unknowns, system.params, None, 'fwd')
                    vec_conns.append((prelname, urelname))
                    src_idx_list.append(sidxs)
                    dest_idx_list.append(didxs)

                    # reverse
                    sidxs, didxs = self._get_global_idxs(urelname, prelname,
                                                         system.unknowns, system.params, None, 'rev')
                    rev_vec_conns.append((prelname, urelname))
                    rev_src_idx_list.append(sidxs)
                    rev_dest_idx_list.append(didxs)

        for (tgt_sys, mode), (srcs, tgts, vec_conns, byobj_conns) in xfer_dict.items():
            src_idxs, tgt_idxs = system.unknowns.merge_idxs(srcs, tgts)
            if vec_conns or byobj_conns:
                self.data_xfer[(tgt_sys, mode)] = \
                    self.impl_factory.create_data_xfer(system, src_idxs, tgt_idxs,
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

            src_idxs, tgt_idxs = system.unknowns.merge_idxs(full_srcs, full_tgts)
            self.data_xfer[('', mode)] = \
                self.impl_factory.create_data_xfer(system, src_idxs, tgt_idxs,
                                                   full_flats, full_byobjs)

    def _transfer_data(self, group, target_sys='', mode='fwd', deriv=False,
                       var_of_interest=None):
        """
        Transfer data to/from target_system depending on mode.

        Parameters
        ----------
        group : `Group`
            `Group` that owns the scattering vectors.

        target_sys : str, optional
            Name of the target `System`.  A name of '', the default, indicates that data
            should be transfered to all subsystems at once.

        mode : { 'fwd', 'rev' }, optional
            Specifies forward or reverse data transfer.

        deriv : bool, optional
            If True, use du/dp for scatter instead of u/p.  Default is False.

        var_of_interest : str or None
            Specifies the variable of interest to determine relevance.

        """
        x = self.data_xfer.get((target_sys, mode))
        if x is not None:
            if deriv:
                x.transfer(group.dumat[var_of_interest], group.dpmat[var_of_interest],
                           mode, deriv=True)
            else:
                x.transfer(group.unknowns, group.params, mode)

    def _get_owning_rank(self, name, sizes_table):
        """
        Parameters
        ----------
        name : str
            Name of the variable to find the owning rank for

        sizes_table : list of ordered dicts mapping name to size
            Size info for all vars in all ranks.

        Returns
        -------
        int
            The rank of the lowest ranked process that has a local copy
            of the variable.
        """
        if self.comm is None:
            return 0

        if sizes_table[self.comm.rank][name]:
            return self.comm.rank
        else:
            for i in range(self.comm.size):
                if sizes_table[i][name]:
                    return i
            else:
                raise RuntimeError("Can't find a source for '%s' with a non-zero size" %
                                   name)
