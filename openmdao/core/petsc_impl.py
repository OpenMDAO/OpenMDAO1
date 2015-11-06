"""PETSc vector and data transfer implementation factory."""

from __future__ import print_function

import os
from six import iteritems

import numpy as np

# import petsc4py
# petsc4py.init(['-start_in_debugger'])  # add petsc init args here
from petsc4py import PETSc

from openmdao.core.vec_wrapper import SrcVecWrapper, TgtVecWrapper

trace = os.environ.get('OPENMDAO_TRACE')
if trace:  # pragma: no cover
    from openmdao.core.mpi_wrap import debug

from mpi4py import MPI

#from openmdao.devtools.debug import diff_mem, mem_usage

class PetscImpl(object):
    """PETSc vector and data transfer implementation factory."""

    idx_arr_type = PETSc.IntType

    @staticmethod
    def world_comm():
        return MPI.COMM_WORLD

    @staticmethod
    def create_src_vecwrapper(sysdata, comm):
        """
        Create a`PetscSrcVecWrapper`.

        Returns
        -------
        `PetscSrcVecWrapper`
        """
        return PetscSrcVecWrapper(sysdata, comm)

    @staticmethod
    def create_tgt_vecwrapper(sysdata, comm):
        """
        Create a `PetscTgtVecWrapper`.

        Returns
        -------
        `PetscTgtVecWrapper`
        """
        return PetscTgtVecWrapper(sysdata, comm)

    @staticmethod
    def create_data_xfer(src_vec, tgt_vec,
                         src_idxs, tgt_idxs, vec_conns, byobj_conns, mode):
        """
        Create an object for performing data transfer between source
        and target vectors.

        Args
        ----
        src_vec : `VecWrapper`
            Variables that are the source of the transfer in fwd mode and
            the destination of the transfer in rev mode.

        tgt_vec : `VecWrapper`
            Variables that are the destination of the transfer in fwd mode and
            the source of the transfer in rev mode.

        src_idxs : array
            Indices of the source variables in the source vector.

        tgt_idxs : array
            Indices of the target variables in the target vector.

        vec_conns : dict
            Mapping of 'pass by vector' variables to the source variables that
            they are connected to.

        byobj_conns : dict
            Mapping of 'pass by object' variables to the source variables that
            they are connected to.

        mode : str
            Either 'fwd' or 'rev', indicating a forward or reverse scatter.

        Returns
        -------
        `PetscDataTransfer`
            A `PetscDataTransfer` object.
        """
        return PetscDataTransfer(src_vec, tgt_vec, src_idxs, tgt_idxs,
                                 vec_conns, byobj_conns, mode)


class PetscSrcVecWrapper(SrcVecWrapper):

    idx_arr_type = PetscImpl.idx_arr_type

    def setup(self, unknowns_dict, relevance, var_of_interest=None,
              store_byobjs=False, shared_vec=None):
        """
        Create internal data storage for variables in unknowns_dict.

        Args
        ----
        unknowns_dict : `OrderedDict`
            A dictionary of absolute variable names keyed to an associated
            metadata dictionary.

        relevance : `Relevance` object
            Object that knows what vars are relevant for each var_of_interest.

        var_of_interest : str or None
            Name of the current variable of interest.

        store_byobjs : bool, optional
            Indicates that 'pass by object' vars should be stored.  This is only true
            for the unknowns vecwrapper.

        shared_vec : ndarray, optional
            If not None, create vec as a subslice of this array.
        """
        super(PetscSrcVecWrapper, self).setup(unknowns_dict, relevance=relevance,
                                              var_of_interest=var_of_interest,
                                              store_byobjs=store_byobjs,
                                              shared_vec=shared_vec)
        if trace:  # pragma: no cover
            debug("'%s': creating src petsc_vec: size(%d) %s vec=%s" %
                  (self._sysdata.pathname, len(self.vec), self.keys(), self.vec))
        self.petsc_vec = PETSc.Vec().createWithArray(self.vec, comm=self.comm)

    def _get_flattened_sizes(self):
        """
        Collect all flattened sizes of vars stored in our internal array.

        Returns
        -------
        list of lists of (name, size) tuples
            Contains an entry for each process in this object's communicator.
        """
        sizes = []
        for name, meta in self._get_vecvars():
            if meta.get('remote'):
                sizes.append((name, 0))
            else:
                sizes.append((name, meta['size']))

        # collect local var sizes from all of the processes that share the same comm
        # these sizes will be the same in all processes except in cases
        # where a variable belongs to a multiprocess component.  In that
        # case, the part of the component that runs in a given process will
        # only have a slice of each of the component's variables.
        if trace:  # pragma: no cover
            debug("'%s': allgathering local unknown sizes: local=%s" %
                     (self._sysdata.pathname, sizes))
        return self.comm.allgather(sizes)

    def norm(self):
        """
        Returns
        -------
        float
            The norm of the distributed vector.
        """
        if trace:  # pragma: no cover
            debug("%s: norm: petsc_vec.assemble" % self._sysdata.pathname)
        self.petsc_vec.assemble()
        return self.petsc_vec.norm()

    def get_view(self, sys_pathname, comm, varmap):
        view = super(PetscSrcVecWrapper, self).get_view(sys_pathname, comm, varmap)
        if trace:  # pragma: no cover
            debug("'%s': creating src petsc_vec (view): (size %d )%s: vec=%s" %
                  (sys_pathname, len(view.vec), view.keys(), view.vec))
        view.petsc_vec = PETSc.Vec().createWithArray(view.vec, comm=comm)
        return view


class PetscTgtVecWrapper(TgtVecWrapper):
    idx_arr_type = PetscImpl.idx_arr_type

    def setup(self, parent_params_vec, params_dict, srcvec, my_params,
              connections, relevance, var_of_interest=None, store_byobjs=False,
              shared_vec=None):
        """
        Configure this vector to store a flattened array of the variables
        in params_dict. Variable shape and value are retrieved from srcvec.

        Args
        ----
        parent_params_vec : `VecWrapper` or None
            `VecWrapper` of parameters from the parent `System`.

        params_dict : `OrderedDict`
            Dictionary of parameter absolute name mapped to metadata dict.

        srcvec : `VecWrapper`
            Source `VecWrapper` corresponding to the target `VecWrapper` we're building.

        my_params : list of str
            A list of absolute names of parameters that the `VecWrapper` we're building
            will 'own'.

        connections : dict of str : str
            A dict of absolute target names mapped to the absolute name of their
            source variable.

        relevance : `Relevance` object
            Object that knows what vars are relevant for each var_of_interest.

        var_of_interest : str or None
            Name of the current variable of interest.

        store_byobjs : bool, optional
            If True, store 'pass by object' variables in the `VecWrapper` we're building.

        shared_vec : ndarray, optional
            If not None, create vec as a subslice of this array.
        """
        super(PetscTgtVecWrapper, self).setup(parent_params_vec, params_dict,
                                              srcvec, my_params,
                                              connections, relevance=relevance,
                                              var_of_interest=var_of_interest,
                                              store_byobjs=store_byobjs,
                                              shared_vec=shared_vec)
        if trace:  # pragma: no cover
            debug("'%s': creating tgt petsc_vec: (size %d) %s: vec=%s" %
                  (self._sysdata.pathname, len(self.vec), self.keys(), self.vec))
        self.petsc_vec = PETSc.Vec().createWithArray(self.vec, comm=self.comm)

    def _get_flattened_sizes(self):
        """
        Returns
        -------
        list of lists of (name, size) tuples
            Contains an entry for each process in this object's communicator.
            Each entry is an `OrderedDict` mapping var name to local size for
            'pass by vector' params.
        """
        psizes = []
        for name, acc in iteritems(self._dat):
            meta = acc.meta
            if acc.owned and not meta.get('pass_by_obj'):
                if meta.get('remote'):
                    psizes.append((name, 0))
                else:
                    psizes.append((name, meta['size']))

        if trace:  # pragma: no cover
            msg = "'%s': allgathering param sizes.  local param sizes = %s"
            debug(msg % (self._sysdata.pathname, psizes))

        return self.comm.allgather(psizes)


class PetscDataTransfer(object):
    """
    Args
    ----
    src_vec : `VecWrapper`
        Variables that are the source of the transfer in fwd mode and
        the destination of the transfer in rev mode.

    tgt_vec : `VecWrapper`
        Variables that are the destination of the transfer in fwd mode and
        the source of the transfer in rev mode.

    src_idxs : array
        indices of the source variables in the source vector.

    tgt_idxs : array
        indices of the target variables in the target vector.

    vec_conns : dict
        mapping of 'pass by vector' variables to the source variables that
        they are connected to.

    byobj_conns : dict
        mapping of 'pass by object' variables to the source variables that
        they are connected to.

    mode : str
        Either 'fwd' or 'rev', indicating a forward or reverse scatter.
    """

    #@diff_mem
    def __init__(self, src_vec, tgt_vec,
                 src_idxs, tgt_idxs, vec_conns, byobj_conns, mode):

        src_idxs = src_vec.merge_idxs(src_idxs)
        tgt_idxs = tgt_vec.merge_idxs(tgt_idxs)

        self.byobj_conns = byobj_conns
        self.comm = comm = src_vec.comm

        uvec = src_vec.petsc_vec
        pvec = tgt_vec.petsc_vec
        name = src_vec._sysdata.pathname

        if trace:
            debug("'%s': creating index sets for '%s' DataTransfer: %s %s" %
                  (name, src_vec._sysdata.pathname, src_idxs, tgt_idxs))

        src_idx_set = PETSc.IS().createGeneral(src_idxs, comm=comm)
        tgt_idx_set = PETSc.IS().createGeneral(tgt_idxs, comm=comm)

        try:
            if trace:  # pragma: no cover
                self.src_idxs = src_idxs
                self.tgt_idxs = tgt_idxs
                self.vec_conns = vec_conns
                arrow = '-->' if mode == 'fwd' else '<--'
                debug("'%s': new %s scatter (sizes: %d, %d)\n   %s %s %s %s %s %s" %
                      (name, mode, len(src_idx_set.indices), len(tgt_idx_set.indices),
                       [v for u, v in vec_conns], arrow, [u for u, v in vec_conns],
                       src_idx_set.indices, arrow, tgt_idx_set.indices))
            self.scatter = PETSc.Scatter().create(uvec, src_idx_set,
                                                  pvec, tgt_idx_set)
        except Exception as err:
            raise RuntimeError("ERROR in %s (src_idxs=%s, tgt_idxs=%s, usize=%d, psize=%d): %s" %
                               (name, src_idxs, tgt_idxs,
                                src_vec.vec.size,
                                tgt_vec.vec.size, str(err)))

    def transfer(self, srcvec, tgtvec, mode='fwd', deriv=False):
        """Performs data transfer between a distributed source vector and
        a distributed target vector.

        Args
        ----
        srcvec : `VecWrapper`
            Variables that are the source of the transfer in fwd mode and
            the destination of the transfer in rev mode.

        tgtvec : `VecWrapper`
            Variables that are the destination of the transfer in fwd mode and
            the source of the transfer in rev mode.

        mode : 'fwd' or 'rev', optional
            Direction of the data transfer, source to target ('fwd', the default)
            or target to source ('rev').

        deriv : bool, optional
            If True, this is a derivative data transfer, so no pass_by_obj
            variables will be transferred.
        """
        if mode == 'rev':
            # in reverse mode, srcvec and tgtvec are switched. Note, we only
            # run in reverse for derivatives, and derivatives accumulate from
            # all targets. This does not involve pass_by_object.
            if trace:  # pragma: no cover
                conns = ['%s <-- %s' % (u, v) for v, u in self.vec_conns]
                debug("%s rev scatter %s  %s <-- %s" %
                      (srcvec._sysdata.pathname, conns, self.src_idxs, self.tgt_idxs))
                debug("%s:    srcvec = %s" % (tgtvec._sysdata.pathname,
                                              tgtvec.petsc_vec.array))
            self.scatter.scatter(tgtvec.petsc_vec, srcvec.petsc_vec, True, True)
            if trace:  # pragma: no cover
                debug("%s:    tgtvec = %s" % (srcvec._sysdata.pathname,
                                              srcvec.petsc_vec.array))
        else:
            # forward mode, source to target including pass_by_object
            if trace:  # pragma: no cover
                conns = ['%s --> %s' % (u, v) for v, u in self.vec_conns]
                debug("%s fwd scatter %s  %s --> %s" %
                      (srcvec._sysdata.pathname, conns, self.src_idxs, self.tgt_idxs))
                debug("%s:    srcvec = %s" % (srcvec._sysdata.pathname,
                                              srcvec.petsc_vec.array))
            self.scatter.scatter(srcvec.petsc_vec, tgtvec.petsc_vec, False, False)

            if trace:  # pragma: no cover
                debug("%s:    tgtvec = %s" % (tgtvec._sysdata.pathname,
                                              tgtvec.petsc_vec.array))

            if not deriv:
                for tgt, src in self.byobj_conns:
                    raise NotImplementedError("can't transfer '%s' to '%s'" %
                                              (src, tgt))
