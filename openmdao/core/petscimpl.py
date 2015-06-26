from __future__ import print_function

import os
import sys
from collections import OrderedDict
import numpy

import petsc4py
#petsc4py.init(['-start_in_debugger']) # add petsc init args here
from petsc4py import PETSc


from openmdao.core.vecwrapper import SrcVecWrapper, TgtVecWrapper
from openmdao.core.dataxfer import DataXfer

trace = os.environ.get('TRACE_PETSC')
if trace:
    from openmdao.devtools.debug import debug

class PetscImpl(object):
    """PETSc vector and data transfer implementation factory."""

    @staticmethod
    def create_src_vecwrapper(pathname, comm):
        """
        Create a`PetscSrcVecWrapper`.

        Returns
        -------
        `PetscSrcVecWrapper`
        """
        return PetscSrcVecWrapper(pathname, comm)

    @staticmethod
    def create_tgt_vecwrapper(pathname, comm):
        """
        Create a `PetscTgtVecWrapper`.

        Returns
        -------
        `PetscTgtVecWrapper`
        """
        return PetscTgtVecWrapper(pathname, comm)

    @staticmethod
    def create_data_xfer(src_vec, tgt_vec,
                         src_idxs, tgt_idxs, vec_conns, byobj_conns):
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

        Returns
        -------
        `PetscDataXfer`
            A `PetscDataXfer` object.
        """
        return PetscDataXfer(src_vec, tgt_vec, src_idxs, tgt_idxs, vec_conns, byobj_conns)


class PetscSrcVecWrapper(SrcVecWrapper):

    idx_arr_type = PETSc.IntType

    def setup(self, unknowns_dict, relevant_vars=None, store_byobjs=False):
        """
        Create internal data storage for variables in unknowns_dict.

        Args
        ----
        unknowns_dict : `OrderedDict`
            A dictionary of absolute variable names keyed to an associated
            metadata dictionary.

        relevant_vars : iter of str
            Names of variables that are relevant a particular variable of
            interest.

        store_byobjs : bool
            Indicates that 'pass by object' vars should be stored.  This is only true
            for the unknowns vecwrapper.
        """
        super(PetscSrcVecWrapper, self).setup(unknowns_dict, relevant_vars=relevant_vars,
                                              store_byobjs=store_byobjs)
        if trace:
            debug("'%s': creating src petsc_vec: %s vec=%s" %
                  (self.pathname, self.keys(), self.vec))
        self.petsc_vec = PETSc.Vec().createWithArray(self.vec, comm=self.comm)

    def _get_flattened_sizes(self):
        """
        Collect all flattened sizes of vars stored in our internal array.

        Returns
        -------
        list of `OrderedDict`
            Contains an entry for each process in this object's communicator.
            Each entry is an `OrderedDict` mapping var name to local size for
            'pass by vector' variables.
        """
        sizes = OrderedDict()
        for name, meta in self.get_vecvars():
            if meta.get('remote'):
                sizes[name] = 0
            else:
                sizes[name] = meta['size']

        # collect local var sizes from all of the processes that share the same comm
        # these sizes will be the same in all processes except in cases
        # where a variable belongs to a multiprocessor component.  In that
        # case, the part of the component that runs in a given process will
        # only have a slice of each of the component's variables.
        if trace:
            debug("'%s': allgathering local unknown sizes: local=%s" % (self.pathname,
                                                                        sizes))
        return self.comm.allgather(sizes)

    def norm(self):
        """
        Returns
        -------
        float
            The norm of the distributed vector.
        """
        if trace:
            debug("%s: norm: petsc_vec.assemble" % self.pathname)
        self.petsc_vec.assemble()
        return self.petsc_vec.norm()

    def get_view(self, sys_pathname, comm, varmap, relevance, var_of_interest):
        view = super(PetscSrcVecWrapper, self).get_view(sys_pathname, comm, varmap,
                                                        relevance, var_of_interest)
        if trace:
            debug("'%s': creating src petsc_vec (view): %s: voi=%s, vec=%s" %
                  (sys_pathname, view.keys(), var_of_interest, view.vec))
        view.petsc_vec = PETSc.Vec().createWithArray(view.vec, comm=comm)
        return view


class PetscTgtVecWrapper(TgtVecWrapper):
    idx_arr_type = PETSc.IntType

    def setup(self, parent_params_vec, params_dict, srcvec, my_params,
              connections, relevant_vars=None, store_byobjs=False):
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

        store_byobjs : bool (optional)
            If True, store 'pass by object' variables in the `VecWrapper` we're building.
        """
        super(PetscTgtVecWrapper, self).setup(parent_params_vec, params_dict,
                                              srcvec, my_params,
                                              connections, relevant_vars=relevant_vars,
                                              store_byobjs=store_byobjs)
        if trace:
            debug("'%s': creating tgt petsc_vec: %s: vec=%s" %
                  (self.pathname, self.keys(), self.vec))
        self.petsc_vec = PETSc.Vec().createWithArray(self.vec, comm=self.comm)

    def _get_flattened_sizes(self):
        """
        Returns
        -------
        list of `OrderedDict`
            Contains an entry for each process in this object's communicator.
            Each entry is an `OrderedDict` mapping var name to local size for
            'pass by vector' params.
        """
        psizes = super(PetscTgtVecWrapper, self)._get_flattened_sizes()[0]

        if trace:
            debug("'%s': allgathering param sizes.  local param sizes = %s" % (self.pathname,
                                                                              psizes))
        return self.comm.allgather(psizes)


class PetscDataXfer(DataXfer):
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
    """
    def __init__(self, src_vec, tgt_vec,
                 src_idxs, tgt_idxs, vec_conns, byobj_conns):
        super(PetscDataXfer, self).__init__(src_idxs, tgt_idxs,
                                            vec_conns, byobj_conns)

        self.comm = comm = src_vec.comm

        uvec = src_vec.petsc_vec
        pvec = tgt_vec.petsc_vec
        name = src_vec.pathname

        if trace:
            debug("'%s': creating index sets for '%s' DataXfer: %s %s" %
                  (name, src_vec.pathname, src_idxs, tgt_idxs))
        src_idx_set = PETSc.IS().createGeneral(src_idxs, comm=comm)
        tgt_idx_set = PETSc.IS().createGeneral(tgt_idxs, comm=comm)

        try:
            if trace:
                self.src_idxs = src_idxs
                self.tgt_idxs = tgt_idxs
                debug("'%s': creating scatter %s --> %s %s --> %s" %
                      (name, [v for u,v in vec_conns], [u for u,v in vec_conns],
                       src_idx_set.indices, tgt_idx_set.indices))
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

        deriv : bool
            If True, this is a derivative data transfer, so no pass_by_obj
            variables will be transferred.
        """
        if mode == 'rev':
            # in reverse mode, srcvec and tgtvec are switched. Note, we only
            # run in reverse for derivatives, and derivatives accumulate from
            # all targets. This does not involve pass_by_object.
            if trace:
                conns = ['%s <-- %s' % (u,v) for u,v in self.vec_conns]
                debug("'%s': rev scatter %s  %s <-- %s" %
                            (srcvec.pathname, conns, self.src_idxs, self.tgt_idxs))
                debug("%s: srcvec = %s\ntgtvec = %s" % (srcvec.pathname,
                                                        srcvec.petsc_vec.array,
                                                        tgtvec.petsc_vec.array))
            self.scatter.scatter(tgtvec.petsc_vec, srcvec.petsc_vec, True, True)
        else:
            # forward mode, source to target including pass_by_object
            if trace:
                conns = ['%s --> %s' % (u,v) for u,v in self.vec_conns]
                debug("'%s': fwd scatter %s  %s --> %s" %
                            (srcvec.pathname, conns, self.tgt_idxs, self.src_idxs))
                debug("%s: srcvec = %s\ntgtvec = %s" % (srcvec.pathname,
                                                        srcvec.petsc_vec.array,
                                                        tgtvec.petsc_vec.array))
            self.scatter.scatter(srcvec.petsc_vec, tgtvec.petsc_vec, False, False)
            if trace: debug("scatter done")

            if not deriv:
                for tgt, src in self.byobj_conns:
                    debug('NotImplemented!!!')
                    raise NotImplementedError("can't transfer '%s' to '%s'" %
                                               (src, tgt))
