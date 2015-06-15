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
from openmdao.devtools.debug import debug

trace = os.environ.get('TRACE_PETSC')

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
    def create_data_xfer(system, src_idxs, tgt_idxs, vec_conns, byobj_conns):
        """
        Create an object for performing data transfer between source
        and target vectors.

        Parameters
        ----------
        system : `System`
            The `System` that manages this data transfer.

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
        return PetscDataXfer(system, src_idxs, tgt_idxs, vec_conns, byobj_conns)


class PetscSrcVecWrapper(SrcVecWrapper):

    idx_arr_type = PETSc.IntType

    def setup(self, unknowns_dict, relevant_vars=None, store_byobjs=False):
        """
        Create internal data storage for variables in unknowns_dict.

        Parameters
        ----------
        unknowns_dict : `OrderedDict`
            A dictionary of absolute variable names keyed to an associated
            metadata dictionary.

        store_byobjs : bool
            Indicates that 'pass by object' vars should be stored.  This is only true
            for the unknowns vecwrapper.
        """
        super(PetscSrcVecWrapper, self).setup(unknowns_dict, relevant_vars=relevant_vars,
                                              store_byobjs=store_byobjs)
        if trace:
            debug("'%s': creating src petsc_vec: vec=%s" %
                  (self.pathname, self.vec))
        self.petsc_vec = PETSc.Vec().createWithArray(self.vec, comm=self.comm)

    def _get_flattened_sizes(self):
        """
        Collect all flattened sizes of vars stored in our internal array.

        Returns
        -------
        list of tuples of the form (name, size)
            Array containing local sizes of 'pass by vector' unknown variables
            for every process in our communicator.
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
        size_table = self.comm.allgather(sizes)

        return size_table

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
            debug("'%s': creating src petsc_vec (view): vec=%s" %
                  (sys_pathname, self.vec))
        view.petsc_vec = PETSc.Vec().createWithArray(view.vec, comm=comm)
        return view


class PetscTgtVecWrapper(TgtVecWrapper):
    idx_arr_type = PETSc.IntType

    def setup(self, parent_params_vec, params_dict, srcvec, my_params,
              connections, relevant_vars=None, store_byobjs=False):
        """
        Configure this vector to store a flattened array of the variables
        in params_dict. Variable shape and value are retrieved from srcvec.

        Parameters
        ----------
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
            debug("'%s': creating tgt petsc_vec: vec=%s" %
                  (self.pathname, self.vec))
        self.petsc_vec = PETSc.Vec().createWithArray(self.vec, comm=self.comm)

    def _get_flattened_sizes(self):
        """
        Create a list of lists, one list per process, where each process list
        contains tuples of the form (varname, local_size).

        Returns
        -------
        list of list of tuples of the form (varname, local_size)
            List containing local sizes of 'pass by vector' params for each
            process.
        """
        psizes = super(PetscTgtVecWrapper, self)._get_flattened_sizes()[0]

        if trace:
            debug("'%s': allgathering param sizes.  local param sizes = %s" % (self.pathname,
                                                                              psizes))
        return self.comm.allgather(psizes)


class PetscDataXfer(DataXfer):
    """
    Parameters
    ----------
    system : `System`
        The `System` that contains the `VecWrappers` used for this data transfer.

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
    def __init__(self, system, src_idxs, tgt_idxs, vec_conns, byobj_conns):
        super(PetscDataXfer, self).__init__(src_idxs, tgt_idxs,
                                            vec_conns, byobj_conns)

        self.comm = comm = system.comm

        uvec = system.unknowns.petsc_vec
        pvec = system.params.petsc_vec

        name = system.unknowns.pathname

        if trace:
            debug("'%s': creating index sets for '%s' DataXfer:\n      %s\n      %s" %
                  (name, system.unknowns.pathname, src_idxs, tgt_idxs))
        src_idx_set = PETSc.IS().createGeneral(src_idxs, comm=comm)
        tgt_idx_set = PETSc.IS().createGeneral(tgt_idxs, comm=comm)

        if trace:
            debug("'%s': petsc indices: %s" % (name, src_idx_set.indices))

        try:
            if trace:
                debug("'%s': creating scatter %s --> %s" % (name, src_idx_set.indices,
                                                          tgt_idx_set.indices))
            self.scatter = PETSc.Scatter().create(uvec, src_idx_set,
                                                  pvec, tgt_idx_set)
        except Exception as err:
            raise RuntimeError("ERROR in %s (src_idxs=%s, tgt_idxs=%s, usize=%d, psize=%d): %s" %
                               (system.name, src_idxs, tgt_idxs,
                                system.unknowns.vec.size,
                                system.params.vec.size, str(err)))

    def transfer(self, srcvec, tgtvec, mode='fwd', deriv=False):
        """Performs data transfer between a distributed source vector and
        a distributed target vector.

        Parameters
        ----------
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

        mode : 'fwd' or 'rev', optional
            Direction of the data transfer, source to target ('fwd', the default)
            or target to source ('rev').
        """
        if mode == 'rev':
            # in reverse mode, srcvec and tgtvec are switched. Note, we only
            # run in reverse for derivatives, and derivatives accumulate from
            # all targets. This does not involve pass_by_object.
            if trace:
                for u,v in self.vec_conns:
                    debug("'%s': reverse scattering %s --> %s" % (srcvec.pathname,
                                                                  u, v))
            self.scatter.scatter(tgtvec.petsc_vec, srcvec.petsc_vec, True, True)
        else:
            # forward mode, source to target including pass_by_object
            if trace:
                for u,v in self.vec_conns:
                    debug("'%s': scattering %s --> %s" % (srcvec.pathname, v, u))
            self.scatter.scatter(srcvec.petsc_vec, tgtvec.petsc_vec, False, False)
            if trace: debug("scatter done")

            if not deriv:
                for tgt, src in self.byobj_conns:
                    debug('NotImplemented!!!')
                    raise NotImplementedError("can't transfer '%s' to '%s'" %
                                               (src, tgt))
