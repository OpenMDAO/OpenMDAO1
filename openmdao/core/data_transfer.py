""" Class definition for the DataTransfer object."""

import numpy as np

from openmdao.util.array_util import to_slices
from openmdao.core.mpi_wrap import MPI

class DataTransfer(object):
    """
    An object that performs data transfer between a source vector and a
    target vector.

    Args
    ----
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
    """

    def __init__(self, src_idxs, tgt_idxs, vec_conns, byobj_conns, mode):

        self.src_idxs = src_idxs
        self.tgt_idxs = tgt_idxs
        self.vec_conns = vec_conns
        self.byobj_conns = byobj_conns

        if not MPI:
            # if in fwd mode, sort using src indices and in rev mode sort using tgt indices,
            # to increase the likelihood of slice conversion for 'get' access in order to
            # avoid array copies.
            if mode == 'fwd':
                self.src_idxs, self.tgt_idxs = to_slices(self.src_idxs, self.tgt_idxs)
            else:
                self.tgt_idxs, self.src_idxs = to_slices(self.tgt_idxs, self.src_idxs)
                if isinstance(self.src_idxs, slice):
                    self._src_unique = True
                else:
                    # check uniqueness of src_idxs to see if we can avoid calling np.add.at
                    self._src_unique = np.unique(self.src_idxs).size == self.src_idxs.size

    def transfer(self, srcvec, tgtvec, mode='fwd', deriv=False):
        """
        Performs data transfer between a source vector and a target vector.

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
            # all targets. byobjs are never scattered in reverse
            if self._src_unique:
                srcvec.vec[self.src_idxs] += tgtvec.vec[self.tgt_idxs]
            else:
                np.add.at(srcvec.vec, self.src_idxs, tgtvec.vec[self.tgt_idxs])
        else:
            tgtvec.vec[self.tgt_idxs] = srcvec.vec[self.src_idxs]
            # forward, include byobjs if not a deriv scatter
            if not deriv:
                for tgt, src in self.byobj_conns:
                    tgtvec[tgt] = srcvec[src]
