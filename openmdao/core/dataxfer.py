""" Class definition for the DataXfer object."""

import numpy as np


class DataXfer(object):
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
    """

    def __init__(self, src_idxs, tgt_idxs, vec_conns, byobj_conns):

        # TODO: change to_slice to to_slices. (should never return an index array)
        #self.src_idxs = to_slice(src_idxs)
        #self.tgt_idxs = to_slice(tgt_idxs)

        self.src_idxs = src_idxs
        self.tgt_idxs = tgt_idxs
        self.vec_conns = vec_conns
        self.byobj_conns = byobj_conns

    def transfer(self, srcvec, tgtvec, mode='fwd', deriv=False):
        """
        Performs data transfer between a source vector and a target vector.

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

        mode : 'fwd' or 'rev', optional
            Direction of the data transfer, source to target ('fwd', the
            default) or target to source ('rev').

        deriv : bool, optional
            If True, this is a derivative scatter, so byobjs should not be
            transferred.
        """
        if mode == 'rev':
            # in reverse mode, srcvec and tgtvec are switched. Note, we only
            # run in reverse for derivatives, and derivatives accumulate from
            # all targets. byobjs are never scattered in reverse
            np.add.at(srcvec.vec, self.src_idxs, tgtvec.vec[self.tgt_idxs])
        else:
            # forward, include byobjs if not a deriv scatter
            tgtvec.vec[self.tgt_idxs] = srcvec.vec[self.src_idxs]
            if not deriv:
                for tgt, src in self.byobj_conns:
                    tgtvec[tgt] = srcvec[src]
