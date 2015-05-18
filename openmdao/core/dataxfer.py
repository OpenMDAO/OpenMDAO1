
from six.moves import zip

import numpy as np

#from openmdao.util.arrayutil import to_slice

class DataXfer(object):
    """
    An object that performs data transfer between a source vector and a
    target vector.
    """
    def __init__(self, src_idxs, tgt_idxs, flat_conns, noflat_conns):
        """
        Parameters
        ----------
        src_idxs : array
            indices of the source variables in the source vector

        tgt_idxs : array
            indices of the target variables in the target vector

        flat_conns : dict
            mapping of flattenable variables to the source variables that
            they are connected to

        noflat_conns : dict
            mapping of non-flattenable variables to the source variables that
            they are connected to
        """

        # TODO: change to_slice to to_slices. (should never return an index array)
        #self.src_idxs = to_slice(src_idxs)
        #self.tgt_idxs = to_slice(tgt_idxs)

        self.src_idxs = src_idxs
        self.tgt_idxs = tgt_idxs
        self.flat_conns = flat_conns
        self.noflat_conns = noflat_conns

    def transfer(self, srcvec, tgtvec, mode='fwd', deriv=False):
        """
        Performs data transfer between a source vector and a target vector.

        Parameters
        ----------
        src_idxs : array
            indices of the source variables in the source vector

        tgt_idxs : array
            indices of the target variables in the target vector

        flat_conns : dict
            mapping of flattenable variables to the source variables that
            they are connected to

        noflat_conns : dict
            mapping of non-flattenable variables to the source variables that
            they are connected to

        mode : 'fwd' or 'rev' (optional)
            direction of the data transfer, source to target ('fwd', the default)
            or target to source ('rev')

        deriv : bool
            If True, this is a derivative scatter, so noflats should not be transferred
        """
        if mode == 'rev':
            # in reverse mode, srcvec and tgtvec are switched. Note, we only
            # run in reverse for derivatives, and derivatives accumulate from
            # all targets. This requires numpy's new add command.
            np.add.at(srcvec.vec, self.src_idxs, tgtvec.vec[self.tgt_idxs])
            print "rev:",self.tgt_idxs,'-->',self.src_idxs, self.flat_conns, 'noflat',self.noflat_conns

            # formerly
            #srcvec.vec[self.src_idxs] += tgtvec.vec[self.tgt_idxs]

            # noflats are never scattered in reverse, so skip that part

        else:  # forward
            tgtvec.vec[self.tgt_idxs] = srcvec.vec[self.src_idxs]
            print "fwd:",self.src_idxs,'-->',self.tgt_idxs, self.flat_conns, 'noflat',self.noflat_conns

            if not deriv:
                for tgt, src in self.noflat_conns:
                    tgtvec[tgt] = srcvec[src]
