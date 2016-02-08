""" Class definition for the DataTransfer object."""

from six.moves import zip

import numpy as np

from openmdao.util.array_util import to_slice
from openmdao.core.mpi_wrap import MPI
from openmdao.core.fileref import FileRef

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

    sysdata : `SysData` object
        The `SysData` object for the Group that will contain this
        `DataTransfer` object.

    mode : str
        Either 'fwd' or 'rev', indicating a forward or reverse scatter.
    """

    def __init__(self, src_idxs, tgt_idxs, vec_conns, byobj_conns, mode,
                 sysdata):
        self.vec_conns = vec_conns
        self.byobj_conns = byobj_conns
        self.sysdata = sysdata

        fwd = mode == 'fwd'

        # sort subarrays wrt each other in ascending order (not internally)
        # this assumes that subarrays are already sorted internally. The
        # only time this won't be true is if an unknown is connected to
        # a param with src_indices that are not sorted or have non unique
        # entries.  In this case we'll just use the array index we were
        # given and won't convert to a slice.
        if fwd:
            keyfunc = lambda l: l[0][0]
        else:
            keyfunc = lambda l: l[1][0]

        scatters = []
        for isrcs, itgts in sorted(zip(src_idxs, tgt_idxs), key=keyfunc):
            srcs = to_slice(isrcs)
            tgts = to_slice(itgts)

            if not fwd and not isinstance(srcs, slice):
                # check uniqueness of src_idxs to see if we can avoid
                # calling np.add.at, which is slower than +=
                src_unique = np.unique(srcs).size == srcs.size
            else:
                src_unique = True

            if scatters: # after the first iteration...
                # try to combine smaller slices into a larger one
                olds, oldt, sunique = scatters[-1]
                if isinstance(olds, slice) and isinstance(oldt, slice) and \
                     isinstance(srcs, slice) and isinstance(tgts, slice) and \
                     olds.stop == srcs.start and oldt.stop == tgts.start and \
                     olds.step == srcs.step and oldt.step == tgts.step:
                    news = slice(olds.start, srcs.stop, srcs.step)
                    newt = slice(oldt.start, tgts.stop, tgts.step)
                    scatters[-1] = (news, newt, src_unique)
                else:
                    scatters.append((srcs, tgts, src_unique))
            else:
                scatters.append((srcs, tgts, src_unique))

        self.scatters = scatters

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
            for isrcs, itgts, src_unique in self.scatters:
                if src_unique:
                    srcvec.vec[isrcs] += tgtvec.vec[itgts]
                else:
                    np.add.at(srcvec.vec, isrcs, tgtvec.vec[itgts])
        else:
            if tgtvec._probdata.in_complex_step:
                for isrcs, itgts, _ in self.scatters:
                    tgtvec.vec[itgts] = srcvec.vec[isrcs]
                    tgtvec.imag_vec[itgts] = srcvec.imag_vec[isrcs]
            else:
                for isrcs, itgts, _ in self.scatters:
                    tgtvec.vec[itgts] = srcvec.vec[isrcs]

            # forward, include byobjs if not a deriv scatter
            if not deriv:
                for tgt, src in self.byobj_conns:
                    if isinstance(srcvec[src], FileRef):
                        tgtvec[tgt]._assign_to(srcvec[src])
                    else:
                        tgtvec[tgt] = srcvec[src]
