

class DataTransfer(object):
    def __init__(self, src_idxs, tgt_idxs, noflat_conns):
        # TODO: later, convert src_idxs and tgt_idxs to slices (or list of slices) to eliminate
        #       copying when running in a single process
        self.src_idxs = src_idxs
        self.tgt_idxs = tgt_idxs

        # TODO: convert noflat_conns to relative naming to avoid abs/rel conversion for
        #    every data xfer
        self.noflat_conns = noflat_conns

    def transfer(self, srcvec, tgtvec, mode):
        if mode == 'rev':
            # in reverse mode, srcvec and tgtvec are switched
            tgtvec.vec[self.src_idxs] += srcvec.vec[self.tgt_idxs]

            # noflats are never scattered in reverse, so skip that part

        else:  # forward
            tgtvec.vec[self.tgt_idxs][:] = srcvec.vec[self.src_idxs]

            for src, tgt in self.noflat_conns:
                tgtvec[tgt] = srcvec[src]

