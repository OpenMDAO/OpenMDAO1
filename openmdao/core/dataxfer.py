
#from openmdao.util.arrayutil import to_slice

class DataXfer(object):
    def __init__(self, src_idxs, tgt_idxs, noflat_conns):
        # TODO: change to_slice to to_slices. (should never return an index array)
        #self.src_idxs = to_slice(src_idxs)
        #self.tgt_idxs = to_slice(tgt_idxs)
        self.src_idxs = src_idxs
        self.tgt_idxs = tgt_idxs
        self.noflat_conns = noflat_conns

    def transfer(self, srcvec, tgtvec, mode='fwd'):
        if mode == 'rev':
            # in reverse mode, srcvec and tgtvec are switched
            srcvec.vec[self.src_idxs] += tgtvec.vec[self.tgt_idxs]

            # noflats are never scattered in reverse, so skip that part

        else:  # forward
            tgtvec.vec[self.tgt_idxs] = srcvec.vec[self.src_idxs]

            for src, tgt in self.noflat_conns:
                tgtvec[tgt] = srcvec[src]

