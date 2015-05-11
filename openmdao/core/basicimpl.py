from openmdao.core.vecwrapper import VecWrapper
from openmdao.core.dataxfer import DataXfer

class BasicImpl(object):
    """Basic vector and data transfer implemenation factory"""

    @staticmethod
    def createVecWrapper():
        return VecWrapper()

    @staticmethod
    def createDataXfer(src_idxs, tgt_idxs, noflat_conns):
        return DataXfer(src_idxs, tgt_idxs, noflat_conns)
