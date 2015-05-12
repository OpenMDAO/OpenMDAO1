from openmdao.core.vecwrapper import VecWrapper
from openmdao.core.dataxfer import DataXfer

class BasicImpl(object):
    """Basic vector and data transfer implementation factory"""

    @staticmethod
    def createVecWrapper():
        """Create an empty `VecWrapper`

        Returns
        -------
        `VecWrapper`
            an empty `VecWrapper`
        """
        return VecWrapper()

    @staticmethod
    def createDataXfer(src_idxs, tgt_idxs, flat_conns, noflat_conns):
        """Create an object for performing data transfer between source
        and target vectors

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

        Returns
        -------
        `DataXfer`
            a `DataXfer` object
        """
        return DataXfer(src_idxs, tgt_idxs, flat_conns, noflat_conns)
