from openmdao.core.vecwrapper import SrcVecWrapper, TgtVecWrapper
from openmdao.core.dataxfer import DataXfer

class BasicImpl(object):
    """Basic vector and data transfer implementation factory"""

    @staticmethod
    def create_src_vecwrapper(comm):
        """
        Create a vecwrapper for source variables.

        Parameters
        ----------
        comm : a fake communicator or None
            This arg is ignored

        Returns
        -------
        `SrcVecWrapper`
        """
        return SrcVecWrapper()

    @staticmethod
    def create_tgt_vecwrapper(comm):
        """
        Create a vecwrapper for target variables.

        Parameters
        ----------
        comm : a fake communicator or None
            This arg is ignored

        Returns
        -------
        `TgtVecWrapper`
        """
        return TgtVecWrapper()

    @staticmethod
    def createDataXfer(src_idxs, tgt_idxs, flat_conns, noflat_conns):
        """
        Create an object for performing data transfer between source
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
