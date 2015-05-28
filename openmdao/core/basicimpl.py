from openmdao.core.vecwrapper import SrcVecWrapper, TgtVecWrapper
from openmdao.core.dataxfer import DataXfer

class BasicImpl(object):
    """Basic vector and data transfer implementation factory."""

    @staticmethod
    def create_src_vecwrapper(pathname, comm):
        """
        Create a vecwrapper for source variables.

        Parameters
        ----------
        comm : a fake communicator or None.
            This arg is ignored.

        Returns
        -------
        `SrcVecWrapper`
        """
        return SrcVecWrapper(pathname, comm)

    @staticmethod
    def create_tgt_vecwrapper(pathname, comm):
        """
        Create a vecwrapper for target variables.

        Parameters
        ----------
        comm : a fake communicator or None.
            This arg is ignored.

        Returns
        -------
        `TgtVecWrapper`
        """
        return TgtVecWrapper(pathname, comm)

    @staticmethod
    def create_data_xfer(varmanager, src_idxs, tgt_idxs, flat_conns, noflat_conns):
        """
        Create an object for performing data transfer between source
        and target vectors.

        Parameters
        ----------
        varmanager : `VarManager`
            The `VarManager` that managers this data transfer.

        src_idxs : array
            Indices of the source variables in the source vector.

        tgt_idxs : array
            Indices of the target variables in the target vector.

        flat_conns : dict
            Mapping of flattenable variables to the source variables that
            they are connected to.

        noflat_conns : dict
            Mapping of non-flattenable variables to the source variables that
            they are connected to.

        Returns
        -------
        `DataXfer`
            A `DataXfer` object.
        """
        return DataXfer(src_idxs, tgt_idxs, flat_conns, noflat_conns)

    @staticmethod
    def create_app_ordering(varmanager):
        pass
