"""Basic vector and data transfer implementation factory."""

from openmdao.core.vec_wrapper import SrcVecWrapper, TgtVecWrapper
from openmdao.core.data_transfer import DataTransfer


class BasicImpl(object):
    """Basic vector and data transfer implementation factory."""

    idx_arr_type = int
    
    @staticmethod
    def create_src_vecwrapper(pathname, comm):
        """
        Create a vecwrapper for source variables.

        Args
        ----
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

        Args
        -----
        comm : a fake communicator or None.
            This arg is ignored.

        Returns
        -------
        `TgtVecWrapper`
        """
        return TgtVecWrapper(pathname, comm)

    @staticmethod
    def create_data_xfer(src_vec, tgt_vec,
                         src_idxs, tgt_idxs, vec_conns, byobj_conns):
        """
        Create an object for performing data transfer between source
        and target vectors.

        Args
        ----
        src_vec : `VecWrapper`
            Source vecwrapper for the transfer.  In rev mode it will be the
            target.

        tgt_vec : `VecWrapper`
            Target vecwrapper for the transfer. In rev mode it will be the
            source.

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

        Returns
        -------
        `DataTransfer`
            A `DataTransfer` object.
        """
        return DataTransfer(src_idxs, tgt_idxs, vec_conns, byobj_conns)
