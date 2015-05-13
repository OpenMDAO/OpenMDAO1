
from openmdao.core.vecwrapper import SrcVecWrapper, TgtVecWrapper
from openmdao.core.dataxfer import DataXfer

class PetscImpl(object):
    """PETSc vector and data transfer implementation factory"""

    @staticmethod
    def create_src_vecwrapper():
        """Create a`PetscSrcVecWrapper`

        Returns
        -------
        `PetscSrcVecWrapper`
        """
        return PetscSrcVecWrapper()

    @staticmethod
    def create_tgt_vecwrapper():
        """Create a `PetscTgtVecWrapper`

        Returns
        -------
        `PetscTgtVecWrapper`
        """
        return PetscTgtVecWrapper()

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


class PetscSrcVecWrapper(SrcVecWrapper):
    def _get_flattened_unknown_sizes(self):
        """
        Collect all flattenable var sizes.

        Returns
        -------
        ndarray
            array containing local sizes of flattenable unknown variables
        """
        sizes = [m['size'] for m in self.values() if not m.get('noflat')]
        return numpy.array([sizes])

    def _get_flattened_param_sizes(self):
        """
        Create a 1x1 numpy array to hold the sum of the sizes of local
        flattenable params.

        Returns
        -------
        ndarray
            array containing sum of local sizes of flattenable params.
        """
        psize = sum([m['size'] for m in self.params.values()
                     if m.get('owned') and not m.get('noflat')])
        return numpy.array([[psize]])

    def get_idxs(self, name):
        """Returns all of the indices for the named variable in this vector

        Parameters
        ----------
        name : str
            name of variable to get the indices for

        Returns
        -------
        ndarray
            Index array containing all indices (possibly distributed) for the named variable.
        """
        # TODO: add support for returning slice objects

        meta = self._vardict[name][0]
        if meta.get('noflat'):
            raise RuntimeError("No vector indices can be provided for non-flattenable variable '%s'" % name)

        start, end = self._slices[name]
        return self.make_idx_array(start, end)

    def norm(self):
        """ Calculates the norm of this vector.

        Returns
        -------
        float
            Norm of the flattenable values in this vector.
        """
        return norm(self.vec)

    def make_idx_array(self, start, end):
        """ Return an index vector of the right int type for
        parallel or serial computation.

        Parameters
        ----------
        start : int
            the starting index

        end : int
            the ending index
        """
        return numpy.arange(start, end, dtype=self.idx_arr_type)

class PetscTgtVecWrapper(TgtVecWrapper):
    pass

class PetscDataXfer(DataXfer):
    pass
