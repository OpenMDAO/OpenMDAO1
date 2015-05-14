
from openmdao.core.vecwrapper import SrcVecWrapper, TgtVecWrapper
from openmdao.core.dataxfer import DataXfer

from petsc4py import PETSc

class PetscImpl(object):
    """PETSc vector and data transfer implementation factory"""

    @staticmethod
    def create_src_vecwrapper(comm):
        """Create a`PetscSrcVecWrapper`

        Returns
        -------
        `PetscSrcVecWrapper`
        """
        return PetscSrcVecWrapper(comm)

    @staticmethod
    def create_tgt_vecwrapper(comm):
        """Create a `PetscTgtVecWrapper`

        Returns
        -------
        `PetscTgtVecWrapper`
        """
        return PetscTgtVecWrapper(comm)

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

    idx_arr_type = PETSc.IntType

    def __init__(self, comm=comm):
        super(PetscSrcVecWrapper, self).__init__()
        self.comm = comm
        
    def setup(self, unknowns_dict):
        """
        Create internal data storage for variables in unknowns_dict.

        Parameters
        ----------
        unknowns_dict : `OrderedDict`
            A dictionary of absolute variable names keyed to an associated
            metadata dictionary.
        """
        super(PetscSrcVecWrapper, self).setup(unknowns_dict)
        self.petsc_vec = PETSc.Vec().createWithArray(self.vec, comm=self.comm)

    def _get_flattened_sizes(self):
        """
        Collect all flattenable var sizes.

        Returns
        -------
        ndarray
            array containing local sizes of flattenable unknown variables
        """
        sizes = [m['size'] for m in self.values() if not m.get('noflat')]

        # create 2D array of variable sizes per process
        self.local_unknown_sizes = numpy.zeros((self.comm.size, len(sizes)), int)

        # create a vec indicating whether a nonflat variable is active
        # in this rank or not
        #self.noflat_isactive = numpy.zeros((size, len(self.noflat_vars)), int)

        # create our row in the local_unknown_sizes table
        ours = numpy.zeros((1, len(sizes)), int)
        for i, (name, meta) in enumerate(self.get_vecvars()):
            ours[0, i] = meta['size']

        our_noflats = numpy.zeros((1, len(self.get_noflats())), int)
        for i, (name, meta) in enumerate(self.get_noflats()):
            our_noflats[0, i] = int(self.is_variable_local(name[0]))

        # collect local var sizes from all of the processes in our comm
        # these sizes will be the same in all processes except in cases
        # where a variable belongs to a multiprocessor component.  In that
        # case, the part of the component that runs in a given process will
        # only have a slice of each of the component's variables.
        comm.Allgather(ours[0,:], self.local_unknown_sizes)
        comm.Allgather(our_noflats[0,:], self.noflat_isactive)

        self.local_unknown_sizes[rank, :] = ours[0, :]

        return numpy.array([sizes])

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
        meta = self._vardict[name][0]
        if meta.get('noflat'):
            raise RuntimeError("No vector indices can be provided for non-flattenable variable '%s'" % name)

        start, end = self._slices[name]
        return self.make_idx_array(start, end)

    def norm(self):
        """
        Returns
        -------
        float
            The norm of the distributed vector
        """
        self.petsc_vec.assemble()
        return self.petsc_vec.norm()


class PetscTgtVecWrapper(TgtVecWrapper):
    idx_arr_type = PETSc.IntType

    def __init__(self, comm=comm):
        super(PetscTgtVecWrapper, self).__init__()
        self.comm = comm
        
    def _get_flattened_sizes(self):
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


class PetscDataXfer(DataXfer):
    pass
