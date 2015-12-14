"""
Baseclass for design-of-experiments Drivers that have pre-determined
parameter sets.
"""
from six.moves import zip

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta
from openmdao.util.array_util import evenly_distrib_idxs
import os

from openmdao.core.mpi_wrap import MPI

trace = os.environ.get('OPENMDAO_TRACE')
if trace: # pragma: no cover
    from openmdao.core.mpi_wrap import debug
else:
    def debug(*arg):
        pass

class PredeterminedRunsDriver(Driver):
    """
    Baseclass for design-of-experiments Drivers that have pre-determined
    parameter sets.

    Args
    ----
    num_par_doe : int, optional
        The number of DOE cases to run concurrently.  Defaults to 1.
    """

    def __init__(self, num_par_doe=1):
        if type(self) == PredeterminedRunsDriver:
            raise Exception('PredeterminedRunsDriver is an abstract class')
        super(PredeterminedRunsDriver, self).__init__()
        self._num_par_doe = num_par_doe
        self._par_doe_id = 0

    def _setup_communicators(self, comm):
        """
        Assign a communicator to the root `System`.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the Problem.
        """
        root = self.root
        if self._num_par_doe < 1:
            raise ValueError("'%s': _num_par_doe must be >= 1 but value is %s." %
                              (self.pathname, self._num_par_doe))
        if not MPI:
            self._num_par_doe = 1

        self._full_comm = comm

        # figure out which parallel DOE we are associated with
        if self._num_par_doe > 1:
            minprocs, maxprocs = root.get_req_procs()
            sizes, offsets = evenly_distrib_idxs(self._num_par_doe, comm.size)

            # a 'color' is assigned to each subsystem, with
            # an entry for each processor it will be given
            # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            color = []
            for i in range(self._num_par_doe):
                color.extend([i]*sizes[i])

            self._par_doe_id = color[comm.rank]

            # create a sub-communicator for each color and
            # get the one assigned to our color/process
            debug('%s: splitting comm, doe_id=%s' % ('.'.join((root.pathname,
                                                               'driver')),
                                                    self._par_doe_id))
            comm = comm.Split(self._par_doe_id)

        root._setup_communicators(comm)

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the
            min and max processors usable by this `Driver`.
        """
        minprocs, maxprocs = self.root.get_req_procs()

        minprocs *= self._num_par_doe
        if maxprocs is not None:
            maxprocs *= self._num_par_doe

        return (minprocs, maxprocs)

    def run(self, problem):
        """Build a runlist and execute the Problem for each set of generated
        parameters.
        """
        self.iter_count = 0

        if MPI and self._num_par_doe > 1:
            runlist = self._distrib_build_runlist()
        else:
            runlist = self._build_runlist()

        # For each runlist entry, run the system and record the results
        for run in runlist:
            for dv_name, dv_val in run:
                self.set_desvar(dv_name, dv_val)

            metadata = create_local_meta(None, 'Driver')

            update_local_meta(metadata, (self.iter_count,))
            problem.root.solve_nonlinear(metadata=metadata)
            self.recorders.record_iteration(problem.root, metadata)
            self.iter_count += 1

    def _distrib_build_runlist(self):
        """
        Returns an iterator over only those cases meant to execute
        in the current rank as part of a parallel DOE. _build_runlist
        will be called on all ranks, but only those cases targeted to
        this rank will run. Override this method
        (see LatinHypercubeDriver) if your DOE generator needs to
        create all cases on one rank and scatter them to other ranks.
        """
        for i, case in enumerate(self._build_runlist()):
            if i % self._num_par_doe == self._par_doe_id:
                yield case
