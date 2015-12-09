""" Defines the base class for a ParallelDOEGroup in OpenMDAO.
It is used as the top-level system for running parallel DOE sweeps"""

from __future__ import print_function

import os
from six import itervalues

from openmdao.core.group import Group
from openmdao.util.array_util import evenly_distrib_idxs
from openmdao.core.mpi_wrap import MPI

trace = os.environ.get('OPENMDAO_TRACE')
if trace: # pragma: no cover
    from openmdao.core.mpi_wrap import debug
else:
    def debug(*arg):
        pass

class ParallelDOEGroup(Group):
    """A Group that can run DOE in parallel.

    Args
    ----
    num_par_doe : int(1)
        Number of DOE's to perform in parallel.  If num_par_doe is 1,
        this just behaves like a normal Group.

    """
    def __init__(self, num_par_doe):
        super(ParallelDOEGroup, self).__init__()

        self._num_par_doe = num_par_doe
        self._par_doe_id = 0

    def _setup_communicators(self, comm):
        """
        Assign communicator to this `Group` and all of its subsystems.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.
        """
        if self._num_par_doe < 1:
            raise ValueError("'%s': _num_par_doe must be >= 1 but value is %s." %
                              (self.pathname, self._num_par_doe))
        if not MPI:
            self._num_par_doe = 1

        self._full_comm = comm

        # figure out which parallel DOE we are associated with
        if self._num_par_doe > 1:
            minprocs, maxprocs = super(ParallelDOEGroup, self).get_req_procs()
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
            debug('%s: splitting comm, doe_id=%s' % (self.pathname,
                                                    self._par_doe_id))
            comm = comm.Split(self._par_doe_id)

        self._local_subsystems = []

        self.comm = comm

        for sub in itervalues(self._subsystems):
            sub._setup_communicators(comm)
            if self.is_active() and sub.is_active():
                self._local_subsystems.append(sub)

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the
            min and max processors usable by this `Group`.
        """
        minprocs, maxprocs = super(ParallelDOEGroup, self).get_req_procs()

        minprocs *= self._num_par_doe
        if maxprocs is not None:
            maxprocs *= self._num_par_doe

        return (minprocs, maxprocs)
