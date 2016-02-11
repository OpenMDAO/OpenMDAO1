"""
Baseclass for design-of-experiments Drivers that have pre-determined
parameter sets.
"""
import os
import traceback
from six.moves import zip
from six import next

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta
from openmdao.util.array_util import evenly_distrib_idxs
from openmdao.core.mpi_wrap import MPI, debug

trace = os.environ.get('OPENMDAO_TRACE')

class PredeterminedRunsDriver(Driver):
    """
    Baseclass for design-of-experiments Drivers that have pre-determined
    parameter sets.

    Args
    ----
    num_par_doe : int, optional
        The number of DOE cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Default is False.
    """

    def __init__(self, num_par_doe=1, load_balance=False):
        if type(self) == PredeterminedRunsDriver:
            raise Exception('PredeterminedRunsDriver is an abstract class')
        super(PredeterminedRunsDriver, self).__init__()
        self._num_par_doe = num_par_doe
        self._par_doe_id = 0
        self._load_balance = load_balance

    def _setup_communicators(self, comm, parent_dir):
        """
        Assign a communicator to the root `System`.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the Problem.

        parent_dir : str
            Absolute dir of parent `System`.
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
            if self._load_balance:
                sizes, offsets = evenly_distrib_idxs(self._num_par_doe-1,
                                                     comm.size-1)
                sizes = [1]+list(sizes)
                offsets = [0]+[o+1 for o in offsets]
            else:
                sizes, offsets = evenly_distrib_idxs(self._num_par_doe,
                                                     comm.size)

            # a 'color' is assigned to each subsystem, with
            # an entry for each processor it will be given
            # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            color = []
            self._id_map = {}
            for i in range(self._num_par_doe):
                color.extend([i]*sizes[i])
                self._id_map[i] = (sizes[i], offsets[i])

            self._par_doe_id = color[comm.rank]

            # create a sub-communicator for each color and
            # get the one assigned to our color/process
            if trace:
                debug('%s: splitting comm, doe_id=%s' % ('.'.join((root.pathname,
                                                               'driver')),
                                                    self._par_doe_id))
            comm = comm.Split(self._par_doe_id)

        root._setup_communicators(comm, parent_dir)

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
            if self._load_balance:
                runlist = self._distrib_lb_build_runlist()
                if self._full_comm.rank == 0:
                    try:
                        next(runlist)
                    except StopIteration:
                        pass
                    return # we're done sending cases
            else:
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
            if (i % self._num_par_doe) == self._par_doe_id:
                yield case

    def _distrib_lb_build_runlist(self):
        """
        Runs a load balanced version of the runlist, with the master
        rank (0) sending a new case to each worker rank as soon as it
        has finished its last case.
        """
        comm = self._full_comm

        if self._full_comm.rank == 0:  # master rank
            runiter = self._build_runlist()
            received = 0
            sent = 0

            # cases left for each par doe
            cases = {n:[] for n in self._id_map}

            # create a mapping of ranks to doe_ids
            doe_ids = {}
            for doe_id, tup in self._id_map.items():
                size, offset = tup
                for i in range(size):
                    doe_ids[i+offset] = doe_id

            # seed the workers
            for i in range(1, self._num_par_doe):
                try:
                    # case is a generator, so must make a list to send
                    case = list(next(runiter))
                except StopIteration:
                    break
                size, offset = self._id_map[i]
                for j in range(size):
                    comm.send(case, j+offset, tag=1)
                    cases[i].append(case)
                    sent += 1

            # send the rest of the cases
            while True:
                if sent == 0:
                    break
                worker = comm.recv()
                received += 1
                clist = cases[doe_ids[worker]]
                clist.pop()
                if not clist:
                    # we've received case from all procs with that doe_id
                    try:
                        case = list(next(runiter))
                    except StopIteration:
                        break
                    size, offset = self._id_map[doe_ids[worker]]
                    for j in range(size):
                        comm.send(case, j+offset, tag=1)
                        cases[doe_ids[worker]].append(case)
                        sent += 1

            # receive all leftover worker replies
            while received < sent:
                worker = comm.recv()
                received += 1

            # tell all workers to stop
            for rank in range(1, self._full_comm.size):
                comm.isend(None, rank, tag=1)

        else:   # worker
            while True:
                # wait on a case from the master
                case = comm.recv(source=0, tag=1)
                if case is None:
                    break
                # yield the case so it can be executed
                yield case
                # tell the master we're done with that case
                comm.isend(comm.rank, 0)
