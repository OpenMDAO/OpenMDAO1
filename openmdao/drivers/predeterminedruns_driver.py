"""
Baseclass for design-of-experiments Drivers that have pre-determined parameter sets.
"""

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta
from openmdao.core.parallel_doe_group import ParallelDOEGroup
from openmdao.util.array_util import evenly_distrib_idxs
from six import iteritems
import os

from openmdao.core.mpi_wrap import MPI

trace = os.environ.get('OPENMDAO_TRACE')
if trace: # pragma: no cover
    from openmdao.core.mpi_wrap import debug
else:
    def debug(*arg):
        pass

class PredeterminedRunsDriver(Driver):
    """ Baseclass for design-of-experiments Drivers that have pre-determined parameter sets.
    """

    def __init__(self):
        if type(self) == PredeterminedRunsDriver:
            raise Exception('PredeterminedRunsDriver is an abstract class')
        super(PredeterminedRunsDriver, self).__init__()

    def _setup(self, root):
        super(PredeterminedRunsDriver, self)._setup(root)
        if MPI and isinstance(root, ParallelDOEGroup): # pragma: no cover
            comm = root._full_comm
            job_list = None
            if comm.rank == 0:
                debug('Parallel DOE using %d threads' % (root._num_par_doe,))
                run_list = list(self._build_runlist()) # need to run the iterator
                run_sizes, run_offsets = evenly_distrib_idxs(root._num_par_doe, len(run_list))
                job_list = [run_list[o:o+s] for o, s in zip(run_offsets, run_sizes)]
            self.run_list = comm.scatter(job_list, root=0)
            debug('Number of DOE jobs: %s' % (len(self.run_list),))
        else:
            self.run_list = self._build_runlist()


    def run(self, problem):
        """Build a runlist and execute the Problem for each set of generated
        parameters.
        """
        self.iter_count = 0

        # For each runlist entry, run the system and record the results
        for run in self.run_list:

            for dv_name, dv_val in iteritems(run):
                self.set_desvar(dv_name, dv_val)

            metadata = create_local_meta(None, 'Driver')

            update_local_meta(metadata, (self.iter_count,))
            problem.root.solve_nonlinear(metadata=metadata)
            self.recorders.record_iteration(problem.root, metadata)
            self.iter_count += 1
