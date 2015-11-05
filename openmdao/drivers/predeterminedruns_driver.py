"""
Baseclass for design-of-experiments Drivers that have pre-determined parameter sets.
"""

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta
from six import iteritems


class PredeterminedRunsDriver(Driver):
    """ Baseclass for design-of-experiments Drivers that have pre-determined parameter sets.
    """

    def __init__(self):
        if type(self) == PredeterminedRunsDriver:
            raise Exception('PredeterminedRunsDriver is an abstract class')
        super(PredeterminedRunsDriver, self).__init__()

    def run(self, problem):
        """Build a runlist and execute the Problem for each set of generated parameters.        
        """

        run_list = self._build_runlist()

        # For each runlist entry, run the system and record the results
        for run in run_list:
            for dv_name, dv_val in iteritems(run):
                self.set_desvar(dv_name, dv_val)

            metadata = create_local_meta(None, 'Driver')
            problem.root.solve_nonlinear(metadata=metadata)
            self.recorders.record_iteration(problem.root, metadata)
