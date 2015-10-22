"""
Baseclass for design-of-experiments drivers that have pre-determined parameter sets.
"""

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta

class PredeterminedRunsDriver(Driver):
    def __init__(self, *args, **kwargs):
        if type(self) == PredeterminedRunsDriver:
            raise Exception('PredeterminedRunsDriver is an abstract class')
        super(PredeterminedRunsDriver, self).__init__(*args, **kwargs)
        self.num_steps = 5 # TODO get this from somewhere

    def run(self, problem):

        # Let's iterate and run
        run_list = self._build_runlist()

        # Do the runs
        for run in run_list:
            for dv_name, dv_val in run.iteritems():
                self.set_desvar(dv_name, dv_val)

            metadata = create_local_meta(None, 'Driver')
            problem.root.solve_nonlinear(metadata=metadata)
            self.recorders.record_iteration(problem.root, metadata)

