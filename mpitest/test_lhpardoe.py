import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, Component
from openmdao.core.mpi_wrap import MPI
from openmdao.drivers.latinhypercube_driver import LatinHypercubeDriver
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    rank = MPI.COMM_WORLD.rank
else:
    from openmdao.api import BasicImpl as impl
    rank = 0



class DistribCompSimple(Component):
    """Uses 2 procs but takes full input vars"""

    def __init__(self, arr_size=2):
        super(DistribCompSimple, self).__init__()

        self._arr_size = arr_size
        self.add_param('invar', 0.)
        self.add_output('outvec', np.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        if rank == 0:
            unknowns['outvec'] = params['invar'] * np.ones(self._arr_size) * 0.25
        elif rank == 1:
            unknowns['outvec'] = params['invar'] * np.ones(self._arr_size) * 0.5

        print 'hello from rank', rank, unknowns['outvec']

    def get_req_procs(self):
        return (2, 2)


class LHParDOETestCase(MPITestCase):
    N_PROCS = 4

    def test_lh_par_doe(self):
        prob = Problem(impl=impl)
        root = prob.root = Group()

        root.add('p1', IndepVarComp('invar', 0.), promotes=['*'])
        root.add('comp', DistribCompSimple(2), promotes=['*'])

        prob.driver = LatinHypercubeDriver(4, num_par_doe=self.N_PROCS/2)

        prob.driver.add_desvar('invar', lower=-5.0, upper=5.0)

        prob.driver.add_objective('outvec')

        prob.setup(check=False)
        prob.run()
