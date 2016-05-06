from __future__ import print_function

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, Component
from openmdao.core.mpi_wrap import MPI
from openmdao.drivers.latinhypercube_driver import LatinHypercubeDriver
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    toprank = MPI.COMM_WORLD.rank
else:
    from openmdao.api import BasicImpl as impl
    toprank = 0



class DistribCompSimple(Component):
    """Uses 2 procs but takes full input vars"""

    def __init__(self, arr_size=2):
        super(DistribCompSimple, self).__init__()

        self._arr_size = arr_size
        self.add_param('invar', 0.)
        self.add_output('outvec', np.ones(arr_size, float))
        self.save = []

    def solve_nonlinear(self, params, unknowns, resids):
        if self.comm.rank == 0:
            unknowns['outvec'] = params['invar'] * np.ones(self._arr_size) * 0.25
        elif self.comm.rank == 1:
            unknowns['outvec'] = params['invar'] * np.ones(self._arr_size) * 0.5

        self.save.append((toprank, self.comm.rank,
                          params['invar'], unknowns['outvec']))

    def get_req_procs(self):
        return (2, 2)


class LHParDOETestCase(MPITestCase):
    N_PROCS = 4

    def test_lh_par_doe(self):
        prob = Problem(impl=impl)
        root = prob.root = Group()

        root.add('p1', IndepVarComp('invar', 0.), promotes=['*'])
        comp = root.add('comp', DistribCompSimple(2), promotes=['*'])

        prob.driver = LatinHypercubeDriver(4, num_par_doe=self.N_PROCS/2)

        prob.driver.add_desvar('invar', lower=-5.0, upper=5.0)

        prob.driver.add_objective('outvec')

        prob.setup(check=False)
        prob.run()

        if MPI:
            saves = self.comm.allgather(comp.save)
            for save in saves:
                self.assertEqual(len(save), 2)
            self.assertEqual(saves[0][0][2], saves[1][0][2])
            self.assertEqual(saves[0][1][2], saves[1][1][2])
            self.assertEqual(saves[2][0][2], saves[3][0][2])
            self.assertEqual(saves[2][1][2], saves[3][1][2])
            
