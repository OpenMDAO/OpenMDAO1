""" Example of using OpenMDAO to solve a variant of Queen Dido's problem.

Find the maximum area that can be enclosed within a given perimeter.

We will do this using rectangular integration where the fence posts are
fixed at (x,y) = (-50,0) and (50, 0).  We then seek the y-component of the
location for the fence posts at x = -49..49.

We do this by treating every meter in x as a rectangular section where
the area of the section is (delta-x * y).  We assume that delta-x is 1 m
in this example.

Is then the summation of the distance between two adjacent fenceposts:

perimeter_i = sqrt( (delta-x)**2 + (y[i]-y[i-1])**2) for i=1, n
"""
import numpy as np

from openmdao.api import Problem, Group, ParallelGroup, Component, \
                         IndepVarComp, ScipyOptimizer
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl

from openmdao.test.util import assert_rel_error


class RectangularSectionComp(Component):

    def __init__(self, dx):
        super(RectangularSectionComp, self).__init__()

        self.deriv_options['type'] = 'fd'
        self.dx = dx

        self.add_param('y', val=0.0,
            desc='y-component of the fencepost in the middle of this section',
            units='m')
        self.add_output('area', val=0.0,
                        desc='approximate area of this section', units='m**2')

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['area'] = self.dx * params['y']


class PerimeterComp(Component):

    def __init__(self, n, dx):
        super(PerimeterComp, self).__init__()

        self.deriv_options['type'] = 'fd'

        self.n = n
        self.dx = dx

        self.add_param('ys', val=np.zeros(n),
                       desc='y components of all fence posts', units='m')

        self.add_output('total_perimeter', val=0.0, desc='total perimeter',
                        units='m')

    def solve_nonlinear(self, params, unknowns, resids):

        ys = params['ys']
        unknowns['total_perimeter'] = 0.0

        for i in range(1, self.n):
            unknowns['total_perimeter'] += np.sqrt( (ys[i]-ys[i-1])**2 + self.dx**2)


class RectangleGroup(ParallelGroup):

    def __init__(self, n, dx):
        super(RectangleGroup, self).__init__()

        for i in range(n):
            self.add(name='section_{0}'.format(i), system=RectangularSectionComp(dx))


class Summer(Component):

    def __init__(self, n):

        super(Summer, self).__init__()

        self.deriv_options['type'] = 'fd'

        self.n = n

        for i in range(n):
            self.add_param(name='area_{0}'.format(i), val=0.0, units='m**2')

        self.add_output(name='total_area', val=0.0, units='m**2')

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['total_area'] = 0.0
        for i in range(self.n):
            unknowns['total_area'] += params['area_{0}'.format(i)]


class TestDido(MPITestCase):
    N_PROCS = 4

    def test_dido(self):

        prob = Problem(root=Group(), impl=impl, driver=ScipyOptimizer())

        # Total horizontal space of area to be enclosed.
        x = 100.0

        # Number of segments used to enclose area.
        n = 50

        # Horizonal size of each segment
        dx = x/n

        prob.root.add(name='ys_ivc',
                      system=IndepVarComp('ys', val=np.zeros(n), units='m'),
                      promotes=['ys'])
        prob.root.add(name='rec_group', system=RectangleGroup(n, dx))
        prob.root.add(name='total_area_comp', system=Summer(n),
                      promotes=['total_area'])
        prob.root.add(name='perimeter_comp', system=PerimeterComp(n, dx),
                      promotes=['ys', 'total_perimeter'])

        for i in range(n):
            prob.root.connect('ys', 'rec_group.section_{0}.y'.format(i),
                              src_indices=[i])
            prob.root.connect('rec_group.section_{0}.area'.format(i),
                              'total_area_comp.area_{0}'.format(i))

        idxs = range(n)[1:-1]

        prob.driver.add_desvar('ys', lower=np.zeros(n-2), indices=idxs)
        prob.driver.add_constraint('total_perimeter', upper=150)
        prob.driver.add_objective('total_area', scaler=-1.0E-3)

        prob.setup(check=False)

        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in data.items():
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

        assert_rel_error(self, 3574.94, prob['total_area'], 0.1)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
