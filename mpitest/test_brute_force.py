import unittest

import numpy as np

from openmdao.api import Problem, Group, ParallelGroup, \
                         Component, IndepVarComp, ExecComp, \
                         ScipyOptimizer, SqliteRecorder

from openmdao.test.sellar import *
from openmdao.test.util import assert_rel_error

from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl


class SellarNoDerivatives(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    without derivatives."""

    def __init__(self):
        super(SellarNoDerivatives, self).__init__()

        # params will be provided by parent group
        # self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        # self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        cycle = self.add('cycle', Group(), promotes=['x', 'z', 'y1', 'y2'])
        cycle.ln_solver = ScipyGMRES()
        cycle.add('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        cycle.add('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0),
                 promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.nl_solver = NLGaussSeidel()
        self.cycle.d1.deriv_options['type'] = 'fd'
        self.cycle.d2.deriv_options['type'] = 'fd'


class Randomize(Component):
    """ add random uncertainty to params
    """
    def __init__(self, n=0, params=[]):
        super(Randomize, self).__init__()

        self.dists = {}

        for name, value, std_dev in params:
            # add param
            self.add_param(name, val=value)

            # generate a standard normal distribution (size n) for this param
            self.dists[name] = std_dev*np.random.normal(0.0, 1.0, n).reshape(n, 1)

            # add an output array var to distribute the modified param values
            if isinstance(value, np.ndarray):
                shape = (n, value.size)
            else:
                shape = (n, 1)
            self.add_output('dist_'+name, val=np.zeros(shape))

    def solve_nonlinear(self, params, unknowns, resids):
        """ add random uncertainty to params
        """
        for name, dist in self.dists.iteritems():
            unknowns['dist_'+name] = params[name] + dist


class Collector(Component):
    """ collect the inputs and compute the mean of each
    """
    def __init__(self, n=10, names=[]):
        super(Collector, self).__init__()

        self.names = names

        # create n params for each input
        for i in xrange(n):
            for name in names:
                self.add_param('%s_%i' % (name, i),  val=0.)

        # create an output for the mean of each input
        for name in names:
            self.add_output(name,  val=0.)

    def solve_nonlinear(self, params, unknowns, resids):
        """ compute the mean of each input
        """
        inputs = {}

        for p in params.iterkeys():
            name = p.split('_', 1)[0]
            if name not in inputs:
                inputs[name] = data = [0.0, 0.0]
            else:
                data = inputs[name]
            data[0] += 1
            data[1] += params[p]

        for name in self.names:
            unknowns[name]  = inputs[name][1]/inputs[name][0]


class BruteForceSellar(Group):
    """ Applies a normal distribution to the design vars and runs all of the
        samples, then collects the values of all of the outputs, calculates
        the mean of those and stuffs that back into the unknowns vector.

        This is the brute force version that just stamps out N separate
        sellar models in a parallel group and sets the input of each
        one to be one of these random design vars.
    """
    def __init__(self, n=10):
        super(BruteForceSellar, self).__init__()

        self.add('indep', IndepVarComp([
                    ('x', 1.0),
                    ('z', np.array([5.0, 2.0]))
                ]),
                promotes=['x', 'z'])

        self.add('random', Randomize(n=n, params=[
                    # name, value, std dev
                    ('x', 1.0, 1e-2),
                    ('z', np.array([5.0, 2.0]), 1e-2)
                ]),
                promotes=['x', 'z', 'dist_x', 'dist_z'])

        self.add('collect', Collector(n=n, names=['obj', 'con1', 'con2']),
                promotes=['obj', 'con1', 'con2'])

        sellars = self.add('sellars', ParallelGroup())
        for i in xrange(n):
            name = 'sellar%i' % i
            sellars.add(name, SellarNoDerivatives())

            self.connect('dist_x', 'sellars.'+name+'.x', src_indices=[i])
            self.connect('dist_z', 'sellars.'+name+'.z', src_indices=[i*2, i*2+1])

            self.connect('sellars.'+name+'.obj',  'collect.obj_%i'  % i)
            self.connect('sellars.'+name+'.con1', 'collect.con1_%i' % i)
            self.connect('sellars.'+name+'.con2', 'collect.con2_%i' % i)


class BruteForceSellarProblem(Problem):
    def __init__(self, n=10):
        super(BruteForceSellarProblem, self).__init__(root=BruteForceSellar(n), impl=impl)

        self.root.deriv_options['type'] = 'fd'

        self.driver = ScipyOptimizer()
        self.driver.options['optimizer'] = 'SLSQP'
        self.driver.options['tol'] = 1.0e-8
        # prob.driver.options['maxiter'] = 10
        # prob.driver.options['disp'] = False

        self.driver.add_desvar('z', lower=np.array([-10.0,  0.0]),
                                    upper=np.array([ 10.0, 10.0]))
        self.driver.add_desvar('x', lower=0.0, upper=10.0)

        self.driver.add_objective('obj')
        self.driver.add_constraint('con1', upper=0.0)
        self.driver.add_constraint('con2', upper=0.0)

        # prob.driver.recorders.append(SqliteRecorder("sellar_bf%i.db" % n))


class TestSellar(MPITestCase):
    N_PROCS=1

    def test_brute_force_(self):
        np.random.seed(42)

        for n in [100, 200, 500]:  #, 1000, 2500, 5000]:
            prob = BruteForceSellarProblem(n)
            prob.setup(check=False)
            prob.run()
            if not MPI or self.comm.rank == 0:
                print "Objective @ n=%i:\t" % n, prob['obj']
                assert_rel_error(self, prob['obj'],  3.183394, 1e-3)
                assert_rel_error(self, prob['z'][0], 1.977639, 1e-3)
                assert_rel_error(self, prob['z'][1], 0.0,      1e-3)
                assert_rel_error(self, prob['x'],    0.0,      1e-3)


if __name__ == "__main__":
    unittest.main()
