import unittest

import numpy as np

from openmdao.api import Problem, Group, ParallelGroup, \
                         Component, IndepVarComp, ExecComp, \
                         Driver, ScipyOptimizer, SqliteRecorder

from openmdao.test.sellar import *
from openmdao.test.util import assert_rel_error

from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl


class SellarDerivatives(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    with derivatives."""

    def __init__(self):
        super(SellarDerivatives, self).__init__()

        # params will be provided by parent group
        # self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        # self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        self.add('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        self.add('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0),
                 promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.nl_solver = NLGaussSeidel()
        self.ln_solver = ScipyGMRES()


class Randomize(Component):
    """ add random uncertainty to params and distribute

    Args
    ----
    n : number of points to generate for each param

    params : collection of (name, value, std_dev) specifying the params
             that are to be randommized.
    """
    def __init__(self, n=0, params=[]):
        super(Randomize, self).__init__()

        self.dists = {}

        for name, value, std_dev in params:
            # add param
            self.add_param(name, val=value)

            # add an output array var to distribute the modified param values
            if isinstance(value, np.ndarray):
                shape = (n, value.size)
            else:
                shape = (n, 1)

            # generate a standard normal distribution (size n) for this param
            self.dists[name] = std_dev*np.random.normal(0.0, 1.0, n*shape[1]).reshape(shape)

            self.add_output('dist_'+name, val=np.zeros(shape))

    def solve_nonlinear(self, params, unknowns, resids):
        """ add random uncertainty to params
        """
        for name, dist in self.dists.iteritems():
            unknowns['dist_'+name] = params[name] + dist

    def linearize(self, params, unknowns, resids):
        """ derivatives
        """
        J = {}
        for u in unknowns:
            name = u.split('_', 1)[1]
            for p in params:
                shape = (unknowns[u].size, params[p].size)
                if p == name:
                    J[u, p] = np.eye(shape[0], shape[1])
                else:
                    J[u, p] = np.zeros(shape)
        return J


class Collector(Component):
    """ collect the inputs and compute the mean of each

    Args
    ----
    n : number of points to collect for each input

    names : collection of `Str` specifying the names of the inputs to
            collect and the resulting outputs.
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

        for p in params:
            name = p.split('_', 1)[0]
            if name not in inputs:
                inputs[name] = data = [0.0, 0.0]
            else:
                data = inputs[name]
            data[0] += 1
            data[1] += params[p]

        for name in self.names:
            unknowns[name]  = inputs[name][1]/inputs[name][0]

    def linearize(self, params, unknowns, resids):
        """ derivatives
        """
        J = {}
        for p in params:
            name, idx = p.split('_', 1)
            for u in unknowns:
                if u == name:
                    J[u, p] = 1
                else:
                    J[u, p] = 0
        return J


class BruteForceSellarProblem(Problem):
    """ Performs optimization on the Sellar problem.

        Applies a normal distribution to the design vars and runs all of the
        samples, then collects the values of all of the outputs, calculates
        the mean of those and stuffs that back into the unknowns vector.

        This is the brute force version that just stamps out N separate
        sellar models in a parallel group and sets the input of each
        one to be one of these random design vars.

    Args
    ----
    n : number of randomized points to generate for each input value

    derivs : if True, use user-defined derivatives, else use Finite Difference
    """
    def __init__(self, n=10, derivs=False):
        super(BruteForceSellarProblem, self).__init__(impl=impl)

        root = self.root = Group()
        if not derivs:
            root.deriv_options['type'] = 'fd'

        sellars = root.add('sellars', ParallelGroup())
        for i in xrange(n):
            name = 'sellar%i' % i
            sellars.add(name, SellarDerivatives())

            root.connect('dist_x', 'sellars.'+name+'.x', src_indices=[i])
            root.connect('dist_z', 'sellars.'+name+'.z', src_indices=[i*2, i*2+1])

            root.connect('sellars.'+name+'.obj',  'collect.obj_%i'  % i)
            root.connect('sellars.'+name+'.con1', 'collect.con1_%i' % i)
            root.connect('sellars.'+name+'.con2', 'collect.con2_%i' % i)

        root.add('indep', IndepVarComp([
                    ('x', 1.0),
                    ('z', np.array([5.0, 2.0]))
                ]),
                promotes=['x', 'z'])

        root.add('random', Randomize(n=n, params=[
                    # name, value, std dev
                    ('x', 1.0, 1e-2),
                    ('z', np.array([5.0, 2.0]), 1e-2)
                ]),
                promotes=['x', 'z', 'dist_x', 'dist_z'])

        root.add('collect', Collector(n=n, names=['obj', 'con1', 'con2']),
                promotes=['obj', 'con1', 'con2'])

        # top level driver setup
        self.driver = ScipyOptimizer()
        self.driver.options['optimizer'] = 'SLSQP'
        self.driver.options['tol'] = 1.0e-8
        self.driver.options['maxiter'] = 50
        self.driver.options['disp'] = False

        self.driver.add_desvar('z', lower=np.array([-10.0,  0.0]),
                                    upper=np.array([ 10.0, 10.0]))
        self.driver.add_desvar('x', lower=0.0, upper=10.0)

        self.driver.add_objective('obj')
        self.driver.add_constraint('con1', upper=0.0)
        self.driver.add_constraint('con2', upper=0.0)

        # prob.driver.recorders.append(SqliteRecorder("sellar_bf%i.db" % n))


class TestSellar(MPITestCase):
    N_PROCS=4

    # nrange = [100, 200, 500, 1000, 2500, 5000]
    nrange = [100]

    def check_results(self, prob):
        """ check for the expected solution
        """
        tol = 1e-3
        assert_rel_error(self, prob['obj'],  3.183394, tol)
        assert_rel_error(self, prob['z'][0], 1.977639, tol)
        assert_rel_error(self, prob['z'][1], 0.0,      tol)
        assert_rel_error(self, prob['x'],    0.0,      tol)

    def test_brute_force_fd(self):
        """ brute force method without derivatives
        """
        for n in self.nrange:
            np.random.seed(42)
            prob = BruteForceSellarProblem(n, derivs=False)
            prob.setup(check=False)
            prob.run()
            print "Objective @ n=%i:\t" % n, prob['obj']
            if not MPI or self.comm.rank == 0:
                self.check_results(prob)

    def test_brute_force_derivs(self):
        """ brute force method with derivatives
        """
        for n in self.nrange:
            np.random.seed(42)
            prob = BruteForceSellarProblem(n, derivs=True)
            prob.setup(check=False)
            prob.run()
            print "Objective @ n=%i:\t" % n, prob['obj']
            if not MPI or self.comm.rank == 0:
                self.check_results(prob)

    def test_check_derivs(self):
        """ check derivatives on new components
        """
        raise unittest.SkipTest('check_derivs skipped')
        np.random.seed(42)
        prob = BruteForceSellarProblem(1, derivs=True)
        # remove optimizer
        prob.driver = Driver()
        # setup and check derivs
        prob.setup(check=False)
        prob.check_partial_derivatives(comps=['random', 'collect'])


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
