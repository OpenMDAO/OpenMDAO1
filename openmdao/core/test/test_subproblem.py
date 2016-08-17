
import sys
import unittest

from six import text_type, PY3
from six.moves import cStringIO

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Component, Problem, Group, IndepVarComp, ExecComp, \
                         Driver, ScipyOptimizer, CaseDriver, SubProblem, \
                         SqliteRecorder, NLGaussSeidel, ScipyGMRES
from openmdao.test.simple_comps import RosenSuzuki
from openmdao.test.example_groups import ExampleByObjGroup, ExampleGroup
from openmdao.test.util import assert_rel_error

#
# expected jacobian
#
expectedJ = {
    'subprob.comp.f': {
        'desvars.x': np.array([
            [ -3., -3., -17.,  9.]
        ])
    },
    'subprob.comp.g': {
        'desvars.x': np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ])
    }
}

expectedJ_array = np.concatenate((
    expectedJ['subprob.comp.f']['desvars.x'],
    expectedJ['subprob.comp.g']['desvars.x']
))


cylinder_opts = [('indep.r', 6.2035), ('indep.h', 12.407)]

class CylinderGroup(Group):
    def __init__(self):
        super(CylinderGroup, self).__init__()

        self.add("indep", IndepVarComp([('r', 1.0, {'units':'cm'}),
                                        ('h', 1.0, {'units':'cm'})]))

        self.add("cylinder", ExecComp(["area = 2.0*pi*r**2+2.0*pi*r*h",
                                       "volume = pi*r**2*h/1000."],
                                       units={'r':'cm','h':'cm',
                                              'volume':'L','area':'cm**2'}))
        self.connect("indep.r", "cylinder.r")
        self.connect("indep.h", "cylinder.h")


class ErrProb(Problem):
    """Raises an exception from the specified method."""
    def __init__(self, which_err=None, *args, **kwargs):
        super(ErrProb, self).__init__(*args, **kwargs)
        if which_err:
            setattr(self, which_err, self._raiseit)

    def _raiseit(self, *args, **kwargs):
        raise RuntimeError("Houston, we have a problem.")


class TestSubProblem(unittest.TestCase):

    def test_general_access(self):
        sprob = Problem(root=Group())
        sroot = sprob.root
        sroot.add('Indep', IndepVarComp('x', 7.0))
        sroot.add('C1', ExecComp(['y1=x1*2.0', 'y2=x2*3.0']))
        sroot.connect('Indep.x', 'C1.x1')

        prob = Problem(root=Group())
        prob.root.add('subprob', SubProblem(sprob,
                                            params=['Indep.x', 'C1.x2'],
                                            unknowns=['C1.y1', 'C1.y2']))

        prob.setup(check=False)

        prob['subprob.Indep.x'] = 99.0 # set a param that maps to an unknown in subproblem
        prob['subprob.C1.x2'] = 5.0  # set a dangling param

        prob.run()

        self.assertEqual(prob['subprob.C1.y1'], 198.0)
        self.assertEqual(prob['subprob.C1.y2'], 15.0)

    def test_simplest_run_w_promote(self):
        subprob = Problem(root=Group())
        subprob.root.add('indep', IndepVarComp('x', 7.0), promotes=['x'])
        subprob.root.add('mycomp', ExecComp('y=x*2.0'), promotes=['x','y'])

        prob = Problem(root=Group())
        prob.root.add('subprob', SubProblem(subprob, ['x'], ['y']))

        prob.setup(check=False)
        prob.run()
        result = prob.root.unknowns['subprob.y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_basic_run(self):
        subprob = Problem(root=ExampleGroup())

        prob = Problem(root=Group())
        prob.root.add('subprob', SubProblem(subprob, ['G2.C1.x'], ['G3.C4.y']))

        prob.setup(check=False)
        prob.run()

        self.assertAlmostEqual(prob['subprob.G3.C4.y'], 40.)

        stream = cStringIO()

        # get test coverage for list_connections and make sure it doesn't barf
        prob.root.subprob.list_connections(stream=stream)

    def test_byobj_run(self):
        subprob = Problem(root=ExampleByObjGroup())

        prob = Problem(root=Group())
        prob.root.add('subprob', SubProblem(subprob,
                                            params=['G2.C1.x'],
                                            unknowns=['G3.C4.y']))

        prob.setup(check=False)
        prob.run()

        self.assertEqual(prob['subprob.G3.C4.y'], 'fooC2C3C4')

    def test_calc_gradient(self):
        root = Group()
        root.add('indep', IndepVarComp('x', np.array([1., 1., 1., 1.])))
        root.add('comp', RosenSuzuki())

        root.connect('indep.x', 'comp.x')

        subprob = Problem(root)
        subprob.driver.add_desvar('indep.x', lower=-10, upper=99)
        subprob.driver.add_objective('comp.f')
        subprob.driver.add_constraint('comp.g', upper=0.)

        prob = Problem(root=Group())
        prob.root.add('desvars', IndepVarComp('x', np.ones(4)))
        prob.root.add('subprob', SubProblem(subprob,
                                            params=['indep.x'],
                                            unknowns=['comp.f', 'comp.g']))
        prob.root.connect('desvars.x', 'subprob.indep.x')

        prob.setup(check=False)
        prob.run()

        indep_list = ['desvars.x']
        unknown_list = ['subprob.comp.f', 'subprob.comp.g']

        # check that calc_gradient returns proper dict value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_almost_equal(J['subprob.comp.f']['desvars.x'],
                           expectedJ['subprob.comp.f']['desvars.x'])
        assert_almost_equal(J['subprob.comp.g']['desvars.x'],
                            expectedJ['subprob.comp.g']['desvars.x'])

        # check that calc_gradient returns proper array value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='array')
        assert_almost_equal(J, expectedJ_array)

        # check that calc_gradient returns proper dict value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_almost_equal(J['subprob.comp.f']['desvars.x'], expectedJ['subprob.comp.f']['desvars.x'])
        assert_almost_equal(J['subprob.comp.g']['desvars.x'], expectedJ['subprob.comp.g']['desvars.x'])

        # check that calc_gradient returns proper array value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='array')
        assert_almost_equal(J, expectedJ_array)

        # check that calc_gradient returns proper dict value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        assert_almost_equal(J['subprob.comp.f']['desvars.x'], expectedJ['subprob.comp.f']['desvars.x'], decimal=5)
        assert_almost_equal(J['subprob.comp.g']['desvars.x'], expectedJ['subprob.comp.g']['desvars.x'], decimal=5)

        # check that calc_gradient returns proper array value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='array')
        assert_almost_equal(J, expectedJ_array, decimal=5)

    def test_opt_cylinder(self):
        # this is just here to make sure that the cylinder optimization works normally so
        # we know if the opt of the nested cylinder fails it's not due to some cylinder
        # model issue.
        prob = Problem(root=CylinderGroup())
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = False

        prob.driver.add_desvar("indep.r", lower=0.0, upper=1.e99)
        prob.driver.add_desvar("indep.h", lower=0.0, upper=1.e99)
        prob.driver.add_objective("cylinder.area")
        prob.driver.add_constraint("cylinder.volume", equals=1.5)

        prob.setup(check=False)
        prob.run()

        for name, opt in cylinder_opts:
            self.assertAlmostEqual(prob[name], opt,
                                   places=4,
                                   msg="%s should be %s, but got %s" %
                                   (name, opt, prob[name]))

        self.assertAlmostEqual(prob['cylinder.volume'], 1.5,
                               places=4,
                               msg="volume should be 1.5, but got %s" %
                               prob['cylinder.volume'])

    def test_opt_cylinder_nested(self, which_err=None, check=False):
        prob = Problem(root=Group())
        driver = prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = False

        prob.driver.add_desvar("indep.r", lower=0.0, upper=1.e99)
        prob.driver.add_desvar("indep.h", lower=0.0, upper=1.e99)
        prob.driver.add_objective("subprob.cylinder.area")
        prob.driver.add_constraint("subprob.cylinder.volume", equals=1.5)

        # we need IndepVarComp for model params at top level because the top level
        # driver has them as design vars.
        prob.root.add("indep", IndepVarComp([('r', 1.0, {'units':'cm'}),
                                             ('h', 1.0, {'units':'cm'})]))

        subprob = ErrProb(which_err=which_err, root=CylinderGroup())
        prob.root.add('subprob', SubProblem(subprob,
                                params=['indep.r', 'indep.h'],
                                unknowns=['cylinder.area', 'cylinder.volume']))
        prob.root.connect('indep.r', 'subprob.indep.r')
        prob.root.connect('indep.h', 'subprob.indep.h')

        # we have to set check=True to test some error handling, but never
        # want to see the output, so just send it to a cStringIO
        prob.setup(check=check, out_stream=cStringIO())
        prob.run()

        self.assertAlmostEqual(prob['subprob.cylinder.volume'], 1.5,
                               places=4,
                               msg="volume should be 1.5, but got %s" %
                               prob['subprob.cylinder.volume'])

        for name, opt in cylinder_opts:
            self.assertAlmostEqual(prob[name], opt,
                                   places=4,
                                   msg="%s should be %s, but got %s" %
                                   (name, opt, prob[name]))

        prob.cleanup()

    def test_errors(self):
        errs = (
            'check_setup',
            'cleanup',
            'get_req_procs',
            'setup',
            'run',
            'calc_gradient'
        )

        for err in errs:
            check = err == 'check_setup'
            try:
                self.test_opt_cylinder_nested(which_err=err, check=check)
            except Exception as err:
                self.assertEqual(str(err),
                             "In subproblem 'subprob': Houston, we have a problem.")
            else:
                self.fail("Exception expected when '%s' failed" % err)

    def test_opt_cylinder_nested_w_promotes(self):
        prob = Problem(root=Group())
        driver = prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = False

        prob.driver.add_desvar("indep.r", lower=0.0, upper=1.e99)
        prob.driver.add_desvar("indep.h", lower=0.0, upper=1.e99)
        prob.driver.add_objective("cylinder.area")
        prob.driver.add_constraint("cylinder.volume", equals=1.5)

        # we need IndepVarComp for model params at top level because the top level
        # driver has them as design vars.
        prob.root.add("indep", IndepVarComp([('r', 1.0, {'units':'cm'}),
                                             ('h', 1.0, {'units':'cm'})]))

        subprob = Problem(root=CylinderGroup())
        prob.root.add('subprob', SubProblem(subprob,
                                params=list(prob.driver._desvars),
                                unknowns=list(driver._cons)+list(driver._objs)),
                                promotes=['indep.r', 'indep.h',
                                          'cylinder.area', 'cylinder.volume'])

        # the names of the indep vars match the promoted names from the subproblem, so
        # they're implicitly connected.

        prob.setup(check=False)
        prob.run()

        for name, opt in cylinder_opts:
            self.assertAlmostEqual(prob[name], opt,
                                   places=4,
                                   msg="%s should be %s, but got %s" %
                                   (name, opt, prob[name]))

        self.assertAlmostEqual(prob['cylinder.volume'], 1.5,
                               places=4,
                               msg="volume should be 1.5, but got %s" %
                               prob['cylinder.volume'])


if __name__ == "__main__":
    unittest.main()
