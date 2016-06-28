import unittest
import math

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp
from openmdao.test.util import assert_rel_error


class TestExecComp(unittest.TestCase):

    def test_bad_kwargs(self):
        prob = Problem(root=Group())
        try:
            C1 = prob.root.add('C1', ExecComp('y=x+1.', xx=2.0))
        except Exception as err:
            self.assertEqual(str(err), "Arg 'xx' in call to ExecComp() does not refer to any variable in the expressions ['y=x+1.']")

    def test_mixed_type(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp('y=numpy.sum(x)',
                                          x=np.arange(10,dtype=float)))
        self.assertTrue('x' in C1._init_params_dict)
        self.assertTrue('y' in C1._init_unknowns_dict)

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], 45.0, 0.00001)

    def test_simple(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp('y=x+1.', x=2.0))
        self.assertTrue('x' in C1._init_params_dict)
        self.assertTrue('y' in C1._init_unknowns_dict)

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], 3.0, 0.00001)

    def test_units(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp('y=x+z+1.', x=2.0, z=2.0, units={'x':'m','y':'m'}))

        self.assertTrue('units' in C1._init_params_dict['x'])
        self.assertTrue(C1._init_params_dict['x']['units'] == 'm')
        self.assertTrue('units' in C1._init_unknowns_dict['y'])
        self.assertTrue(C1._init_unknowns_dict['y']['units'] == 'm')
        self.assertFalse('units' in C1._init_params_dict['z'])

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], 5.0, 0.00001)

    def test_math(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp('y=sin(x)', x=2.0))
        self.assertTrue('x' in C1._init_params_dict)
        self.assertTrue('y' in C1._init_unknowns_dict)

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], math.sin(2.0), 0.00001)

    def test_array(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp('y=x[1]', x=np.array([1.,2.,3.]), y=0.0))
        self.assertTrue('x' in C1._init_params_dict)
        self.assertTrue('y' in C1._init_unknowns_dict)

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], 2.0, 0.00001)

    def test_array_lhs(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp(['y[0]=x[1]', 'y[1]=x[0]'],
                                          x=np.array([1.,2.,3.]), y=np.array([0.,0.])))
        self.assertTrue('x' in C1._init_params_dict)
        self.assertTrue('y' in C1._init_unknowns_dict)

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], np.array([2.,1.]), 0.00001)

    def test_simple_array_model(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', ExecComp(['y[0]=2.0*x[0]+7.0*x[1]',
                                        'y[1]=5.0*x[0]-3.0*x[1]'],
                                       x=np.zeros([2]), y=np.zeros([2])))

        prob.root.add('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        assert_rel_error(self, data['comp'][('y','x')]['abs error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][2], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][2], 0.0, 1e-5)

    def test_simple_array_model2(self):
        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', ExecComp('y = mat.dot(x)',
                                              x=np.zeros((2,)), y=np.zeros((2,)),
                                              mat=np.array([[2.,7.],[5.,-3.]])))

        p1 = prob.root.add('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        assert_rel_error(self, data['comp'][('y','x')]['abs error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['abs error'][2], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('y','x')]['rel error'][2], 0.0, 1e-5)

    def test_complex_step(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp(['y=2.0*x+1.'], x=2.0))

        self.assertTrue('x' in C1._init_params_dict)
        self.assertTrue('y' in C1._init_unknowns_dict)

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], 5.0, 0.00001)

        J = C1.linearize(C1.params, C1.unknowns, C1.resids)

        assert_rel_error(self, J[('y','x')], 2.0, 0.00001)

    def test_complex_step2(self):
        prob = Problem(Group())
        comp = prob.root.add('comp', ExecComp('y=x*x + x*2.0'))
        prob.root.add('p1', IndepVarComp('x', 2.0))
        prob.root.connect('p1.x', 'comp.x')

        comp.deriv_options['type'] = 'user'

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.y']['p1.x'], np.array([6.0]), 0.00001)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp.y']['p1.x'], np.array([6.0]), 0.00001)

    def test_abs_complex_step(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp('y=2.0*abs(x)', x=-2.0))

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], 4.0, 0.00001)

        # any negative C1.x should give a -2.0 derivative for dy/dx
        prob['C1.x'] = -1.0e-10
        J = C1.linearize(C1.params, C1.unknowns, C1.resids)
        assert_rel_error(self, J[('y','x')], -2.0, 0.00001)

        prob['C1.x'] = 3.0
        J = C1.linearize(C1.params, C1.unknowns, C1.resids)
        assert_rel_error(self, J[('y','x')], 2.0, 0.00001)

        prob['C1.x'] = 0.0
        J = C1.linearize(C1.params, C1.unknowns, C1.resids)
        assert_rel_error(self, J[('y','x')], 2.0, 0.00001)

    def test_abs_array_complex_step(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp('y=2.0*abs(x)',
                                          x=np.ones(3)*-2.0, y=np.zeros(3)))

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['y'], np.ones(3)*4.0, 0.00001)

        # any negative C1.x should give a -2.0 derivative for dy/dx
        prob['C1.x'] = np.ones(3)*-1.0e-10
        J = C1.linearize(C1.params, C1.unknowns, C1.resids)
        assert_rel_error(self, J[('y','x')], np.eye(3)*-2.0, 0.00001)

        prob['C1.x'] = np.ones(3)*3.0
        J = C1.linearize(C1.params, C1.unknowns, C1.resids)
        assert_rel_error(self, J[('y','x')], np.eye(3)*2.0, 0.00001)

        prob['C1.x'] = np.zeros(3)
        J = C1.linearize(C1.params, C1.unknowns, C1.resids)
        assert_rel_error(self, J[('y','x')], np.eye(3)*2.0, 0.00001)

        prob['C1.x'] = np.array([1.5, -0.6, 2.4])
        J = C1.linearize(C1.params, C1.unknowns, C1.resids)
        expect = np.zeros((3,3))
        expect[0,0] = 2.0
        expect[1,1] = -2.0
        expect[2,2] = 2.0

        assert_rel_error(self, J[('y','x')], expect, 0.00001)

    def test_colon_names(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp('a:y=a:x+1.+b', inits={'a:x':2.0}, b=0.5))
        self.assertTrue('a:x' in C1._init_params_dict)
        self.assertTrue('a:y' in C1._init_unknowns_dict)

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['a:y'], 3.5, 0.00001)

    def test_complex_step_colons(self):
        prob = Problem(root=Group())
        C1 = prob.root.add('C1', ExecComp(['foo:y=2.0*foo:bar:x+1.'], inits={'foo:bar:x':2.0}))

        self.assertTrue('foo:bar:x' in C1._init_params_dict)
        self.assertTrue('foo:y' in C1._init_unknowns_dict)

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, C1.unknowns['foo:y'], 5.0, 0.00001)

        J = C1.linearize(C1.params, C1.unknowns, C1.resids)

        assert_rel_error(self, J[('foo:y','foo:bar:x')], 2.0, 0.00001)

    def test_complex_step2_colons(self):
        prob = Problem(Group())
        comp = prob.root.add('comp', ExecComp('foo:y=foo:x*foo:x + foo:x*2.0'))
        prob.root.add('p1', IndepVarComp('x', 2.0))
        prob.root.connect('p1.x', 'comp.foo:x')

        comp.deriv_options['type'] = 'user'

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x'], ['comp.foo:y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.foo:y']['p1.x'], np.array([6.0]), 0.00001)

        J = prob.calc_gradient(['p1.x'], ['comp.foo:y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp.foo:y']['p1.x'], np.array([6.0]), 0.00001)

    def test_simple_array_model2_colons(self):
        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', ExecComp('foo:y = foo:mat.dot(x)',
                                              inits={'foo:y':np.zeros((2,)),
                                                     'foo:mat':np.array([[2.,7.],[5.,-3.]])},
                                              x=np.zeros((2,))))

        p1 = prob.root.add('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        assert_rel_error(self, data['comp'][('foo:y','x')]['abs error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['abs error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['abs error'][2], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['rel error'][0], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['rel error'][1], 0.0, 1e-5)
        assert_rel_error(self, data['comp'][('foo:y','x')]['rel error'][2], 0.0, 1e-5)

if __name__ == "__main__":
    unittest.main()
