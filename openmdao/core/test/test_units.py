""" Tests the ins and outs of automatic unit conversion in OpenMDAO."""

import unittest
from six import iteritems
from six.moves import cStringIO

import numpy as np

from openmdao.api import IndepVarComp, Component, Group, Problem, ExecComp
from openmdao.test.util import assert_rel_error


class SrcComp(Component):

    def __init__(self):
        super(SrcComp, self).__init__()

        self.add_param('x1', 100.0)
        self.add_output('x2', 100.0, units='degC')

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x2'] = params['x1']

    def linearize(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('x2', 'x1')] = np.array([1.0])
        return J


class TgtCompF(Component):

    def __init__(self):
        super(TgtCompF, self).__init__()

        self.add_param('x2', 100.0, units='degF')
        self.add_output('x3', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']

    def linearize(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('x3', 'x2')] = np.array([1.0])
        return J


class TgtCompFMulti(Component):
    # Some extra inputs that might trip things up.

    def __init__(self):
        super(TgtCompFMulti, self).__init__()

        self.add_param('_x2', 100.0, units='degF')
        self.add_param('x2', 100.0, units='degF')
        self.add_param('x2_', 100.0, units='degF')
        self.add_output('_x3', 100.0)
        self.add_output('x3', 100.0)
        self.add_output('x3_', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']

    def linearize(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('_x3', 'x2')] = np.array([1.0])
        J[('_x3', '_x2')] = 0.0
        J[('_x3', 'x2_')] = 0.0
        J[('x3', 'x2')] = np.array([1.0])
        J[('x3', '_x2')] = 0.0
        J[('x3', 'x2_')] = 0.0
        J[('x3_', 'x2')] = np.array([1.0])
        J[('x3_', '_x2')] = 0.0
        J[('x3_', 'x2_')] = 0.0
        return J


class TgtCompC(Component):

    def __init__(self):
        super(TgtCompC, self).__init__()

        self.add_param('x2', 100.0, units='degC')
        self.add_output('x3', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']

    def linearize(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('x3', 'x2')] = np.array([1.0])
        return J


class TgtCompK(Component):

    def __init__(self):
        super(TgtCompK, self).__init__()

        self.add_param('x2', 100.0, units='degK')
        self.add_output('x3', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']

    def linearize(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('x3', 'x2')] = np.array([1.0])
        return J


class TestUnitConversion(unittest.TestCase):
    """ Testing automatic unit conversion."""

    def test_basic(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgtF', TgtCompF())
        prob.root.add('tgtC', TgtCompC())
        prob.root.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')
        prob.root.connect('src.x2', 'tgtC.x2')
        prob.root.connect('src.x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Make sure we don't convert equal units
        self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         None)

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Need to clean up after FD gradient call, so just rerun.
        prob.run()

        # Make sure check partials handles conversion
        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-6)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-6)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-6)

        stream = cStringIO()
        conv = prob.root.list_unit_conv(stream=stream)
        self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)
        self.assertTrue((('src.x2', 'tgtK.x2'), ('degC', 'degK')) in conv)

    def test_list_unit_conversions_no_unit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.add('src', SrcComp())
        prob.root.add('tgt', ExecComp('yy=xx', xx=0.0))
        prob.root.connect('src.x2', 'tgt.xx')

        prob.setup(check=False)
        prob.run()

        stream = cStringIO()
        conv = prob.root.list_unit_conv(stream=stream)
        self.assertTrue((('src.x2', 'tgt.xx'), ('degC', None)) in conv)

    def test_basic_input_input(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgtF', TgtCompF())
        prob.root.add('tgtC', TgtCompC())
        prob.root.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtC.x2')
        prob.root.connect('tgtC.x2', 'tgtF.x2')
        prob.root.connect('tgtC.x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Make sure we don't convert equal units
        self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         None)

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_implicit_conn(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp(), promotes=['x1', 'x2'])
        prob.root.add('tgtF', TgtCompF(), promotes=['x2'])
        prob.root.add('tgtC', TgtCompC(), promotes=['x2'])
        prob.root.add('tgtK', TgtCompK(), promotes=['x2'])
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Make sure we don't convert equal units
        self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         None)

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_grouped(self):

        prob = Problem()
        prob.root = Group()
        sub1 = prob.root.add('sub1', Group())
        sub2 = prob.root.add('sub2', Group())
        sub1.add('src', SrcComp())
        sub2.add('tgtF', TgtCompF())
        sub2.add('tgtC', TgtCompC())
        sub2.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'sub1.src.x1')
        prob.root.connect('sub1.src.x2', 'sub2.tgtF.x2')
        prob.root.connect('sub1.src.x2', 'sub2.tgtC.x2')
        prob.root.connect('sub1.src.x2', 'sub2.tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['sub1.src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtK.x3'], 373.15, 1e-6)

        # Make sure we don't convert equal units
        self.assertEqual(prob.root.sub2.params.metadata('tgtC.x2').get('unit_conv'),
                         None)

        indep_list = ['x1']
        unknown_list = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        stream = cStringIO()
        conv = prob.root.sub1.list_unit_conv(stream=stream)
        self.assertTrue(len(conv) == 0)


    def test_list_unit_connections_sub(self):

        prob = Problem()
        prob.root = Group()
        sub1 = prob.root.add('sub1', Group())
        sub2 = prob.root.add('sub2', Group())
        sub1.add('src', SrcComp())
        sub1.add('tgtF', TgtCompF())
        sub2.add('tgtC', TgtCompC())
        sub2.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'sub1.src.x1')
        prob.root.connect('sub1.src.x2', 'sub1.tgtF.x2')
        prob.root.connect('sub1.src.x2', 'sub2.tgtC.x2')
        prob.root.connect('sub1.src.x2', 'sub2.tgtK.x2')

        prob.setup(check=False)
        prob.run()

        stream = cStringIO()
        conv = prob.root.sub1.list_unit_conv(stream=stream)
        self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)

    def test_basic_grouped_bug_from_pycycle(self):

        prob = Problem()
        root = prob.root = Group()
        sub1 = prob.root.add('sub1', Group(), promotes=['x2'])
        sub1.add('src', SrcComp(), promotes = ['x2'])
        root.add('tgtF', TgtCompFMulti())
        root.add('tgtC', TgtCompC())
        root.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'sub1.src.x1')
        prob.root.connect('x2', 'tgtF.x2')
        prob.root.connect('x2', 'tgtC.x2')
        prob.root.connect('x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_grouped_grouped_implicit(self):

        prob = Problem()
        root = prob.root = Group()
        sub1 = prob.root.add('sub1', Group(), promotes=['x2'])
        sub2 = prob.root.add('sub2', Group(), promotes=['x2'])
        sub1.add('src', SrcComp(), promotes = ['x2'])
        sub2.add('tgtF', TgtCompFMulti(), promotes=['x2'])
        sub2.add('tgtC', TgtCompC(), promotes=['x2'])
        sub2.add('tgtK', TgtCompK(), promotes=['x2'])
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'sub1.src.x1')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtK.x3'], 373.15, 1e-6)

        indep_list = ['x1']
        unknown_list = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_apply_linear_adjoint(self):
        # Make sure we can index into dparams

        class Attitude_Angular(Component):
            """ Calculates angular velocity vector from the satellite's orientation
            matrix and its derivative.
            """

            def __init__(self, n=2):
                super(Attitude_Angular, self).__init__()

                self.n = n

                # Inputs
                self.add_param('O_BI', np.zeros((3, 3, n)), units="ft",
                               desc="Rotation matrix from body-fixed frame to Earth-centered "
                               "inertial frame over time")

                self.add_param('Odot_BI', np.zeros((3, 3, n)), units="km",
                               desc="First derivative of O_BI over time")

                # Outputs
                self.add_output('w_B', np.zeros((3, n)), units="1/s",
                                desc="Angular velocity vector in body-fixed frame over time")

                self.dw_dOdot = np.zeros((n, 3, 3, 3))
                self.dw_dO = np.zeros((n, 3, 3, 3))

            def solve_nonlinear(self, params, unknowns, resids):
                """ Calculate output. """

                O_BI = params['O_BI']
                Odot_BI = params['Odot_BI']
                w_B = unknowns['w_B']

                for i in range(0, self.n):
                    w_B[0, i] = np.dot(Odot_BI[2, :, i], O_BI[1, :, i])
                    w_B[1, i] = np.dot(Odot_BI[0, :, i], O_BI[2, :, i])
                    w_B[2, i] = np.dot(Odot_BI[1, :, i], O_BI[0, :, i])

            def linearize(self, params, unknowns, resids):
                """ Calculate and save derivatives. (i.e., Jacobian) """

                O_BI = params['O_BI']
                Odot_BI = params['Odot_BI']

                for i in range(0, self.n):
                    self.dw_dOdot[i, 0, 2, :] = O_BI[1, :, i]
                    self.dw_dO[i, 0, 1, :] = Odot_BI[2, :, i]

                    self.dw_dOdot[i, 1, 0, :] = O_BI[2, :, i]
                    self.dw_dO[i, 1, 2, :] = Odot_BI[0, :, i]

                    self.dw_dOdot[i, 2, 1, :] = O_BI[0, :, i]
                    self.dw_dO[i, 2, 0, :] = Odot_BI[1, :, i]

            def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
                """ Matrix-vector product with the Jacobian. """

                dw_B = dresids['w_B']

                if mode == 'fwd':
                    for k in range(3):
                        for i in range(3):
                            for j in range(3):
                                if 'O_BI' in dparams:
                                    dw_B[k, :] += self.dw_dO[:, k, i, j] * \
                                        dparams['O_BI'][i, j, :]
                                if 'Odot_BI' in dparams:
                                    dw_B[k, :] += self.dw_dOdot[:, k, i, j] * \
                                        dparams['Odot_BI'][i, j, :]

                else:

                    for k in range(3):
                        for i in range(3):
                            for j in range(3):

                                if 'O_BI' in dparams:
                                    dparams['O_BI'][i, j, :] += self.dw_dO[:, k, i, j] * \
                                        dw_B[k, :]

                                if 'Odot_BI' in dparams:
                                    dparams['Odot_BI'][i, j, :] -= -self.dw_dOdot[:, k, i, j] * \
                                        dw_B[k, :]

        prob = Problem()
        root = prob.root = Group()
        prob.root.add('comp', Attitude_Angular(n=5), promotes=['*'])
        prob.root.add('p1', IndepVarComp('O_BI', np.ones((3, 3, 5))), promotes=['*'])
        prob.root.add('p2', IndepVarComp('Odot_BI', np.ones((3, 3, 5))), promotes=['*'])

        prob.setup(check=False)
        prob.run()

        indep_list = ['O_BI', 'Odot_BI']
        unknown_list = ['w_B']
        Jf = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                                return_format='dict')

        indep_list = ['O_BI', 'Odot_BI']
        unknown_list = ['w_B']
        Jr = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                                return_format='dict')

        for key, val in iteritems(Jr):
            for key2 in val:
                diff = abs(Jf[key][key2] - Jr[key][key2])
                assert_rel_error(self, diff, 0.0, 1e-10)

    def test_incompatible_connections(self):

        class BadComp(Component):
            def __init__(self):
                super(BadComp, self).__init__()

                self.add_param('x2', 100.0, units='m')
                self.add_output('x3', 100.0)

        # Explicit Connection
        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('dest', BadComp())
        prob.root.connect('src.x2', 'dest.x2')
        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected_msg = "Unit 'degC' in source 'src.x2' is incompatible with unit 'm' in target 'dest.x2'."

        self.assertTrue(expected_msg in str(cm.exception))

        # Implicit Connection
        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp(), promotes=['x2'])
        prob.root.add('dest', BadComp(),promotes=['x2'])
        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected_msg = "Unit 'degC' in source 'src.x2' (x2) is incompatible with unit 'm' in target 'dest.x2' (x2)."

        self.assertTrue(expected_msg in str(cm.exception))


class PBOSrcComp(Component):

    def __init__(self):
        super(PBOSrcComp, self).__init__()

        self.add_param('x1', 100.0)
        self.add_output('x2', 100.0, units='degC', pass_by_obj=True)
        self.fd_options['force_fd'] = True

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x2'] = params['x1']


class PBOTgtCompF(Component):

    def __init__(self):
        super(PBOTgtCompF, self).__init__()

        self.add_param('x2', 100.0, units='degF', pass_by_obj=True)
        self.add_output('x3', 100.0)
        self.fd_options['force_fd'] = True

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']


class TestUnitConversionPBO(unittest.TestCase):
    """ Tests support for unit conversions on pass_by_obj connections."""

    def test_basic(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', PBOSrcComp())
        prob.root.add('tgtF', PBOTgtCompF())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)

        indep_list = ['x1']
        unknown_list = ['tgtF.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)

        stream = cStringIO()
        conv = prob.root.list_unit_conv(stream=stream)
        self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)


if __name__ == "__main__":
    unittest.main()
