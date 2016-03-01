""" Test the examples directory to keep them in working order.

   NOTE: If you make any changes to this file, you must make the corresponding
   change to the example file.
"""

import unittest
from six.moves import cStringIO

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizer, \
     Newton, ScipyGMRES
from openmdao.test.util import assert_rel_error

from beam_tutorial import BeamTutorial
from fd_comp_example import Model as Model1
from fd_group_example import Model as Model2
from fd_model_example import Model as Model3
from implicit import SimpleImplicitComp
from implicit_ext_solve import SimpleImplicitComp as SIC2
from intersect_parabola_line import Balance, Parabola, Line
from krig_sin import TrigMM
from paraboloid_example import Paraboloid
from paraboloid_optimize_constrained import Paraboloid as ParaboloidOptCon
from paraboloid_optimize_unconstrained import Paraboloid as ParaboloidOptUnCon
from sellar_MDF_optimize import SellarDerivatives
from sellar_state_MDF_optimize import SellarStateConnection


class TestExamples(unittest.TestCase):

    def test_paraboloid_example(self):

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.setup(check=False)
        top.run()

        assert_rel_error(self, root.p.unknowns['f_xy'], -15.0, 1e-6)

    def test_paraboloid_optimize_constrained(self):

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', ParaboloidOptCon())

        # Constraint Equation
        root.add('con', ExecComp('c = x-y'))

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')
        root.connect('p.x', 'con.x')
        root.connect('p.y', 'con.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['disp'] = False

        top.driver.add_desvar('p1.x', lower=-50, upper=50)
        top.driver.add_desvar('p2.y', lower=-50, upper=50)
        top.driver.add_objective('p.f_xy')
        top.driver.add_constraint('con.c', lower=15.0)

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['p.x'], 7.166667, 1e-6)
        assert_rel_error(self, top['p.y'], -7.833333, 1e-6)

    def test_paraboloid_optimize_unconstrained(self):

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', ParaboloidOptUnCon())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['disp'] = False

        top.driver.add_desvar('p1.x', lower=-50, upper=50)
        top.driver.add_desvar('p2.y', lower=-50, upper=50)
        top.driver.add_objective('p.f_xy')

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['p.x'], 6.666667, 1e-6)
        assert_rel_error(self, top['p.y'], -7.333333, 1e-6)

    def test_beam_tutorial(self):

        top = Problem()
        top.root = BeamTutorial()

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['tol'] = 1.0e-8
        top.driver.options['maxiter'] = 10000 #maximum number of solver iterations
        top.driver.options['disp'] = False

        #room length and width bounds
        top.driver.add_desvar('ivc_rlength.room_length', lower=5.0*12.0, upper=50.0*12.0) #domain: 1in <= length <= 50ft
        top.driver.add_desvar('ivc_rwidth.room_width', lower=5.0*12.0, upper=30.0*12.0) #domain: 1in <= width <= 30ft

        top.driver.add_objective('d_neg_area.neg_room_area') #minimize negative area (or maximize area)

        top.driver.add_constraint('d_len_minus_wid.length_minus_width', lower=0.0) #room_length >= room_width
        top.driver.add_constraint('d_deflection.deflection', lower=720.0) #deflection >= 720
        top.driver.add_constraint('d_bending.bending_stress_ratio', upper=0.5) #bending < 0.5
        top.driver.add_constraint('d_shear.shear_stress_ratio', upper=1.0/3.0) #shear < 1/3


        top.setup(check=False)
        top.run()

        assert_rel_error(self, -top['d_neg_area.neg_room_area'], 51655.257618, .01)
        assert_rel_error(self, top['ivc_rwidth.room_width'], 227.277956, .01)
        assert_rel_error(self,top['ivc_rlength.room_length'], 227.277904, .01)
        assert_rel_error(self,top['d_deflection.deflection'], 720, .01)
        assert_rel_error(self,top['d_bending.bending_stress_ratio'], 0.148863, .001)
        assert_rel_error(self,top['d_shear.shear_stress_ratio'], 0.007985, .0001)

    def test_line_parabola_intersect(self):

        from intersect_parabola_line import Line, Parabola, Balance

        top = Problem()
        root = top.root = Group()
        root.add('line', Line())
        root.add('parabola', Parabola())
        root.add('bal', Balance())

        root.connect('line.y', 'bal.y1')
        root.connect('parabola.y', 'bal.y2')
        root.connect('bal.x', 'line.x')
        root.connect('bal.x', 'parabola.x')

        root.nl_solver = Newton()
        root.ln_solver = ScipyGMRES()

        top.setup(check=False)

        stream = cStringIO()

        # Positive solution
        top['bal.x'] = 7.0
        root.list_states(stream)
        top.run()
        assert_rel_error(self, top['bal.x'], 1.430501, 1e-5)
        assert_rel_error(self, top['line.y'], 1.138998, 1e-5)

        # Negative solution
        top['bal.x'] = -7.0
        root.list_states(stream)
        top.run()
        assert_rel_error(self, top['bal.x'], -2.097168, 1e-5)
        assert_rel_error(self, top['line.y'], 8.194335, 1e-5)

    def test_sellar_MDF_optimize(self):

        top = Problem()
        top.root = SellarDerivatives()

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['tol'] = 1.0e-8
        top.driver.options['disp'] = False

        top.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]))
        top.driver.add_desvar('x', lower=0.0, upper=10.0)

        top.driver.add_objective('obj')
        top.driver.add_constraint('con1', upper=0.0)
        top.driver.add_constraint('con2', upper=0.0)

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['z'][0], 1.977639, 1e-5)
        assert_rel_error(self, top['z'][1], 0.0, 1e-5)
        assert_rel_error(self, top['x'], 0.0, 1e-5)
        assert_rel_error(self, top['obj'], 3.1833940, 1e-5)

    def test_sellar_state_connection(self):

        top = Problem()
        top.root = SellarStateConnection()

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['tol'] = 1.0e-8
        top.driver.options['disp'] = False

        top.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]))
        top.driver.add_desvar('x', lower=0.0, upper=10.0)

        top.driver.add_objective('obj')
        top.driver.add_constraint('con1', upper=0.0)
        top.driver.add_constraint('con2', upper=0.0)

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['z'][0], 1.977639, 1e-5)
        assert_rel_error(self, top['z'][1], 0.0, 1e-5)
        assert_rel_error(self, top['x'], 0.0, 1e-5)
        assert_rel_error(self, top['obj'], 3.1833940, 1e-5)

    def test_intersect_parabola_line(self):

        top = Problem()
        root = top.root = Group()
        root.add('line', Line())
        root.add('parabola', Parabola())
        root.add('bal', Balance())

        root.connect('line.y', 'bal.y1')
        root.connect('parabola.y', 'bal.y2')
        root.connect('bal.x', 'line.x')
        root.connect('bal.x', 'parabola.x')

        root.nl_solver = Newton()
        root.ln_solver = ScipyGMRES()

        top.setup(check=False)

        # Positive solution
        top['bal.x'] = 7.0
        top.run()
        assert_rel_error(self, top['bal.x'], 1.430501, 1e-5)
        assert_rel_error(self, top['line.y'], 1.1389998, 1e-5)

        # Negative solution
        top['bal.x'] = -7.0
        top.run()
        assert_rel_error(self, top['bal.x'], -2.097168, 1e-5)
        assert_rel_error(self, top['line.y'], 8.194335, 1e-5)

    def test_implicit(self):
        top = Problem()
        root = top.root = Group()
        root.add('comp', SimpleImplicitComp())

        root.ln_solver = ScipyGMRES()
        top.setup(check=False)

        top.run()
        assert_rel_error(self, top['comp.z'], 2.666667, 1e-5)

    def test_implicit_ext_solve(self):
        top = Problem()
        root = top.root = Group()
        root.add('p1', IndepVarComp('x', 0.5))
        root.add('comp', SimpleImplicitComp())
        root.add('comp2', ExecComp('zz = 2.0*z'))

        root.connect('p1.x', 'comp.x')
        root.connect('comp.z', 'comp2.z')

        root.ln_solver = ScipyGMRES()
        root.nl_solver = Newton()
        top.setup(check=False)

        top.run()
        assert_rel_error(self, top['comp.z'], 2.666667, 1e-5)

    def test_fd_comp_example(self):

        top = Problem()
        top.root = Model1()

        top.setup(check=False)
        top.root.comp1.print_output = False
        top.root.comp2.print_output = False
        top.root.comp3.print_output = False
        top.root.comp4.print_output = False
        top.run()

        J = top.calc_gradient(['px.x'], ['comp4.y'])
        assert_rel_error(self, J[0][0], 81.0, 1e-5)

    def test_fd_group_example(self):

        top = Problem()
        top.root = Model2()

        top.setup(check=False)
        top.root.comp1.print_output = False
        top.root.sub.comp2.print_output = False
        top.root.sub.comp3.print_output = False
        top.root.comp4.print_output = False
        top.run()

        J = top.calc_gradient(['px.x'], ['comp4.y'])
        assert_rel_error(self, J[0][0], 81.0, 1e-5)

    def test_fd_model_example(self):

        top = Problem()
        top.root = Model3()

        top.setup(check=False)
        top.root.comp1.print_output = False
        top.root.comp2.print_output = False
        top.root.comp3.print_output = False
        top.root.comp4.print_output = False
        top.run()

        J = top.calc_gradient(['px.x'], ['comp4.y'])
        assert_rel_error(self, J[0][0], 81.0, 1e-5)

    def test_krig_sin(self):

        prob = Problem()
        prob.root = TrigMM()
        prob.setup(check=False)

        #traning data is just set manually. No connected input needed, since
        #  we're assuming the data is pre-existing
        prob['sin_mm.train:x'] = np.linspace(0,10,20)
        prob['sin_mm.train:f_x:float'] = np.sin(prob['sin_mm.train:x'])
        prob['sin_mm.train:f_x:norm_dist'] = np.cos(prob['sin_mm.train:x'])

        prob['sin_mm.x'] = 2.1 #prediction happens at this value
        prob.run()

        assert_rel_error(self, prob['sin_mm.f_x:float'], 0.8632, 1e-3)
        assert_rel_error(self, prob['sin_mm.f_x:norm_dist'][0], -0.5048, 1e-3)

if __name__ == "__main__":
    unittest.main()
