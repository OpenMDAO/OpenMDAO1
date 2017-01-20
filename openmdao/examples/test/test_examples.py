""" Test the examples directory to keep them in working order.

   NOTE: If you make any changes to this file, you must make the corresponding
   change to the example file.
"""

import inspect
import sys
import math
import unittest
from six.moves import cStringIO

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizer, \
     Newton, ScipyGMRES, inf_bound
from openmdao.test.util import assert_rel_error, set_pyoptsparse_opt

from openmdao.examples.beam_tutorial import BeamTutorial
from openmdao.examples.fd_comp_example import Model as Model1
from openmdao.examples.fd_group_example import Model as Model2
from openmdao.examples.fd_model_example import Model as Model3
from openmdao.examples.implicit import SimpleImplicitComp
from openmdao.examples.implicit_ext_solve import SimpleImplicitComp as SIC2
from openmdao.examples.intersect_parabola_line import Balance, Parabola, Line
from openmdao.examples.krig_sin import TrigMM
from openmdao.examples.paraboloid_example import Paraboloid
from openmdao.examples.paraboloid_optimize_constrained import Paraboloid as ParaboloidOptCon
from openmdao.examples.paraboloid_optimize_unconstrained import Paraboloid as ParaboloidOptUnCon
from openmdao.examples.sellar_MDF_optimize import SellarDerivatives
from openmdao.examples.sellar_state_MDF_optimize import SellarStateConnection
from openmdao.examples.sellar_sand_architecture import SellarSAND
from openmdao.examples.subproblem_example import main as subprob_main
from openmdao.examples.cylinder_opt_example import opt_cylinder1, opt_cylinder2
from openmdao.examples.hohmann_transfer import VCircComp, DeltaVComp, TransferOrbitComp

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

    def test_paraboloid_optimize_constrained_explicit_infinite_bounds(self):

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

        top.driver.add_desvar('p1.x', lower=-inf_bound, upper=50)
        top.driver.add_desvar('p2.y', lower=-50, upper=inf_bound)
        top.driver.add_objective('p.f_xy')
        top.driver.add_constraint('con.c', lower=15.0, upper=inf_bound)

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

    def test_sellar_sand_architecture(self):

        top = Problem()
        top.root = SellarSAND()

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['tol'] = 1.0e-12
        top.driver.options['disp'] = False

        top.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                                   upper=np.array([10.0, 10.0]))
        top.driver.add_desvar('x', lower=0.0, upper=10.0)
        top.driver.add_desvar('y1', lower=-10.0, upper=10.0)
        top.driver.add_desvar('y2', lower=-10.0, upper=10.0)

        top.driver.add_objective('obj')
        top.driver.add_constraint('con1', upper=0.0)
        top.driver.add_constraint('con2', upper=0.0)
        top.driver.add_constraint('resid1', equals=0.0)
        top.driver.add_constraint('resid2', equals=0.0)

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, top['z'][1], 0.0000, 1e-3)
        assert_rel_error(self, top['x'], 0.0000, 1e-3)
        assert_rel_error(self, top['d1.y1'], 3.1600, 1e-3)
        assert_rel_error(self, top['d1.y2'], 3.7553, 1e-3)
        assert_rel_error(self, top['obj'], 3.1834, 1e-3)

        # Minimum found at (z1,z2,x) = (1.9776, 0.0000, 0.0000)
        # Coupling vars: 3.1600, 3.7553
        # Minimum objective: 3.1834

    def test_subproblem(self):
        if sys.platform == 'win32':
            # avoid a weird nested multiprocessing pickling issue using py3 on windows
            num_par_doe = 1
        else:
            num_par_doe = 2

        global_opt = subprob_main(num_par_doe)
        assert_rel_error(self, global_opt['subprob.comp.fx'], -1.-math.pi/10., 1e-5)
        assert_rel_error(self, global_opt['subprob.indep.x'], math.pi, 1e-5)

    def test_opt_cylinder(self):
        expected = {
            'indep.r': 6.2035,
            'indep.h': 12.407,
            'cylinder.area': 725.396379,
            'cylinder.volume': 1.5
        }

        for name, val in opt_cylinder1():
            assert_rel_error(self, expected[name], val, 1e-5)

        for name, val in opt_cylinder2():
            assert_rel_error(self, expected[name], val, 1e-5)

    def test_discs(self):

        OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        # So we compare the same starting locations.
        np.random.seed(123)

        radius = 1.0
        pin = 15.0
        n_disc = 7

        prob = Problem()
        prob.root = root = Group()

        from openmdao.api import pyOptSparseDriver
        driver = prob.driver = pyOptSparseDriver()
        driver.options['optimizer'] = 'SNOPT'
        driver.options['print_results'] = False

        # Note, active tolerance requires relevance reduction to work.
        root.ln_solver.options['single_voi_relevance_reduction'] = True

        # Also, need to be in adjoint
        root.ln_solver.options['mode'] = 'rev'

        obj_expr = 'obj = '
        sep = ''
        for i in range(n_disc):

            dist = "dist_%d" % i
            x1var = 'x_%d' % i

            # First disc is pinned
            if i == 0:
                root.add('p_%d' % i, IndepVarComp(x1var, pin), promotes=(x1var, ))

            # The rest are design variables for the optimizer.
            else:
                init_val = 5.0*np.random.random() - 5.0 + pin
                root.add('p_%d' % i, IndepVarComp(x1var, init_val), promotes=(x1var, ))
                driver.add_desvar(x1var)

            for j in range(i):

                x2var = 'x_%d' % j
                yvar = 'y_%d_%d' % (i, j)
                name = dist + "_%d" % j
                expr = '%s= (%s - %s)**2' % (yvar, x1var, x2var)
                root.add(name, ExecComp(expr), promotes = (x1var, x2var, yvar))

                # Constraint (you can experiment with turning on/off the active_tol)
                #driver.add_constraint(yvar, lower=radius)
                driver.add_constraint(yvar, lower=radius, active_tol=radius*3.0)

                # This pair's contribution to objective
                obj_expr += sep + yvar
                sep = ' + '

        root.add('sum_dist', ExecComp(obj_expr), promotes=('*', ))
        driver.add_objective('obj')

        prob.setup(check=False)
        prob.run()

        # Just making sure there are no syntax errors.

    def test_hohmann_result(self):
        prob = Problem(root=Group())

        root = prob.root

        root.add('mu_comp', IndepVarComp('mu', val=0.0, units='km**3/s**2'),
                 promotes=['mu'])

        root.add('r1_comp', IndepVarComp('r1', val=0.0, units='km'),
                 promotes=['r1'])
        root.add('r2_comp', IndepVarComp('r2', val=0.0, units='km'),
                 promotes=['r2'])

        root.add('dinc1_comp', IndepVarComp('dinc1', val=0.0, units='deg'),
                 promotes=['dinc1'])
        root.add('dinc2_comp', IndepVarComp('dinc2', val=0.0, units='deg'),
                 promotes=['dinc2'])

        root.add('leo', system=VCircComp())
        root.add('geo', system=VCircComp())

        root.add('transfer', system=TransferOrbitComp())

        root.connect('r1', ['leo.r', 'transfer.rp'])
        root.connect('r2', ['geo.r', 'transfer.ra'])

        root.connect('mu', ['leo.mu', 'geo.mu', 'transfer.mu'])

        root.add('dv1', system=DeltaVComp())

        root.connect('leo.vcirc', 'dv1.v1')
        root.connect('transfer.vp', 'dv1.v2')
        root.connect('dinc1', 'dv1.dinc')

        root.add('dv2', system=DeltaVComp())

        root.connect('transfer.va', 'dv2.v1')
        root.connect('geo.vcirc', 'dv2.v2')
        root.connect('dinc2', 'dv2.dinc')

        root.add('dv_total', system=ExecComp('delta_v=dv1+dv2',
                                             units={'delta_v': 'km/s',
                                                    'dv1': 'km/s',
                                                    'dv2': 'km/s'}),
                 promotes=['delta_v'])

        root.connect('dv1.delta_v', 'dv_total.dv1')
        root.connect('dv2.delta_v', 'dv_total.dv2')

        root.add('dinc_total', system=ExecComp('dinc=dinc1+dinc2',
                                               units={'dinc': 'deg',
                                                      'dinc1': 'deg',
                                                      'dinc2': 'deg'}),
                 promotes=['dinc'])

        root.connect('dinc1', 'dinc_total.dinc1')
        root.connect('dinc2', 'dinc_total.dinc2')

        prob.driver = ScipyOptimizer()

        prob.driver.add_desvar('dinc1', lower=0, upper=28.5)
        prob.driver.add_desvar('dinc2', lower=0, upper=28.5)
        prob.driver.add_constraint('dinc', lower=28.5, upper=28.5, scaler=1.0)
        prob.driver.add_objective('delta_v', scaler=1.0)

        # Setup the problem

        prob.setup()

        # Set initial values

        prob['mu'] = 398600.4418
        prob['r1'] = 6778.137
        prob['r2'] = 42164.0

        prob['dinc1'] = 0.0
        prob['dinc2'] = 0.0

        # Go!

        prob.run()

        self.assertAlmostEqual(prob['delta_v'], 4.19629634132, places=4)
        self.assertAlmostEqual(prob['dinc1'], 2.2348615725962211, places=4)
        self.assertAlmostEqual(prob['dinc2'], 26.26513842740378, places=4)

if __name__ == "__main__":
    unittest.main()
