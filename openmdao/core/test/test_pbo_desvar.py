
import unittest
from openmdao.api import Component, Problem, Group, IndepVarComp, ExecComp, \
                         Driver, ScipyOptimizer
from openmdao.util.record_util import create_local_meta, update_local_meta

try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    pyOptSparseDriver = None

class GradFreeDriver(Driver):
    def __init__(self, *args, **kwargs):
        super(GradFreeDriver, self).__init__(*args, **kwargs)

    def _setup(self):
        self.supports['gradients'] = False
        super(GradFreeDriver, self)._setup()

    def run(self, problem):
        self.set_desvar('p1.x', 'var_x')
        self.set_desvar('p2.y', 123.0)
        metadata = create_local_meta(None, 'Driver')
        problem.root.solve_nonlinear(metadata=metadata)

class GradDriver(Driver):
    def __init__(self, *args, **kwargs):
        super(GradDriver, self).__init__(*args, **kwargs)

    def run(self, problem):
        self.set_desvar('p1.x', 'var_x')
        self.set_desvar('p2.y', 123.0)
        metadata = create_local_meta(None, 'Driver')
        problem.root.solve_nonlinear(metadata=metadata)


class PassByObjParaboloid(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

    def __init__(self):
        super(PassByObjParaboloid, self).__init__()

        self.add_param('x', val=u'var_x', pass_by_obj=True)
        self.add_param('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """

        x = hash(params['x'])
        y = params['y']

        unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def linearize(self, params, unknowns, resids):
        """ Jacobian for our paraboloid."""

        x = hash(params['x'])
        y = params['y']
        J = {}

        J['f_xy', 'x'] = 2.0*x - 6.0 + y
        J['f_xy', 'y'] = 2.0*y + 8.0 + x
        return J

class PBOobjective(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

    def __init__(self):
        super(PBOobjective, self).__init__()

        self.add_param('x', val=0.0)
        self.add_output('f_x', val=0.0, pass_by_obj=True)


class TestPBODesvar(unittest.TestCase):
    """Test for adding pass_by_obj variables to a gradient free
    driver.
    """

    def test_pbo_desvar_grad_free(self):
        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', u'var_x', pass_by_obj=True))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', PassByObjParaboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = GradFreeDriver()

        top.driver.add_desvar('p1.x')
        top.driver.add_desvar('p2.y')
        top.driver.add_objective('p.f_xy')

        top.setup(check=False)
        top.run()

    def test_pbo_desvar_grad(self):
        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', u'var_x', pass_by_obj=True))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', PassByObjParaboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = GradDriver()

        top.driver.add_desvar('p1.x')
        top.driver.add_desvar('p2.y')
        top.driver.add_objective('p.f_xy')

        try:
            top.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err), "Parameter 'p1.x' is a 'pass_by_obj' variable and "
                             "can't be used with a gradient based driver of type "
                             "'GradDriver'.")
        else:
            self.fail("Exception expected")

    def test_pbo_obj_grad(self):
        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', -4.0))
        root.add('p', PBOobjective())

        root.connect('p1.x', 'p.x')

        top.driver = GradDriver()

        top.driver.add_desvar('p1.x')
        top.driver.add_objective('p.f_x')

        try:
            top.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err), "Objective 'p.f_x' is a 'pass_by_obj' variable and "
                             "can't be used with a gradient based driver of type "
                             "'GradDriver'.")
        else:
            self.fail("Exception expected")

    def test_pbo_constraint_grad(self):
        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', -4.0))
        root.add('p', PBOobjective())

        root.connect('p1.x', 'p.x')

        top.driver = GradDriver()

        top.driver.add_desvar('p1.x')
        top.driver.add_constraint('p.f_x', lower=0.0)

        try:
            top.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err), "Constraint 'p.f_x' is a 'pass_by_obj' variable and "
                             "can't be used with a gradient based driver of type "
                             "'GradDriver'.")
        else:
            self.fail("Exception expected")

    def test_pbo_desvar_nsga2(self):
        if pyOptSparseDriver is None:
            raise unittest.SkipTest("pyOptSparse not installed")

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', u'var_x', pass_by_obj=True))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', PassByObjParaboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = pyOptSparseDriver()
        top.driver.options['optimizer'] = 'NSGA2'

        top.driver.add_desvar('p1.x')
        top.driver.add_desvar('p2.y')
        top.driver.add_objective('p.f_xy')

        top.setup(check=False)

    def test_pbo_desvar_slsqp(self):
        if pyOptSparseDriver is None:
            raise unittest.SkipTest("pyOptSparse not installed")

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', u'var_x', pass_by_obj=True))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', PassByObjParaboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = pyOptSparseDriver()
        top.driver.options['optimizer'] = 'SLSQP'

        top.driver.add_desvar('p1.x')
        top.driver.add_desvar('p2.y')
        top.driver.add_objective('p.f_xy')

        try:
            top.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err), "Parameter 'p1.x' is a 'pass_by_obj' variable and "
                             "can't be used with a gradient based driver of type 'SLSQP'.")
        else:
            self.fail("Exception expected")

    def test_pbo_desvar_slsqp_scipy(self):
        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', u'var_x', pass_by_obj=True))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', PassByObjParaboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'

        top.driver.add_desvar('p1.x')
        top.driver.add_desvar('p2.y')
        top.driver.add_objective('p.f_xy')

        try:
            top.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err), "Parameter 'p1.x' is a 'pass_by_obj' variable and "
                             "can't be used with a gradient based driver of type 'SLSQP'.")
        else:
            self.fail("Exception expected")

    def test_pbo_desvar_scipy_grad_free(self):
        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', u'var_x', pass_by_obj=True))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', PassByObjParaboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'Nelder-Mead'

        top.driver.add_desvar('p1.x')
        top.driver.add_desvar('p2.y')
        top.driver.add_objective('p.f_xy')

        top.setup(check=False)

if __name__ == "__main__":
    unittest.main()
