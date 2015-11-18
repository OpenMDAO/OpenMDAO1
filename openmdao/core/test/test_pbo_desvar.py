
import unittest
from openmdao.api import Component, Problem, Group, IndepVarComp, ExecComp, Driver
from openmdao.core.checks import ConnectError
from openmdao.util.record_util import create_local_meta, update_local_meta


class DummyDriver(Driver):
    def __init__(self, *args, **kwargs):
        super(DummyDriver, self).__init__(*args, **kwargs)

    def run(self, problem):
        self.set_desvar('p1.x', u'var_x')
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

class TestPBODesvar(unittest.TestCase):
    """Test for adding pass_by_obj variables to a gradient free
    driver.
    """

    def test_pbo_desvar(self):
        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', u'var_x', pass_by_obj=True))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', PassByObjParaboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = DummyDriver()

        top.driver.add_desvar('p1.x')
        top.driver.add_desvar('p2.y')
        top.driver.add_objective('p.f_xy')

        top.setup()
        top.run()
        # print(root.p.unknowns['f_xy'])


if __name__ == "__main__":
    unittest.main()
