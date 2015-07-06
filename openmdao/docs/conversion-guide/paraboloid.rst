Paraboloid Example
------------------

Consider the case of a simple Component class called Paraboloid.  First we'll look
at the class written in OpenMDAO Classic:

::

    """ Paraboloid class written in OpenMDAO Classic """
    from openmdao.main.api import Component
    from openmdao.lib.datatypes.api import Float


    class Paraboloid(Component):
        """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

        # set up interface to the framework
        x = Float(0.0, iotype='in', desc='The variable x')
        y = Float(0.0, iotype='in', desc='The variable y')

        f_xy = Float(0.0, iotype='out', desc='F(x,y)')


        def execute(self):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
                Minimum: x = 6.6667; y = -7.3333
            """

            x = self.x
            y = self.y

            self.f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0


...Then consider the new arrangement, in OpenMDAO 1.0:

::

    """ Paraboloid class written in OpenMDAO 1.0 """
    """ paraboloid.py - Evaluates the equation (x-3)^2 + xy + (y+4)^2 = 3
    """
    from openmdao.core.component import Component


    class Paraboloid(Component):
        """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

        def __init__(self):
            super(Paraboloid, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', val=0.0)

        def solve_nonlinear(self, params, unknowns, resids):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
            Optimal solution (minimum): x = 6.6667; y = -7.3333
            """

            x = params['x']
            y = params['y']

            unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

        def jacobian(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy','x'] = 2.0*x - 6.0 + y
            J['f_xy','y'] = 2.0*y + 8.0 + x
            return J


[discussion of how the two differ, perhaps doing a functional line-by-line comparison.]
