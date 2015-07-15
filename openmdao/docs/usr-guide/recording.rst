.. _OpenMDAO-Examples:

=========
Recording
=========

In a previous example, we looked at the Paraboloid component. This tutorial
builds on this example by adding optimization and demonstrating how to
save the data generated for future use. Consider the code below:

::

    from openmdao.components.paramcomp import ParamComp
    from openmdao.core.component import Component
    from openmdao.core.group import Group
    from openmdao.core.problem import Problem
    from openmdao.drivers.scipy_optimizer import ScipyOptimizer
    from openmdao.recorders.shelverecorder import ShelveRecorder


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

            unknowns['f_xy'] = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

        def jacobian(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy', 'x'] = 2.0 * x - 6.0 + y
            J['f_xy', 'y'] = 2.0 * y + 8.0 + x
            return J


    if __name__ == '__main__':
        top = Problem()

        root = top.root = Group()

        paraboloid = root.add('p', Paraboloid(), promotes=['*'])

        root.add('p1', ParamComp('x', 3.0), promotes=['*'])
        root.add('p2', ParamComp('y', -4.0), promotes=['*'])

        top.driver = driver = ScipyOptimizer()

        driver.add_param('x')
        driver.add_param('y')
        driver.add_objective('f_xy')

        recorder = ShelveRecorder('paraboloid')
        driver.add_recorder(recorder)

        top.setup()
        top.run()

This script is very similar to the code in the Paraboloid tutorial, with a few important differences.

::

    top.driver = driver = ScipyOptimizer()

    driver.add_param('x')
    driver.add_param('y')
    driver.add_objective('f_xy')

We add an optimizer to the problem and initialize it.

::

    recorder = ShelveRecorder('paraboloid')
    driver.add_recorder(recorder)

These two lines are all it takes to record the state of the problem as the
optimizer progresses. We initialize a `ShelveRecorder` by passing it a
`filename` argument. This recorder uses Python's `shelve` module to store the
data generated. In this case, `shelve` will open a file named 'paraboloid'
to use as a backend. Note that depending on your operating system and version
of Python, the actual file generated may have a different name (e.g.
paraboloid.db), but `shelve` will be able to open the correct file.

We then attach the recorder to the driver using `driver.add_recorder`.
Depending on your needs, you