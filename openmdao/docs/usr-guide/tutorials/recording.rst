.. _OpenMDAO-Recording:

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

        root.add('p1', ParamComp('x', 3.0), promotes=['*'])
        root.add('p2', ParamComp('y', -4.0), promotes=['*'])
        paraboloid = root.add('p', Paraboloid(), promotes=['*'])

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
Depending on your needs, you are able to attach more recorders by using
additional `driver.add_recorder` calls. Solver also have an `add_recorder`
method that is invoked the same way. This allows you to record the evolution
of variables at lower levels.


Includes and Excludes
=====================

Over the course of an analysis or optimization, the model may generate a very
large amount of data. Since you may not be interested in the value of every
variable at every step, OpenMDAO allows you to filter what variables are
recorded through the use of includes and excludes. The recorder will store
anything that matches the includes filter and that does not match the exclude
filter. By default, the includes are set to `['*']` and the excludes are set to
`[]`, i.e. include everything and exclude nothing.

The includes and excludes filters are set via the `options` structure in the
recorder. If we were only interested in the variable `x` from our Paraboloid
model, we could record that by setting the includes as follows:

::

    recorder = ShelveRecorder('paraboloid')
    recorder.options['includes'] = ['x']

    driver.add_recorder(recorder)

Similarly, if we were interested in everything except the value of `f_xy`, we
could exclude that by doing the following:
::

    recorder = ShelveRecorder('paraboloid')
    recorder.options['excludes'] = ['f_xy']

    driver.add_recorder(recorder)

The includes and excludes filters will accept glob arguments. For example,
`recorder.options['excludes'] = ['comp1.*']` would exclude any variable
that starts with "comp1.".

Accessing Recorded Data
=======================

While each recorder stores data slightly differently in order to match the
file format, the common theme for accessing data is the iteration coordinate.
The iteration coordinate describes where and when in the execution hierarchy
the data was collected. Iteration coordinates are strings formatted as pairs
of names and iteration numbers separated by '/'. For example,
'SLSQP/1/root/2/G1/3' would describe the third iteration of 'G1' during the
second iteration of 'root' during the first iteration of 'SLSQP'. Some solvers
and drivers may have sub-steps that are recorded. In those cases, the
iteration number may be of the form '1-3', indicating the third sub-step of the
first iteration.

Since our Paraboloid only has a recorder attached to the driver, our
'paraboloid' shelve file will contain keys of the form 'SLSQP/1', 'SLSQP/2',
etc. To access the data from our run, we can use the following code:

::

    import shelve
    f = shelve.open('paraboloid')

Now, we can access the data using an iteration coordinate.

::

    data = f['SLSQP/1']

This `data` variable has three keys, 'Parameters', 'Unknowns', and 'Residuals'.
Using any of these keys will yield a dictionary containing variable names
mapped to values. For example,

::

    p = data['Parameters']
    print(p)

will print out the dictionary {'p.x': 3.0, 'p.y': -4.0}. Generally, the
variables of interest will be contained in the 'Unknowns' key since that will
contain the objective function values and the values controlled by the
optimizer. For example,

::

    u = data['Unknowns']
    print(u)

will print out the dictionary {'f_xy': -15.0, 'x': 3.0, 'y': -4.0}.
