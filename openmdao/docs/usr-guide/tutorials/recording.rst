.. _OpenMDAO-Recording:

=========
Recording
=========

This tutorial is builds on the :ref:`Optimization of the Paraboloid Tutorial <paraboloid_optimization_tutorial>`
by demonstrating how to save the data generated for future use. Consider the code below:

.. testcode:: recording

    from openmdao.components import IndepVarComp
    from openmdao.core import Component, Group, Problem
    from openmdao.drivers import ScipyOptimizer
    from openmdao.recorders import SqliteRecorder


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

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'

        top.driver.add_desvar('p1.x', low=-50, high=50)
        top.driver.add_desvar('p2.y', low=-50, high=50)
        top.driver.add_objective('p.f_xy')

        recorder = SqliteRecorder('paraboloid')
        top.driver.add_recorder(recorder)

        top.setup()
        top.run()

        top.driver.recorders[0].close()

        print('\n')
        print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))


Two lines are all it takes to record the state of the problem as the
optimizer progresses.

::

    recorder = SqliteRecorder('paraboloid')
    driver.add_recorder(recorder)

We initialize a `SqliteRecorder` by passing it a
`filename` argument. This recorder indirectly uses Python's `sqlite3` module to store the
data generated. In this case, `sqlite3` will open a database file named 'paraboloid'
to use as a back-end.
Actually, OpenMDAO's `SqliteRecorder` makes use of the
`sqlitedict module <https://pypi.python.org/pypi/sqlitedict>`_ because it has a
simple, Pythonic dict-like interface to Python’s sqlite3 database.

We then add the recorder to the driver using `driver.add_recorder`.
Depending on your needs, you are able to add more recorders by using
additional `driver.add_recorder` calls. Solvers also have an `add_recorder`
method that is invoked the same way. This allows you to record the evolution
of variables at lower levels.

While it might not be an issue, it is good practice to close the
recorder explicitly before the program terminates.
For this tutorial with one recorder added to the driver, this is simply done with:

::

    driver.recorders[0].close()

If your model has recorders added to both drivers and solvers,
a way to make sure all recorders are closed is to use code like this:

::

    for recorder in top.driver.recorders:
        recorder.close()

    for sub in top.root.subgroups(recurse=True, include_self=True):
        for recorder in sub.nl_solver.recorders:
            recorder.close()
        for recorder in sub.ln_solver.recorders:
            recorder.close()


Includes and Excludes
=====================

Over the course of an analysis or optimization, the model may generate a very
large amount of data. Since you may not be interested in the value of every
variable at every step, OpenMDAO allows you to filter which variables are
recorded through the use of includes and excludes. The recorder will store
anything that matches the includes filter and that does not match the exclude
filter. By default, the includes are set to `['*']` and the excludes are set to
`[]`, i.e. include everything and exclude nothing.

The includes and excludes filters are set via the `options` structure in the
recorder. If we were only interested in the variable `x` from our Paraboloid
model, we could record that by setting the includes as follows:

::

    recorder = SqliteRecorder('paraboloid')
    recorder.options['includes'] = ['x']

    driver.add_recorder(recorder)

Similarly, if we were interested in everything except the value of `f_xy`, we
could exclude that by doing the following:
::

    recorder = SqliteRecorder('paraboloid')
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

Since our Paraboloid only has a recorder added to the driver, our
'paraboloid' sqlite file will contain keys of the form 'SLSQP/1', 'SLSQP/2',
etc. To access the data from our run, we can use the following code:

::

    import sqlitedict

    db = sqlitedict.SqliteDict( 'paraboloid', 'openmdao' )

There are two arguments to create an instance of SqliteDict. The first, `'paraboloid'`,
is the name of the sqlite database file. The second, `'openmdao'`, is the name of the table
in the sqlite database. For the SqliteRecorder in OpenMDAO, all the
recording is done to the `'openmdao'` table.

Now, we can access the data using an iteration coordinate.

::

    data = db['SLSQP/1']

This `data` variable has four keys, 'timestamp', 'Parameters', 'Unknowns', and 'Residuals'. 'timestamp'
yields the time at which data was recorded:

::

    p = data['timestamp']
    print(p)

The remaining keys will yield a dictionary containing variable names mapped to values. For example,

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
