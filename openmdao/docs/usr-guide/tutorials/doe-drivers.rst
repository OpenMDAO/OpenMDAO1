.. _OpenMDAO-DoE_Drivers:

=========
DoE Drivers
=========

This tutorial shows you how to use the OptimizedLatinHypercubeDriver to run a design of experiments (DoE).  Below is an example exploring the 'design space' of a simple Paraboloid; real-world applications would obviously be much more complex.

.. testcode:: olhc_driver

    from openmdao.api import IndepVarComp, Group, Problem, ScipyOptimizer, ExecComp, DumpRecorder
    from openmdao.test.paraboloid import Paraboloid

    from openmdao.drivers.latinhypercube_driver import LatinHypercubeDriver, OptimizedLatinHypercubeDriver

    top = Problem()
    root = top.root = Group()

    root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
    root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
    root.add('comp', Paraboloid(), promotes=['*'])

    top.driver = LatinHypercubeDriver(10)
    top.driver.add_desvar('x', low=-50.0, high=50.0)
    top.driver.add_desvar('y', low=-50.0, high=50.0)

    top.driver.add_objective('f_xy')

    recorder = DumpRecorder('paraboloid')
    recorder.options['record_params'] = True
    recorder.options['record_unknowns'] = False
    recorder.options['record_resids'] = False
    top.driver.add_recorder(recorder)

    top.setup(check=False)
    top.run()

    top.driver.recorders[0].close()


.. Copy over the recorded file so we can test reading it later and so other testing code does not mess it up
.. testcleanup:: recording_run

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

.. testsetup:: recording

    from openmdao.api import SqliteRecorder, Problem, Group
    top = Problem()
    root = top.root = Group()

These next four lines are all it takes to record the state of the problem as the
optimizer progresses. Notice that because by default, recorders only record 
`Unknowns`, if we also want to record `Parameters` and `metadata`, we must 
set those recording options. (We could also record `Resids` by using the 
`record_metadata` option but this problem does not have residuals. )

.. testcode:: recording

    recorder = SqliteRecorder('paraboloid')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    top.driver.add_recorder(recorder)

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

.. testcode:: recording

    top.driver.recorders[0].close()

If your model has recorders added to both drivers and solvers,
a way to make sure all recorders are closed is to use code like this:

.. testcode:: recording

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

.. testcode:: recording

    recorder = SqliteRecorder('paraboloid')
    recorder.options['includes'] = ['x']

    top.driver.add_recorder(recorder)

Similarly, if we were interested in everything except the value of `f_xy`, we
could exclude that by doing the following:

.. testcode:: recording

    recorder = SqliteRecorder('paraboloid')
    recorder.options['excludes'] = ['f_xy']

    top.driver.add_recorder(recorder)

The includes and excludes filters will accept glob arguments. For example,
`recorder.options['excludes'] = ['comp1.*']` would exclude any variable
that starts with "comp1.".

.. testcleanup:: recording

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')


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

.. testsetup:: reading

    from openmdao.api import IndepVarComp, Component, Group, Problem, ScipyOptimizer, SqliteRecorder

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


    # to keep the output of the run from doctest which does not handle output from setup well!
    import os
    import sys
    f = open(os.devnull, 'w')
    sys.stdout = f

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
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    top.driver.add_recorder(recorder)

    top.setup()
    top.run()

    top.driver.recorders[0].close()

.. testoutput:: reading
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: [-27.33333333]
                Iterations: 5
                Function evaluations: 6
                Gradient evaluations: 5
    Optimization Complete
    -----------------------------------


    Minimum of -27.333333 found at (6.666667, -7.333333)



.. testcode:: reading

    import sqlitedict
    from pprint import pprint

    db = sqlitedict.SqliteDict( 'paraboloid', 'openmdao' )


There are two arguments to create an instance of SqliteDict. The first, `'paraboloid'`,
is the name of the sqlite database file. The second, `'openmdao'`, is the name of the table
in the sqlite database. For the SqliteRecorder in OpenMDAO, all the
recording is done to the `'openmdao'` table.

Now, we can access the data using an iteration coordinate.

.. testcode:: reading

    data = db['SLSQP/1']

This `data` variable has four keys, 'timestamp', 'Parameters', 'Unknowns', and 'Residuals'. 'timestamp'
yields the time at which data was recorded:

.. testcode:: reading

    p = data['timestamp']
    print(p)

.. testoutput:: reading
   :hide:
   :options: +ELLIPSIS

   ...

The remaining keys will yield a dictionary containing variable names mapped to values. Generally, the
variables of interest will be contained in the 'Unknowns' key since that will
contain the objective function values and the values controlled by the
optimizer. For example,

.. testcode:: reading

    u = data['Unknowns']
    pprint(u)

.. testoutput:: reading
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    {'p.f_xy': -15.0, 'p1.x': 3.0, 'p2.y': -4.0}

will print out the dictionary:

::

    {'f_xy': -15.0, 'x': 3.0, 'y': -4.0}

You can also access the values for the `Parameters`:

.. testcode:: reading

    p = data['Parameters']
    pprint(p)

.. testoutput:: reading
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    {'p.x': 3.0, 'p.y': -4.0}

Which will print out the dictionary:

::

    {'p.x': 3.0, 'p.y': -4.0}

Finally, since our code told the recorder to record metadata, we can read that from the file as well.
Notice that since metadata is only recorded once, it is a top level element of the dictionary, rather than a
sub-dictionary of an interation coordinate. It contains sub-dictionaries for metadata about
`Unknowns`, `Parameters`, `Resids`.

.. testcode:: reading

    data = db['metadata']
    u_meta = data['Unknowns']
    pprint(u_meta)
    p_meta = data['Parameters']
    pprint(p_meta)

.. testoutput:: reading
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    {'p.f_xy': {'pathname': 'p.f_xy',
                'promoted_name': 'p.f_xy',
                'shape': 1,
                'size': 1,
                'top_promoted_name': 'p.f_xy',
                'val': array([ 0.])},
     'p1.x': {'pathname': 'p1.x',
              'promoted_name': 'p1.x',
              'shape': 1,
              'size': 1,
              'top_promoted_name': 'p1.x',
              'val': array([ 3.])},
     'p2.y': {'pathname': 'p2.y',
              'promoted_name': 'p2.y',
              'shape': 1,
              'size': 1,
              'top_promoted_name': 'p2.y',
              'val': array([-4.])}}
    {'p.x': {'owned': True,
             'pathname': 'p.x',
             'promoted_name': 'p.x',
             'shape': 1,
             'size': 1,
             'top_promoted_name': 'p.x',
             'val': array([ 0.])},
     'p.y': {'owned': True,
             'pathname': 'p.y',
             'promoted_name': 'p.y',
             'shape': 1,
             'size': 1,
             'top_promoted_name': 'p.y',
             'val': array([ 0.])}}

This code prints out the following:

::

    {'p.f_xy': {'pathname': 'p.f_xy',
                'promoted_name': 'p.f_xy',
                'shape': 1,
                'size': 1,
                'top_promoted_name': 'p.f_xy',
                'val': array([ 0.])},
     'p1.x': {'pathname': 'p1.x',
              'promoted_name': 'p1.x',
              'shape': 1,
              'size': 1,
              'top_promoted_name': 'p1.x',
              'val': array([ 3.])},
     'p2.y': {'pathname': 'p2.y',
              'promoted_name': 'p2.y',
              'shape': 1,
              'size': 1,
              'top_promoted_name': 'p2.y',
              'val': array([-4.])}}
    {'p.x': {'owned': True,
             'pathname': 'p.x',
             'promoted_name': 'p.x',
             'shape': 1,
             'size': 1,
             'top_promoted_name': 'p.x',
             'val': array([ 0.])},
     'p.y': {'owned': True,
             'pathname': 'p.y',
             'promoted_name': 'p.y',
             'shape': 1,
             'size': 1,
             'top_promoted_name': 'p.y',
             'val': array([ 0.])}}


.. testcleanup:: reading

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')
