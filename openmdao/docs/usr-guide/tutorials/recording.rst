.. _OpenMDAO-Recording:

Recording - Saving Data Generated for Future Use
================================================

This tutorial is builds on the :ref:`Optimization of the Paraboloid Tutorial <paraboloid_optimization_tutorial>`
by demonstrating how to save the data generated for future use. Consider the code below:

.. testsetup:: recording_run

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

.. testcode:: recording_run

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

        def linearize(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy', 'x'] = 2.0 * x - 6.0 + y
            J['f_xy', 'y'] = 2.0 * y + 8.0 + x
            return J


    top = Problem()

    root = top.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))
    root.add('p', Paraboloid())

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'

    top.driver.add_desvar('p1.x', lower=-50, upper=50)
    top.driver.add_desvar('p2.y', lower=-50, upper=50)
    top.driver.add_objective('p.f_xy')

    recorder = SqliteRecorder('paraboloid')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    top.driver.add_recorder(recorder)

    top.setup()
    top.run()

    top.cleanup()  # this closes all recorders

    print('\n')
    print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))


.. testoutput:: recording_run
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: ...-27.333333...
                Iterations: 5
                Function evaluations: 6
                Gradient evaluations: 5
    Optimization Complete
    -----------------------------------


    Minimum of -27.333333 found at (6.666667, -7.333333)


.. testcleanup:: recording_run

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

.. testsetup:: recording1

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

    from openmdao.api import SqliteRecorder, Problem, Group
    top = Problem()
    root = top.root = Group()

These next four lines are all it takes to record the state of the problem as the
optimizer progresses. Notice that because by default, recorders only record
`Unknowns`, if we also want to record `Parameters` and `metadata`, we must
set those recording options. (We could also record `Resids` by using the
`record_resids` option but this problem does not have residuals. )

.. testcode:: recording1

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

While it might not be an issue, it is good practice to tell
the `Problem` explicitly to clean things up before the program terminates.
This will close all recorders and potentially release other operating system
resources.

This is simply done in this case by calling:

.. testcode:: recording1

    top.cleanup()


.. testcleanup:: recording1

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')


Includes and Excludes
------------------------------

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

.. testsetup:: recording3

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

    from openmdao.api import SqliteRecorder, Problem, Group
    top = Problem()
    root = top.root = Group()

.. testcode:: recording3

    recorder = SqliteRecorder('paraboloid')
    recorder.options['includes'] = ['x']

    top.driver.add_recorder(recorder)

.. testcleanup:: recording3

    top.cleanup()

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

Similarly, if we were interested in everything except the value of `f_xy`, we
could exclude that by doing the following:

.. testsetup:: recording4

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

    from openmdao.api import SqliteRecorder, Problem, Group
    top = Problem()
    root = top.root = Group()

.. testcode:: recording4

    recorder = SqliteRecorder('paraboloid')
    recorder.options['excludes'] = ['f_xy']

    top.driver.add_recorder(recorder)

The includes and excludes filters will accept glob arguments. For example,
`recorder.options['excludes'] = ['comp1.*']` would exclude any variable
that starts with "comp1.".

.. testcleanup:: recording4

    top.cleanup()

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')


Accessing Recorded Data
------------------------------

While each recorder stores data differently in order to match the
file format, the common theme for accessing data is the iteration coordinate.
The iteration coordinate describes where and when in the execution hierarchy
the data was collected. Iteration coordinates are strings formatted as pairs
of names and iteration numbers separated by '|'. For example,
'rank0:SLSQP|1|root|2|G1|3' would describe the third iteration of 'G1' during the
second iteration of 'root' during the first iteration of 'SLSQP'. Some solvers
and drivers may have sub-steps that are recorded. In those cases, the
iteration number may be of the form '1-3', indicating the third sub-step of the
first iteration.

Since our Paraboloid only has a recorder added to the driver, our
'paraboloid' SQLite file will contain keys of the form 'rank0:SLSQP|1', 'rank0:SLSQP|2',
etc. To access the data from our run, we can use the following code:

.. testsetup:: reading

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

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

        def linearize(self, params, unknowns, resids):
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

    top.driver.add_desvar('p1.x', lower=-50, upper=50)
    top.driver.add_desvar('p2.y', lower=-50, upper=50)
    top.driver.add_objective('p.f_xy')

    recorder = SqliteRecorder('paraboloid')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    top.driver.add_recorder(recorder)

    top.setup()
    top.run()

    top.cleanup()

.. testoutput:: reading
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: ...-27.333333...
                Iterations: 5
                Function evaluations: 6
                Gradient evaluations: 5
    Optimization Complete
    -----------------------------------


    Minimum of -27.333333 found at (6.666667, -7.333333)



.. testcode:: reading

    import sqlitedict
    from pprint import pprint

    db = sqlitedict.SqliteDict( 'paraboloid', 'iterations' )


There are two arguments to create an instance of SqliteDict. The first, `'paraboloid'`,
is the name of the SQLite database file. The second, `'iterations'`, is the name of the table
in the SQLite database containing the iteration values.

Now, we can access the data using an iteration coordinate. It is not always obvious what are the
iteration coordinates. To see what iteration coordinates were recorded, use the `keys` method
on the `db` object:

.. testcode:: reading

    print( list( db.keys() ) ) # list() needed for compatibility with Python 3. Not needed for Python 2

which will print out:

.. testoutput:: reading
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    ['rank0:SLSQP|1', 'rank0:SLSQP|2', 'rank0:SLSQP|3', 'rank0:SLSQP|4', 'rank0:SLSQP|5', 'rank0:SLSQP|6']

::

    ['rank0:SLSQP|1', 'rank0:SLSQP|2', 'rank0:SLSQP|3', 'rank0:SLSQP|4', 'rank0:SLSQP|5', 'rank0:SLSQP|6']


Now we can get the values for the first iteration coordinate:

.. testcode:: reading

    data = db['rank0:SLSQP|1']

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


.. testcleanup:: reading

    db.close()
    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')


Accessing Recorded Metadata
===========================

Finally, since our code told the recorder to record metadata, we can read that from the file as well.
The metadata is only recorded once and is in its own table in the SQLite database.
The name of the SQLite table containing the derivatives is called `metadata`.

.. testsetup:: reading_metadata

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

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

        def linearize(self, params, unknowns, resids):
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

    top.driver.add_desvar('p1.x', lower=-50, upper=50)
    top.driver.add_desvar('p2.y', lower=-50, upper=50)
    top.driver.add_objective('p.f_xy')

    recorder = SqliteRecorder('paraboloid')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    recorder.options['record_derivs'] = True
    top.driver.add_recorder(recorder)

    top.setup()
    top.run()

    top.cleanup()

.. testoutput:: reading_metadata
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: ...-27.333333...
                Iterations: 5
                Function evaluations: 6
                Gradient evaluations: 5
    Optimization Complete
    -----------------------------------


    Minimum of -27.333333 found at (6.666667, -7.333333)



.. testcode:: reading_metadata

    import sqlitedict
    from pprint import pprint

    db = sqlitedict.SqliteDict( 'paraboloid', 'metadata' )




.. testcode:: reading_metadata

    u_meta = db['Unknowns']
    pprint(u_meta)
    p_meta = db['Parameters']
    pprint(p_meta)
    print(db['format_version'])

.. testoutput:: reading_metadata
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    {'p.f_xy': {'is_objective': True,
                'pathname': 'p.f_xy',
                'shape': 1,
                'size': 1,
                'top_promoted_name': 'p.f_xy',
                'val': 0.0},
     'p1.x': {'_canset_': True,
              'is_desvar': True,
              'pathname': 'p1.x',
              'shape': 1,
              'size': 1,
              'top_promoted_name': 'p1.x',
              'val': 3.0},
     'p2.y': {'_canset_': True,
              'is_desvar': True,
              'pathname': 'p2.y',
              'shape': 1,
              'size': 1,
              'top_promoted_name': 'p2.y',
              'val': -4.0}}
    {'p.x': {'pathname': 'p.x',
             'shape': 1,
             'size': 1,
             'top_promoted_name': 'p.x',
             'val': 0.0},
     'p.y': {'pathname': 'p.y',
             'shape': 1,
             'size': 1,
             'top_promoted_name': 'p.y',
             'val': 0.0}}
    4

This code prints out the following:

::

    {'p.f_xy': {'is_objective': True,
                'pathname': 'p.f_xy',
                'shape': 1,
                'size': 1,
                'top_promoted_name': 'p.f_xy',
                'val': 0.0},
     'p1.x': {'_canset_': True,
              'is_desvar': True,
              'pathname': 'p1.x',
              'shape': 1,
              'size': 1,
              'top_promoted_name': 'p1.x',
              'val': 3.0},
     'p2.y': {'_canset_': True,
              'is_desvar': True,
              'pathname': 'p2.y',
              'shape': 1,
              'size': 1,
              'top_promoted_name': 'p2.y',
              'val': -4.0}}
    {'p.x': {'pathname': 'p.x',
             'shape': 1,
             'size': 1,
             'top_promoted_name': 'p.x',
             'val': 0.0},
     'p.y': {'pathname': 'p.y',
             'shape': 1,
             'size': 1,
             'top_promoted_name': 'p.y',
             'val': 0.0}}
    4


.. testcleanup:: reading_metadata

    db.close()
    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')


Accessing Recorded Derivatives
==============================

Sometimes it is useful for debugging purposes to look at the derivatives computed. If the user has turned on recording
using the option:

::

    recorder.options['record_derivs'] = True

then the derivatives are also recorded to the case recording file.

.. testsetup:: reading_derivs

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

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

        def linearize(self, params, unknowns, resids):
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

    top.driver.add_desvar('p1.x', lower=-50, upper=50)
    top.driver.add_desvar('p2.y', lower=-50, upper=50)
    top.driver.add_objective('p.f_xy')

    recorder = SqliteRecorder('paraboloid')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    recorder.options['record_derivs'] = True
    top.driver.add_recorder(recorder)

    top.setup()
    top.run()

    top.cleanup()

.. testoutput:: reading_derivs
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: ...-27.3333333333...
                Iterations: 5
                Function evaluations: 6
                Gradient evaluations: 5
    Optimization Complete
    -----------------------------------


    Minimum of -27.333333 found at (6.666667, -7.333333)



.. testcode:: reading_derivs

    import sqlitedict
    from pprint import pprint

    db = sqlitedict.SqliteDict( 'paraboloid', 'derivs' )


The name of the SQLite table containing the derivatives is called `derivs`.

Just like before, we can access the data using an iteration coordinate. The derivative value can either be an `ndarray` or a
`dict`, depending on the optimizer being used.

.. testcode:: reading_derivs

    data = db['rank0:SLSQP|1']
    u = data['Derivatives']
    pprint(u)

.. testoutput:: reading_derivs
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    array([[-4.,  3.]])

will print out:

::

    array([[-4.,  3.]])

.. testcleanup:: reading_derivs

    db.close()
    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

The CaseReader
==============
The SqliteCaseRecorder and HDF5CaseRecorder are the two main ways to save data from an OpenMDAO run.  Accessing
the data, as the previous section shows, requires some knowledge of the structure of the recorded file, which is
a function of the recorder used.  Furthermore, finding the key of the desired iteration coordinate is a process
that needs to be repeated each time recorded data is loaded.

In an effort to make this process independent of the recorder used, the CaseReader class gives the user a
common interface to recorded data, regardless of format.  Iteration coordinates are accessible by both their
coordinate string descriptor, or as a standard python index.

.. testsetup:: casereader

   import os
   if os.path.exists('paraboloid'):
        os.remove('paraboloid')

   from openmdao.api import IndepVarComp, Component, Group, Problem, ScipyOptimizer, SqliteRecorder

.. testcode:: casereader

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

       def linearize(self, params, unknowns, resids):
           """ Jacobian for our paraboloid."""

           x = params['x']
           y = params['y']
           J = {}

           J['f_xy', 'x'] = 2.0 * x - 6.0 + y
           J['f_xy', 'y'] = 2.0 * y + 8.0 + x
           return J


   top = Problem()

   root = top.root = Group()

   root.add('p1', IndepVarComp('x', 3.0))
   root.add('p2', IndepVarComp('y', -4.0))
   root.add('p', Paraboloid())

   root.connect('p1.x', 'p.x')
   root.connect('p2.y', 'p.y')

   top.driver = ScipyOptimizer()
   top.driver.options['optimizer'] = 'SLSQP'

   top.driver.add_desvar('p1.x', lower=-50, upper=50)
   top.driver.add_desvar('p2.y', lower=-50, upper=50)
   top.driver.add_objective('p.f_xy')

   recorder = SqliteRecorder('paraboloid')
   recorder.options['record_params'] = True
   recorder.options['record_metadata'] = True
   top.driver.add_recorder(recorder)

   top.setup()
   top.run()

   top.cleanup()  # this closes all recorders

.. testoutput:: casereader
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    Optimization terminated successfully.    (Exit mode 0)
                Current function value: ...-27.333333...
                Iterations: 5
                Function evaluations: 6
                Gradient evaluations: 5
    Optimization Complete
    -----------------------------------

A CaseReader instance contains two main sets of data:  metadata for the parameters and unknowns, and data from
each case.  The metadata is accessed via the properties `parameters` and `unknowns`.  For instance, in the following
code

.. testcode:: casereader

    from openmdao.api import CaseReader

    cr = CaseReader('paraboloid')
    cr.unknowns

`cr` will contain a dictionary:

::    {'p1.x': {'val': 3.0, 'is_desvar': True, 'shape': 1, 'pathname': 'p1.x', 'top_promoted_name': 'p1.x', '_canset_': True, 'size': 1}, 'p.f_xy': {'is_objective': True, 'val': 0.0, 'shape': 1, 'pathname': 'p.f_xy', 'top_promoted_name': 'p.f_xy', 'size': 1}, 'p2.y': {'val': -4.0, 'is_desvar': True, 'shape': 1, 'pathname': 'p2.y', 'top_promoted_name': 'p2.y', '_canset_': True, 'size': 1}}

.. testoutput:: casereader
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE



To show the case iteration coordinates in the recorded file:

.. testcode:: casereader

   print(cr.list_cases())

which outputs:

::

   ('rank0:SLSQP|1', 'rank0:SLSQP|2', 'rank0:SLSQP|3', 'rank0:SLSQP|4', 'rank0:SLSQP|5', 'rank0:SLSQP|6')

.. testoutput:: casereader
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    ('rank0:SLSQP|1', 'rank0:SLSQP|2', 'rank0:SLSQP|3', 'rank0:SLSQP|4', 'rank0:SLSQP|5', 'rank0:SLSQP|6')

It's common to only care about the final case (the solution) of the optimization.  To load the data from the
final case we can either access it via its case iteration coordinate:

.. testcode:: casereader

   last_case = cr.get_case('rank0:SLSQP|6')

or, simply use an index (where -1 is the Pythonic way for accessing the last index of a list)

.. testcode:: casereader

   last_case = cr.get_case(-1)


The get_case method returns a Case object, which has properties for `parameters`, `unkowns`, `derivs`,
and `resids`.  Each of these is a dictionary, in which the path of the appropriate variable returns
the respective value of the param, unknown, deriv, or resid.  In general, the most commonly accessed
information are the unknowns.  If we access the case as a dictionary where unknown variables are the
keys, it will return values of those unknowns.  For instance, we can access the values
of x, y, and f at the solution of the paraboloid using:

.. testcode:: casereader

   x = last_case['p1.x']
   y = last_case['p2.y']
   f_xy = last_case['p.f_xy']

   print('Minimum is {0:7.4f} at x={1:7.4f} and y={2:7.4f}'.format(f_xy, x, y))


which outputs

::

   Minimum is -27.3333 at x= 6.6667 and y=-7.3333

.. testoutput:: casereader
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    Minimum is -27.3333 at x= 6.6667 and y=-7.3333

.. testcleanup:: casereader

    import os
    if os.path.exists('paraboloid'):
        os.remove('paraboloid')

.. tags:: Tutorials, Data Recording
