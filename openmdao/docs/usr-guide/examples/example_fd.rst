.. index:: Finite Difference Tutorial

Finite Difference
-----------------

OpenMDAO allows you to specify analytic derivatives for your models, but it
is not a requirement. You can choose instead to allow any part or all of your
model to be finite differenced to your specifications. Any `System` (i.e.,
`Component` or `Group`) has a set of options called `fd_options` which can be
used to turn on finite difference and control its settings. The following
settings are available for all groups.

force_fd : bool
    Set to True to finite difference this system
form : string
    Finite difference mode ('forward', 'backward', 'central')
step_size : float
    Default finite difference stepsize
step_type : string
    Set to 'absolute' or 'relative'

The following examples will show you how to turn on finite difference for a
`Component`, a `Group`, and a full model.

Finite Difference on a Component
================================

Let us define a simple component that takes an input and multiplies it by
3.0, and include analytic derivatives. We will insert some print statements
so that we can watch what it does.

.. testcode:: fd_example

        from __future__ import print_function

        from openmdao.api import Component, Group, Problem, IndepVarComp


        class SimpleComp(Component):
            """ A simple component that provides derivatives. """

            def __init__(self):
                super(SimpleComp, self).__init__()

                # Params
                self.add_param('x', 2.0)

                # Unknowns
                self.add_output('y', 0.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much.  Just multiply by 3"""
                unknowns['y'] = 3.0*params['x']
                print('Execute', self.name)

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}
                J[('y', 'x')] = 3.0
                print('Calculate Derivatives:', self.name)
                return J

Now let's chain 4 of them together in series. OpenMDAO can calculate a
gradient across the chain using all of the individual derivatives. However,
let's finite difference the 2nd and 4th component in the chain.

.. testcode:: fd_example

                class Model(Group):
                    """ Simple model to experiment with finite difference."""

                    def __init__(self):
                        super(Model, self).__init__()

                        self.add('px', IndepVarComp('x', 2.0))

                        self.add('comp1', SimpleComp())
                        self.add('comp2', SimpleComp())
                        self.add('comp3', SimpleComp())
                        self.add('comp4', SimpleComp())

                        self.connect('px.x', 'comp1.x')
                        self.connect('comp1.y', 'comp2.x')
                        self.connect('comp2.y', 'comp3.x')
                        self.connect('comp3.y', 'comp4.x')

                        # Tell these components to finite difference
                        self.comp2.fd_options['force_fd'] = True
                        self.comp2.fd_options['form'] = 'central'
                        self.comp2.fd_options['step_size'] = 1.0e-4

                        self.comp4.fd_options['force_fd'] = True
                        self.comp4.fd_options['form'] = 'central'
                        self.comp4.fd_options['step_size'] = 1.0e-4

To do so, we set 'force_fd' to True in comp2 and comp4. To further ilustrate
setting options, we select central difference with a stepsize of 1.0e-4. Now
let's run the model.

.. testcode:: fd_example

    # Setup and run the model.
    top = Problem()
    top.root = Model()
    top.setup()
    top.run()

    print('\n\nStart Calc Gradient')
    print ('-'*25)

    J = top.calc_gradient(['px.x'], ['comp4.y'])
    print(J)

We get output that looks like this:

.. testoutput:: fd_example
   :options: +ELLIPSIS

   ...
   Start Calc Gradient
   -------------------------
   Calculate Derivatives: comp1
   Execute comp2
   Execute comp2
   Calculate Derivatives: comp3
   Execute comp4
   Execute comp4
   [[ 81.]]


The output shows that comp2 and comp4 aren't using their `linearize` function,
but instead are executing twice, as would be expected when using central
difference.


Complex Step on a Component
===========================

If you have a pure python component (or an external code that can support
complex inputs and outputs) then you can also choose to use complex step to
calculate the Jacobian of a component. This will give more accurate
derivatives that are insensitive to the step size. Like finite difference,
complex step runs your component using the `apply_nonlinear` or
`solve_nonlinear` functions, but it applies a step in the complex direction.
To activate it, you just need to set the `form` option on a Compontent to
"complex_step":

.. testcode:: fd_example
    :hide:

    # Setup and run the model.
    top = Problem()
    top.root = Model()
    top.setup()
    top.run()
    self = top.root

.. testoutput:: fd_example
   :hide:

   Execute comp1
   Execute comp2
   Execute comp3
   Execute comp4

.. testcode:: fd_example

    self.comp2.fd_options['form'] = 'complex_step'

In many cases, this will require no other changes to your code, as long as
all of the calculation in your `solve_nonlinear` and `apply_nonlinear`
support complex numbers. During a complex step, the incoming `params` vector
will return a complex number when a variable is being stepped. Likewise, the
`unknowns` and `resids` vectors will accept complex values. If you are
allocating temporary numpy arrays, remember to conditionally set their dtype
based on the dtype in the unknowns vector.

At present, complex step is not supported on groups of components, so you will need to complex step them individually.

Finite Difference on Groups of Components
=========================================

Next, we show how to finite difference a group of components together. For
this example, let's finite difference comp2 and comp3 as one entity. To do
this, we need to add a Group to the model called 'sub' and place comp2 and
comp3 in that group.

.. testcode:: fd_example

    class Model(Group):
        """ Simple model to experiment with finite difference."""

        def __init__(self):
            super(Model, self).__init__()

            self.add('px', IndepVarComp('x', 2.0))

            self.add('comp1', SimpleComp())

            # 2 and 3 are in a sub Group
            sub = self.add('sub', Group())
            sub.add('comp2', SimpleComp())
            sub.add('comp3', SimpleComp())

            self.add('comp4', SimpleComp())

            self.connect('px.x', 'comp1.x')
            self.connect('comp1.y', 'sub.comp2.x')
            self.connect('sub.comp2.y', 'sub.comp3.x')
            self.connect('sub.comp3.y', 'comp4.x')

            # Tell the group with comps 2 and 3 to finite difference
            self.sub.fd_options['force_fd'] = True
            self.sub.fd_options['step_size'] = 1.0e-4

To turn on finite difference, we have set 'force_fd' to True in `self.sub`.

There is no change to the execution code. The result looks like this:

.. testcode:: fd_example
    :hide:

    # Setup and run the model.
    top = Problem()
    top.root = Model()
    top.setup()
    top.run()

    print('\n\nStart Calc Gradient')
    print ('-'*25)

    J = top.calc_gradient(['px.x'], ['comp4.y'])
    print(J)

.. testoutput:: fd_example
   :options: +ELLIPSIS

   ...
   Start Calc Gradient
   -------------------------
   Calculate Derivatives: comp1
   Execute comp2
   Execute comp3
   Calculate Derivatives: comp4
   [[ 81.]]

Here we see that, instead of calling 'linearize', comp2 and comp3 execute
during finite differnce of the group that owns them. This is as we expect.

Finite Difference on an Entire Model
====================================

Finally, let's finite difference the whole model in one operation. We tell
OpenMDAO to do this by setting force_fd in the top `Group`.

.. testcode:: fd_example

    class Model(Group):
        """ Simple model to experiment with finite difference."""

        def __init__(self):
            super(Model, self).__init__()

            self.add('px', IndepVarComp('x', 2.0))

            self.add('comp1', SimpleComp())
            self.add('comp2', SimpleComp())
            self.add('comp3', SimpleComp())
            self.add('comp4', SimpleComp())

            self.connect('px.x', 'comp1.x')
            self.connect('comp1.y', 'comp2.x')
            self.connect('comp2.y', 'comp3.x')
            self.connect('comp3.y', 'comp4.x')

            # Tell the whole model to finite difference
            self.fd_options['force_fd'] = True

Nothing else changes in the original model. When we run it, we get:

.. testcode:: fd_example
    :hide:

    # Setup and run the model.
    top = Problem()
    top.root = Model()
    top.setup()
    top.run()

    print('\n\nStart Calc Gradient')
    print ('-'*25)

    J = top.calc_gradient(['px.x'], ['comp4.y'])
    print(J)

.. testoutput:: fd_example
   :options: +ELLIPSIS

   ...
   Start Calc Gradient
   -------------------------
   Execute comp1
   Execute comp2
   Execute comp3
   Execute comp4
   [[ 81.00000002]]

So here, `linearize` is never called in any component as the finite difference
just executes the components in sequence. This is also as expected.

.. _`parallel_finite_difference`:

Parallel Finite Difference
==========================

Suppose you need to calculate a bunch of finite differences, either because
you have a bunch of different design variables, or maybe just a single design
variable that happens to be an array.  OpenMDAO has a special `Group` called
a `ParallelFDGroup` that will allow you to calculate multiple finite differences
in parallel.

Let's start off our example by creating a `Component` that has array inputs
and outputs.

.. testcode:: fd_par_example

    import numpy
    import time
    from openmdao.api import Problem, Component, ParallelFDGroup, IndepVarComp
    from openmdao.core.mpi_wrap import MPI

    class ArrayFDComp(Component):
        """ A simple component takes an array input, produces
        an array output, and does not provide derivatives.

        Args
        ----
        size : int
            The size of the input and output variables.

        delay : float
            The number of seconds to sleep during the solve_nonlinear
            call.
        """

        def __init__(self, size, delay):
            super(ArrayFDComp, self).__init__()

            self.delay = delay

            # Params
            self.add_param('x', numpy.zeros(size))

            # Unknowns
            self.add_output('y', numpy.zeros(size))

        def solve_nonlinear(self, params, unknowns, resids):
            """ Doesn't do much.  Just multiply by 3"""
            time.sleep(self.delay)
            unknowns['y'] = 3.0*params['x']

The following check is only here so that our doc tests, which don't run
under MPI, will pass.  If you use a `ParallelFDGroup` when you're not running
under MPI, it will behave like a regular Group.

.. testcode:: fd_par_example

    if MPI:
        from openmdao.api import PetscImpl as impl
    else:
        from openmdao.api import BasicImpl as impl

    prob = Problem(impl=impl)

For this simple example, we'll do parallel finite difference at the top level
of our model, by using a `ParallelFDGroup` in place of a regular `Group`,
but you can use `ParallelFDGroup` to replace other `Groups` inside of your
model as well.  `ParallelFDGroup` takes an argument that tells it how many finite
differences to perform in parallel.  In this case, we'll do two parallel
finite differences.  The size of our design variable is 10, so we'll perform
5 finite differences in each of our two processes.  Note that the number of
design variables doesn't have to divide equally among the processes, but
you'll get the best speedup when it does.

.. testcode:: fd_par_example

    # Create a ParallelFDGroup that does 2 finite differences in parallel.
    prob.root = ParallelFDGroup(2)

    # let's use size 10 arrays and a delay of 0.1 seconds
    size = 10
    delay = 0.1

    prob.root.add('P1', IndepVarComp('x', numpy.ones(size)))
    prob.root.add('C1', ArrayFDComp(size, delay=delay))

    prob.root.connect('P1.x', 'C1.x')

    prob.driver.add_desvar('P1.x')
    prob.driver.add_objective('C1.y')

    prob.setup(check=False)
    prob.run()

Now we'll calculate the Jacobian using our parallel finite difference setup.

.. testcode:: fd_par_example

    J = prob.calc_gradient(['P1.x'], ['C1.y'], mode='fd',
                           return_format='dict')

    print(J['C1.y']['P1.x'])


When we're done, our J should look like this:


.. testoutput:: fd_par_example
    :options: +ELLIPSIS

    [[ 3.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  3.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  3.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  3.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  3.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  3.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  3.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  3.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  3.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  3.]]

You can experiment with this example by changing the size of the arrays and
the length of the delay.  You'll find that you get the most speedup from
parallel finite difference when the delay is longer.
