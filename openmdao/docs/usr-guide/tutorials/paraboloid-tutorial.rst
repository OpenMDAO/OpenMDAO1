.. _`paraboloid_tutorial`:

Paraboloid Tutorial - Simple Optimization Problem
=================================================

This tutorial will show you how to set up a simple optimization of a paraboloid.
You'll create a paraboloid `Component` (with analytic derivatives), then put it
into a `Problem` and set up an optimizer `Driver` to minimize an objective function.

Here is the code that defines the paraboloid and then runs it. You can copy
this code into a file, and run it directly.

.. testcode:: parab

    from __future__ import print_function

    from openmdao.api import IndepVarComp, Component, Problem, Group

    class Paraboloid(Component):
        """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

        def __init__(self):
            super(Paraboloid, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', shape=1)

        def solve_nonlinear(self, params, unknowns, resids):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
            """

            x = params['x']
            y = params['y']

            unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

        def linearize(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy', 'x'] = 2.0*x - 6.0 + y
            J['f_xy', 'y'] = 2.0*y + 8.0 + x
            return J

    if __name__ == "__main__":

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.setup()
        top.run()

        print(top['p.f_xy'])


Now we will go through each section and explain how this code works.

Building the component
----------------------

::

    from __future__ import print_function

    from openmdao.api import IndepVarComp, Component, Problem, Group

We need to import some OpenMDAO classes. We also import the print_function to
ensure compatibility between Python 2.x and 3.x. You don't need the import if
you are running in Python 3.x.

::

    class Paraboloid(Component):

OpenMDAO provides a base class, `Component`, which you should inherit from to build
your own components and wrappers for analysis codes. `Components` can declare
three kinds of variables, *parameters*, *outputs* and *states*. A `Component`
operates on its parameters to compute unknowns, which can be explicit
outputs or implicit states. For the `Paraboloid` `Component`, we will only be
using explicit outputs.

::

        def __init__(self):
            super(Paraboloid, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', shape=1)


This code defines the input parameters of the `Component`, `x` and `y`, and
initializes them to 0.0. These will be design variables which could be used to
minimize the output when doing optimization. It also defines the explicit
output, `f_xy`, but only gives it a shape. If shape is 1, the value is
initialized to *0.0*, a scalar.  If shape is any other value, the value
of the variable is initialized to *numpy.zeros(shape, dtype=float)*.

::

        def solve_nonlinear(self, params, unknowns, resids):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
            Optimal solution (minimum): x = 6.6667; y = -7.3333
            """

            x = params['x']
            y = params['y']

            unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

The `solve_nonlinear` method is responsible for calculating outputs for a
given set of parameters. The parameters are given in the `params` dictionary
that is passed in to this method. Similarly, the outputs are assigned values
using the `unknowns` dictionary that is passed in.

::

        def linearize(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy','x'] = 2.0*x - 6.0 + y
            J['f_xy','y'] = 2.0*y + 8.0 + x
            return J

The `linearize` method is used to compute analytic partial derivatives of the
`unknowns` with respect to `params` (partial derivatives in OpenMDAO context refer to
derivatives for a single component by itself). The returned value, in this case `J`,
should be a dictionary whose keys are tuples of the form (‘unknown’, ‘param’) and
whose values are n-d arrays or scalars. Just like for `solve_nonlinear`, the values for the
parameters are accessed using dictionary arguments to the function.

The definition of the Paraboloid Component class is now complete. We will now
make use of this class to run a model.

Setting up the model
--------------------

::

    if __name__ == "__main__":

        top = Problem()
        root = top.root = Group()

An instance of an OpenMDAO `Problem` is always the top object for running a
model. Each `Problem` in OpenMDAO must contain a root `Group`. A `Group` is a
`System` that contains other `Components` or `Groups`.

This code instantiates a `Problem` object and sets the root to be an empty `Group`.

::

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))

Now it is time to add components to the empty group. `IndepVarComp`
is a `Component` that provides the source for a variable which we can later give
to a `Driver` as a design variable to control.

We created two `IndepVarComps` (one for each param on the `Paraboloid`
component), gave them names, and added them to the root `Group`. The `add`
method takes a name as the first argument, and a `Component` instance as the
second argument.  The numbers 3.0 and -4.0 are values chosen for each as starting points
for the optimizer.

.. note:: Take care setting the initial values, as in some cases, various initial points for the optimization will lead to different results.


::

    root.add('p', Paraboloid())

Then we add the paraboloid using the same syntax as before, giving it the name 'p'.

::

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

Then we connect up the outputs of the `IndepVarComps` to the parameters of the
`Paraboloid`. Notice the dotted naming convention used to refer to variables.
So, for example, `p1` represents the first `IndepVarComp` that we created to set
the value of `x` and so we connect that to parameter `x` of the `Paraboloid`.
Since the `Paraboloid` is named `p` and has a parameter
`x`, it is referred to as `p.x` in the call to the `connect` method.

Every problem has a `Driver` and for most situations, we would want to set a
`Driver` for the `Problem` using code like this

::

    top.driver = SomeDriver()

For this very simple tutorial, we do not need to set a `Driver`, we will just
use the default, built-in driver, which is
`Driver`. ( `Driver` also serves as the base class for all `Drivers`. )
`Driver` is the simplest driver possible, running a `Problem` once.

::

    top.setup()

Before we can run our model we need to do some setup. This is done using the
`setup` method on the `Problem`. This method performs all the setup of vector
storage, data transfer, etc., necessary to perform calculations. Calling
`setup` is required before running the model.

::

    top.run()

Now we can run the model using the `run` method of `Problem`.

::

    print(top['p.f_xy'])

Finally, we print the output of the `Paraboloid` Component using the
dictionary-style method of accessing variables on the problem instance.
Putting it all together:

.. testcode:: parab

    top = Problem()
    root = top.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))
    root.add('p', Paraboloid())

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

    top.setup()
    top.run()

    print(top['p.f_xy'])

The output should look like this:

.. testoutput:: parab
   :options: +ELLIPSIS

   -15.0

The `IndepVarComp` component is used to define a source for an unconnected
`param` that we want to use as an independent variable that can be declared as
a design variable for a driver. In our case, we want to optimize the
Paraboloid model, finding values for 'x' and 'y' that minimize the output
'f_xy.'

Sometimes we just want to run our component once to see the result.
Similarly, sometimes we have `params` that will be constant through our
optimization, and thus don't need to be design variables. In either of these
cases, the `IndepVarComp` is not required, and we can build our model while
leaving those parameters unconnected. All unconnected params use their default
value as the initial value. You can set the values of any unconnected params
the same way as any other variables by doing the following:

.. testcode:: parab

    top = Problem()
    root = top.root = Group()

    root.add('p', Paraboloid(), promotes=['x', 'y'])

    top.setup()

    # Set values for x and y
    top['x'] = 5.0
    top['y'] = 2.0

    top.run()

    print(top['p.f_xy'])

This can only be done after `setup` is called. Note that the promoted names
'x' and 'y' are used.

The new output should look like this:

.. testoutput:: parab
   :options: +ELLIPSIS

   47.0

Future tutorials will show more complex `Problems`.

.. _`paraboloid_optimization_tutorial`:

Optimization of the Paraboloid
------------------------------

Now that we have the paraboloid model set up, let's do a simple unconstrained
optimization. Let's find the minimum point on the Paraboloid over the
variables x and y. This requires the addition of just a few more lines.

First, we need to import the optimizer.

.. testcode:: parab

    from openmdao.api import ScipyOptimizer

The main optimizer built into OpenMDAO is a wrapper around Scipy's `minimize`
function. OpenMDAO supports 9 of the optimizers built into `minimize`. The
ones that will be most frequently used are SLSQP and COBYLA, since they are the
only two in the `minimize` package that support constraints. We will use
SLSQP because it supports OpenMDAO-supplied gradients.

.. testcode:: parab

        top = Problem()
        root = top.root = Group()

        # Initial value of x and y set in the IndepVarComp.
        root.add('p1', IndepVarComp('x', 13.0))
        root.add('p2', IndepVarComp('y', -14.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'

        top.driver.add_desvar('p1.x', lower=-50, upper=50)
        top.driver.add_desvar('p2.y', lower=-50, upper=50)
        top.driver.add_objective('p.f_xy')

        top.setup()

        # You can also specify initial values post-setup
        top['p1.x'] = 3.0
        top['p2.y'] = -4.0

        top.run()

        print('\n')
        print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))

Every driver has an `options` dictionary which contains important settings for the driver.
These settings tell `ScipyOptimizer` which optimization method to use, so here we
select 'SLSQP'. For all optimizers, you can specify a convergence tolerance
'tol' and a maximum number of iterations 'maxiter.'

Next, we select the parameters the optimizer will drive by calling
`add_param` and giving it the `IndepVarComp` unknowns that we have created. We
also set high and low bounds for this problem. It is not required to set
these (they will default to -1e99 and 1e99 respectively), but it is generally
a good idea.

Finally, we add the objective. You can use any `unknown` in your model as the
objective.

Once we have called setup on the model, we can specify the initial conditions
for the design variables just like we did with unconnected params.

Since SLSQP is a gradient based optimizer, OpenMDAO will call the `linearize` method
on the `Paraboloid` while calculating the total gradient of the objective
with respect to the two design variables. This is done automatically.

Finally, we made a change to the print statement so that we can print the
objective and the parameters. This time, we get the value by keying into the
problem instance ('top') with the full variable path to the quantities we
want to see. This is equivalent to what was shown in the first tutorial.

Putting this all together, when we run the model, we get output that looks
like this (note, the optimizer may print some things before this, depending on
settings):

.. testoutput:: parab
   :options: +ELLIPSIS

   ...
   Minimum of -27.333333 found at (6.666667, -7.333333)


Optimization of the Paraboloid with a Constraint
------------------------------------------------

Finally, let's take this optimization problem and add a constraint to it. Our
constraint takes the form of an inequality we want to satisfy: x - y >= 15.

First, we need to add one more import to the beginning of our model.

.. testcode:: parab

    from openmdao.api import ExecComp


We'll use an `ExecComp` to represent our constraint in the model. An ExecComp
is a shortcut that lets us easily create a component that defines a simple
expression for us.


.. testcode:: parab

    top = Problem()

    root = top.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))
    root.add('p', Paraboloid())

    # Constraint Equation
    root.add('con', ExecComp('c = x-y'))

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')
    root.connect('p.x', 'con.x')
    root.connect('p.y', 'con.y')

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'

    top.driver.add_desvar('p1.x', lower=-50, upper=50)
    top.driver.add_desvar('p2.y', lower=-50, upper=50)
    top.driver.add_objective('p.f_xy')
    top.driver.add_constraint('con.c', lower=15.0)

    top.setup()
    top.run()

    print('\n')
    print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))

Here, we added an ExecComp named 'con' to represent part of our
constraint inequality. Our constraint is "x - y >= 15", so we have created an
ExecComp that will evaluate the expression "x - y" and place that result into
the unknown 'con.c'. To complete the definition of the constraint, we also
need to connect our 'con' expression to 'x' and 'y' on the paraboloid.

Finally, we need to tell the driver to use the unknown "con.c" as a
constraint using the `add_constraint` method. This method takes the name of
the variable and an "upper" or "lower" bound. Here we give it a lower bound
of 15, which completes the inequality constraint "x - y >= 15".

OpenMDAO also supports the specification of double sided constraints, so if
you wanted to constrain x-y to lie on a band between 15 and 16 which is "16 > x-y > 15",
you would just do the following:

::

    top.driver.add_constraint('con.c', lower=15.0, upper=16.0)


So now, putting it all together, we can run the model and get this:

.. testoutput:: parab
   :options: +ELLIPSIS

   ...
   Minimum of -27.083333 found at (7.166667, -7.833333)

A new optimum is found because the original one was infeasible (i.e., that
design point violated the constraint equation).

.. tags:: Tutorials, Component, Paraboloid, Optimization
