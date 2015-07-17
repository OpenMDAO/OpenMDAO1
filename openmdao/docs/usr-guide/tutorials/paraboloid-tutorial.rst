Paraboloid Tutorial
-------------------

This tutorial will show you how to set up a simple optimization of a paraboloid.
You'll create a paraboloid `Component` (with analytic derivatives), then put it
into a `Problem` and set up an optimizer `Driver` to minimize an objective function.

Here is the code that defines the paraboloid and then runs it. You can copy
this code into a file, and run it directly.

.. testcode:: parab

    from __future__ import print_function

    from openmdao.components.paramcomp import ParamComp
    from openmdao.core.component import Component
    from openmdao.core.problem import Problem, Group

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

            J['f_xy', 'x'] = 2.0*x - 6.0 + y
            J['f_xy', 'y'] = 2.0*y + 8.0 + x
            return J

    if __name__ == "__main__":

        top = Problem()

        root = top.root = Group()

        root.add('p1', ParamComp('x', 3.0))
        root.add('p2', ParamComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.setup()
        top.run()

        print(root.p.unknowns['f_xy'])


Now we will go through each section and explain how this code works.

Building the component
=========================

::

    from __future__ import print_function

    from openmdao.components.paramcomp import ParamComp
    from openmdao.core.component import Component
    from openmdao.core.problem import Problem, Group

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

        def jacobian(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy','x'] = 2.0*x - 6.0 + y
            J['f_xy','y'] = 2.0*y + 8.0 + x
            return J

The `jacobian` method is used to compute analytic partial derivatives of the
`unknowns` with respect to `params` (partial derivatives in OpenMDAO context refer to
derivatives for a single component by itself). The returned value, in this case `J`,
should be a dictionary whose keys are tuples of the form (‘unknown’, ‘param’) and
whose values are n-d arrays or scalars. Just like for `solve_nonlinear`, the values for the
parameters are accessed using dictionary arguments to the function.

The definition of the Paraboloid Component class is now complete. We will now
make use of this class to run a model.

Setting up the model
=========================

::

    if __name__ == "__main__":

        top = Problem()
        root = top.root = Group()

An instance of an OpenMDAO `Problem` is always the top object for running a
model. Each `Problem` in OpenMDAO must contain a root `Group`. A `Group` is a
`System` that contains other `Components` or `Groups`.

This code instantiates a `Problem` object and sets the root to be an empty `Group`.

::

    root.add('p1', ParamComp('x', 3.0))
    root.add('p2', ParamComp('y', -4.0))

Now it is time to add components to the empty group. They must be
added in execution order, so we need to start with the parameters. `ParamComp`
is a `Component` that provides the source for a variable which we can later give
to a `Driver` as a design variable to control.

We created two `ParamComps` (one for each param on the `Paraboloid`
component), gave them names, and added them to the root `Group`. The `add`
method takes a name as the first argument, and a `Component` instance as the
second argument.

::

    root.add('p', Paraboloid())

Then we add the paraboloid using the same syntax as before, giving it the name 'p'.

::

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

Then we connect up the outputs of the `ParamComps` to the parameters of the
`Paraboloid`. Notice the dotted naming convention used to refer to variables.
So, for example, `p1` represents the first `ParamComp` that we created to set
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
storage, data transfer, etc.., necessary to perform calculations. Calling
`setup` is required before running the model.

::

    top.run()

Now we can run the model using the `run` method of `Problem`.

::

    print(root.p.unknowns['f_xy'])

Finally, we print the output of the `Paraboloid` Component using the
dictionary-style method of accessing the outputs from a `Component` instance.
Putting it all together:

.. testcode:: parab

    top = Problem()
    root = top.root = Group()

    root.add('p1', ParamComp('x', 3.0))
    root.add('p2', ParamComp('y', -4.0))
    root.add('p', Paraboloid())

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

    top.setup()
    top.run()

    print(root.p.unknowns['f_xy'])

The output should look like this:

.. testoutput:: parab
   :options: +ELLIPSIS

   -15.0

Future tutorials will show more complex `Problems`.


Optimization of the Paraboloid
==============================

Now that we have the paraboloid model set up, let's do a simple unconstrained
optimization. Let's find the minimum point on the Paraboloid over the
variables x and y. This requires the addition of just a few more lines.

First, we need to import the optimizer.

.. testcode:: parab

    from openmdao.drivers.scipy_optimizer import ScipyOptimizer

The main optimizer built into OpenMDAO is a wrapper around Scipy's `minimize`
function. OpenMDAO supports 9 of the optimizers built into `minimize`. The
ones that will be most frequently used are SLSQP and COBYLA, since they are the
only two in the `minimize` package that support constraints. We will use
SLSQP because it supports OpenMDAO-supplied gradients.

.. testcode:: parab

        top = Problem()

        root = top.root = Group()

        root.add('p1', ParamComp('x', 3.0))
        root.add('p2', ParamComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'

        top.driver.add_param('p1.x', low=-50, high=50)
        top.driver.add_param('p2.y', low=-50, high=50)
        top.driver.add_objective('p.f_xy')

        top.setup()
        top.run()

        print('\n')
        print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))

Every driver has an `options` dictionary which contains important settings for the driver.
These settings tell `ScipyOptimizer` which optimization method to use, so here we
select 'SLSQP'. For all optimizers, you can specify a convergence tolerance
'tol' and a maximum number of iterations 'maxiter.'

Next, we select the parameters the optimizer will drive by calling
`add_param` and giving it the `ParamComp` unknowns that we have created. We
also set a high and low bounds for this problem. It is not required to set
these (they will default to 1e99 and 1e99 respectively), but it is generally
a good idea.

Finally, we add the objective. You can use any `unknown` in your model as the
objective.

Since SLSQP is a gradient optimizer, OpenMDAO will call the `jacobian` method
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
================================================

Finally, let's take this optimization problem and add a constraint to it. Our
constraint takes the form of an inequality we want to satisfy: x - y > 15.

First, we need to add one more import to the beginning of our model.

.. testcode:: parab

    from openmdao.components.execcomp import ExecComp

In OpenMDAO, we cannot (yet) implement an inequality, so we need to turn the
constraint equation into an equality. With a little rearrangement, x - y > 15
becomes c = 15 - x + y. When c is less than 0, the original inequality is
satisfied. Likewise, when c is greater than zero, the inequality is violated.
Optimizers in OpenMDAO use this convention to evaluate an inequality
constraint that points to an `unknown` in your model.

.. testcode:: parab

    top = Problem()

    root = top.root = Group()

    root.add('p1', ParamComp('x', 3.0))
    root.add('p2', ParamComp('y', -4.0))
    root.add('p', Paraboloid())

    # Constraint Equation
    root.add('con', ExecComp('c = 15.0 - x + y'))

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')
    root.connect('p.x', 'con.x')
    root.connect('p.y', 'con.y')

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'

    top.driver.add_param('p1.x', low=-50, high=50)
    top.driver.add_param('p2.y', low=-50, high=50)
    top.driver.add_objective('p.f_xy')
    top.driver.add_constraint('con.c')

    top.setup()
    top.run()

    print('\n')
    print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))

Here, we added a component named 'con' to represent our constraint equation.
We use a new component called `ExecComp`. This utility component takes an equation
string expression as input, and parses that string to create a component with
the specified inputs and outputs. So for our expression here, 'con' is
created with inputs 'x' and 'y' and output 'c'. The `solve_nonlinear` and
`jacobian` functions are implemented based on the equation.

We also need to connect our 'con' expression to 'x' and 'y' on the
paraboloid. Finally, we call add_constraint on the driver, giving it the
output from the constraint component, which is 'con.c'. The default
behavior for `add_constraint` is to add a nonlinear constraint like the one
in our problem. You can also add a linear constraint, provided that your
optimizer supports it (SLSQP does), by setting the ctype call attribute to
'lin'.


So now, putting it all together, we can run the model and get this:

.. testoutput:: parab
   :options: +ELLIPSIS

   ...
   Minimum of -27.083333 found at (7.166667, -7.833333)

A new optimum is found because the original one was infeasible (i.e., that
design point violated the constraint equation.)
