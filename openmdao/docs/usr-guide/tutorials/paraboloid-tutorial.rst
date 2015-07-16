Paraboloid Tutorial
-------------------

This tutorial will take you through code needed to create a Component class
based on the equation of a paraboloid and run it.

Here is the code that defines this Component and then runs it.

.. testcode:: parab

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

        print root.p.unknowns['f_xy']


Now we will go through each section and explain how this code works.

::

    from openmdao.components.paramcomp import ParamComp
    from openmdao.core.component import Component
    from openmdao.core.problem import Problem, Group

We need to import some OpenMDAO classes.

::

    class Paraboloid(Component):

OpenMDAO provides a base class `Component` system. We will use that to create
the `Paraboloid` component. `Components` can declare variables and they
operate on their parameters to produce unknowns, which can be explicit
outputs or implicit states. For the `Paraboloid` Component, we will only be
using explicit outputs.

::

        def __init__(self):
            super(Paraboloid, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', val=0.0)


This code defines the input parameters of the Component, `x` and `y`, and
initializes them to 0.0. These could be design variables which could used to
minimize the output when doing optimization, for example.

It also defines the explicit output, `f_xy` and initializes it to 0.0.

::

        def solve_nonlinear(self, params, unknowns, resids):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
            Optimal solution (minimum): x = 6.6667; y = -7.3333
            """

            x = params['x']
            y = params['y']

            unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

The `solve_nonlinear` method is responsible for calculating outputs for a
given set of parameters. The parameters are given in the `params` variable
that is passed in to this method. You can access the values of the parameters
in this variable as if it was a Python dictionary.

Similarly, the outputs are assigned values using the `unknowns` variable that
is passed in. The output variables are accessed as if `unknowns` was a
dictionary.

::

        def jacobian(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy','x'] = 2.0*x - 6.0 + y
            J['f_xy','y'] = 2.0*y + 8.0 + x
            return J

The `jacobian` method is used to compute analytic values for Jacobian of this
Component. The returned value, in this case `J`, should be a dictionary whose
keys are tuples of the form (‘unknown’, ‘param’) and whose values are
ndarrays or scalars. Just like for `solve_nonlinear`, the values for the
parameters are accessed using dictionary style addressing.

The definition of the Paraboloid Component class is now complete. We will now
make use of this class to run a model.

::

    if __name__ == "__main__":

        top = Problem()
        root = top.root = Group()

An instance of an OpenMDAO `Problem` is always the top object for running an
model. Each `Problem` in OpenMDAO must contain a root `Group`. A `Group` is a
`System` that contains other `Components` or `Groups`.

This code instantiates a `Problem` object, sets the root to be an empty `Group`.

::

    root.add('p1', ParamComp('x', 3.0))
    root.add('p2', ParamComp('y', -4.0))

Now it is time to add components to the empty group. At present, they must be
aded in execution order, so we need to start with the parameters. `ParamComp`
is a `Component` that provides the source for a variable which we can assign
as a parameter for a driver.

So here, we created two `ParamComps`, one for each param on the `Paraboloid`
component, and gave them names and added them to the root `Group`. The `add`
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
the value of `x` and so we connect that to parameter `x` of the `Paraboloid`,
which is named `x`. Since the `Paraboloid` is named `p` and has a parameter
`x`, it is referred to as `p.x` in the call to the `connect` method.

Every problem has a `Driver` and for most situations, we would want to set a
`Driver` for the `Problem` using code like this

::

    top.driver = SomeDriver()

For this very simple tutorial, we will just use the default which is
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

    print root.p.unknowns['f_xy']

Finally, we print the output of the `Paraboloid` Component using the
dictionary-style method of accessing the outputs. Putting it all together:

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

    print root.p.unknowns['f_xy']

The output should look like this:

.. testoutput:: parab
   :options: +ELLIPSIS

   -15.0

Future tutorials will show more complex `Problems`.


Optimization with the Paraboloid
================================

Now that we have the paraboloid model set up, let's do a simple unconstrained optimization.