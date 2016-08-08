.. _`implicit_comp_tutorial`:

Components with Implicit States
===============================

In this tutorial, we show how to build a component that contains an implicit
function. So far, we have learned how to define an OpenMDAO component that
represents an explicit function of unknowns with respect to its params.

::

   y = F(x)

OpenMDAO also allows us to define a component that contains an implicit
function of the same variables:

::

   R(x, y) = 0

Here, the variable 'x' is a known param that is passed in to the component, but
the variable 'y', called the state, is an unknown that needs to be solved to
satisfy the equation. The left-hand side of the implicit equation is called
the residual. Since an implicit function may not have a closed-form solution,
the state is typically determined by numerically solving the residual
equation, or in other words, iterating on the state until the residual is
driven to zero.

Some equations can easily be converted from implicit to explicit, but there
are cases that are difficult or impossible to represent in an explicit form,
so for that reason we support implicit equations.

In the following tutorial, we will build a component to solve the following
equations:

::

   f(x,z) = xz + z - 4 = 0
   y = x + 2z

The first equation is an implicit function of the param 'z' and the state
'x'. This example is fairly easy to convert to explicit, and you can do this
if you would like a direct comparison between implicit and explicit
components. The second equation is a normal explicit one. We will include
this in the same component to show how an OpenMDAO component can contain any
number of implicit and explicit relationships.

There are 3 ways to solve for the state in an implicit component.

1. The component can solve it (and OpenMDAO does nothing).
2. OpenMDAO can solve it (and the component does nothing).
3. Both OpenMDAO and the component can solve it.

Most of the time, you will be using '2' or '3', mainly because if you
implement '1', there is no reason that OpenMDAO needs to know about the
state, so it might as well be an output.

We will show you how to set up and run each of these.

Implicit Component that Solves Itself
-------------------------------------

So let's implement a component with param 'x', output 'y', and state 'z'.

.. testcode:: Implicit

        from __future__ import print_function
        import numpy as np

        from openmdao.api import Component, Group, Problem, ScipyGMRES

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', 0.5)

                # Unknowns
                self.add_output('y', 0.0)

                # States
                self.add_state('z', 0.0)

                self.maxiter = 25
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                """ Simple iterative solve. (Babylonian method)."""

                x = params['x']
                z = unknowns['z']
                znew = z

                itercount = 0
                eps = 1.0e99
                while itercount < self.maxiter and abs(eps) > self.atol:
                    z = znew
                    znew = 4.0 - x*z

                    eps = x*znew + znew - 4.0
                    itercount += 1

                # Our State
                unknowns['z'] = znew

                # Our Output
                unknowns['y'] = x + 2.0*znew

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = x*z + z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x + 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.array([1.0])
                J[('y', 'z')] = np.array([2.0])

                # State equation
                J[('z', 'z')] = np.array([params['x'] + 1.0])
                J[('z', 'x')] = np.array([unknowns['z']])

                return J

Since we are solving the implicit equation in our component, we include code
in `solve_nonlinear` that iterates on the implicit equation using the
Babylonian method, which is essentially fixed point iteration. The
'solve_nonlinear' method must also set values for the unknowns, in this case
'y'.

When you have explicit equations, you must also specify a new method called
'apply_nonlinear'. This method is called when OpenMDAO wants to evaluate the
residuals, so in our case we want to return the current value of:

::

    R = xz + z - 4

This is an in-place evaluation using current values of all states, params,
and unknowns as they appear in the vectors. You should never set anything in
this method except residuals.

The value of a residual is placed in the `resids` vector in the variable
named for the appropriate state. In our case, the residual for state 'x' is
placed in `resids['x']`.

If your component has unknowns, then there is one less obvious thing you need
to do. For each unknown, you need to define a residual. This is done by
rearranging the equation so that it is in implicit form.

::

    y = x + 2.0*z
    R = (x + 2.0*z) - y

By convention, OpenMDAO expects the current value of the output to be
subtracted as shown. Note that the residuals are important so that this
component can be correctly converged by solvers in the containing group; they
don't impact the self-solve case.

Finally, we show how to declare derivatives for the implicit comp. The
derivatives for the output equation are as expected. For the implicit
equation, we evaluate the derivatives of the state equation with respect to
all inputs and states. All derivatives are assigned to the state output in
the Jacobian, so the derivative of the residual with respect to the state
resides in the ('z', 'z') key.

Now, let's put the implicit component into a simple model and run it.

.. testcode:: Implicit

    top = Problem()
    root = top.root = Group()
    root.add('comp', SimpleImplicitComp())

    root.ln_solver = ScipyGMRES()
    top.setup()

    top.run()

    print('Solution: x = %f, z = %f, y = %f' % (top['comp.x'], top['comp.z'], top['comp.y']))

Note that we need to specify ScipyGMRES as our linear solver as we need one
that can handle implicit states. We aren't actually calculating any
derivatives here, but if we wanted to, for example, place this in a larger
model and optimize it, GMRES would be needed here so we add it.

.. testoutput:: Implicit
   :options: +ELLIPSIS

   Solution: x = 0.500000, z = 2.666667, y = 5.833333

This matches the expected answer.

Implicit Component that is Solved by OpenMDAO
---------------------------------------------

Coming soon.

Implicit Component that is Solved by both itself and OpenMDAO
-------------------------------------------------------------

Coming soon.

.. testcode:: Implicit

        from __future__ import print_function
        import numpy as np

        from openmdao.api import Component, Group, Problem, ScipyGMRES

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', 0.5)

                # Unknowns
                self.add_output('y', 0.0)

                # States
                self.add_state('z', 0.0)

                self.maxiter = 25
                self.atol = 1.0e-3

            def solve_nonlinear(self, params, unknowns, resids):
                """ Simple iterative solve. (Babylonian method)."""

                x = params['x']
                z = unknowns['z']
                znew = z

                itercount = 0
                eps = 1.0e99
                while itercount < self.maxiter and abs(eps) > self.atol:
                    z = znew
                    znew = 4.0 - x*z

                    eps = x*znew + znew - 4.0
                    itercount += 1

                # Our State
                unknowns['z'] = znew

                # Our Output
                unknowns['y'] = x + 2.0*znew

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = x*z + z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x + 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.array([1.0])
                J[('y', 'z')] = np.array([2.0])

                # State equation
                J[('z', 'z')] = np.array([params['x'] + 1.0])
                J[('z', 'x')] = np.array([unknowns['z']])

                return J

.. tags:: Tutorials, Component 
