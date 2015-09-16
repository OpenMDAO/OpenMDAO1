.. index:: Finite Difference Tutorial

Finite Difference
-----------------------

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

        from openmdao.components import IndepVarComp
        from openmdao.core import Component, Group, Problem


        class SimpleComp(Component):
            """ A simple component that provides derivatives. """

            def __init__(self):
                super(SimpleComp, self).__init__()

                # Params
                self.add_desvar('x', 2.0)

                # Unknowns
                self.add_output('y', 0.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much.  Just multiply by 3"""
                unknowns['y'] = 3.0*params['x']
                print('Execute', self.name)

            def jacobian(self, params, unknowns, resids):
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


The output shows that comp2 and comp4 aren't using their `jacobian` function,
but instead are executing twice, as would be expected when using central
difference.


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

Here we see that, instead of calling 'jacobian', comp2 and comp3 execute
during finite differnce of the group that owns them. This is as we expect.

Finite Difference on an Entire Model
====================================

Finally, let's finite difference the whole model in one operation. We tell
OpenMDAO to do this by setting force_fd in the parent `Group`.

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

So here, `jacobian` is never called in any component as the finite difference
just executes the components in sequence. This is also as expected.
