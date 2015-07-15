.. index:: Finite Difference Tutorial

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

        from openmdao.components.paramcomp import ParamComp
        from openmdao.core.component import Component
        from openmdao.core.group import Group
        from openmdao.core.problem import Problem


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

                        self.add('px', ParamComp('x', 2.0))

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

::

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

::

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

.. testcode:: fd_example

        class Model(Group):
            """ Simple model to experiment with finite difference."""

            def __init__(self):
                super(Model, self).__init__()

                self.add('px', ParamComp('x', 2.0))

                self.add('comp1', SimpleComp())
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


Finite Difference on an Entire Model
====================================
