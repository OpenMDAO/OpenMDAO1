
.. index:: MDAO tutorial problem


The Sellar Problem
==================

This tutorial illustrates how to set up a coupled disciplinary problem in
OpenMDAO and prepare it for optimization. This tutorial builds on what you
have learned in previous tutorials about defining components with analytic
derivatives. The problem we will show you is the familiar Sellar Problem,
which consists of two disciplines as follows:


.. figure:: SellarResized.png
   :align: center
   :alt: Equations showing the two disciplines for the Sellar problem

Variables *z1, z2,* and *x1* are the design variables over which we'd like to minimize
the objective. Both disciplines are functions of *z1* and *z2,* so they are called the
*global* design variables, while only the first discipline is a function of *x1,* so it
is called the *local* design variable. The two disciplines are coupled by the
coupling variables *y1* and *y2.* Discipline 1 takes *y2* as an input, and computes *y1* as
an output, while Discipline 2 takes *y1* as an input and computes *y2* as an output. As
such, the two disciplines depend on each other's output, so iteration is required to
find a set of coupling variables that satisfies both equations.

First, disciplines 1 and 2 were implemented in OpenMDAO as components.

.. testcode:: Disciplines

        import numpy as np

        from openmdao.core.component import Component


        class SellarDis1(Component):
            """Component containing Discipline 1."""

            def __init__(self):
                super(SellarDis1, self).__init__()

                # Global Design Variable
                self.add_param('z', val=np.zeros(2))

                # Local Design Variable
                self.add_param('x', val=0.)

                # Coupling parameter
                self.add_param('y2', val=0.)

                # Coupling output
                self.add_output('y1', val=1.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """Evaluates the equation
                y1 = z1**2 + z2 + x1 - 0.2*y2"""

                z1 = params['z'][0]
                z2 = params['z'][1]
                x1 = params['x']
                y2 = params['y2']

                unknowns['y1'] = z1**2 + z2 + x1 - 0.2*y2

            def jacobian(self, params, unknowns, resids):
                """ Jacobian for Sellar discipline 1."""
                J = {}

                J['y1','y2'] = -0.2
                J['y1','z'] = np.array([[2*params['z'][0], 1.0]])
                J['y1','x'] = 1.0

                return J


        class SellarDis2(Component):
            """Component containing Discipline 2."""

            def __init__(self):
                super(SellarDis2, self).__init__()

                # Global Design Variable
                self.add_param('z', val=np.zeros(2))

                # Coupling parameter
                self.add_param('y1', val=0.)

                # Coupling output
                self.add_output('y2', val=1.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """Evaluates the equation
                y2 = y1**(.5) + z1 + z2"""

                z1 = params['z'][0]
                z2 = params['z'][1]
                y1 = params['y1']

                # Note: this may cause some issues. However, y1 is constrained to be
                # above 3.16, so lets just let it converge, and the optimizer will
                # throw it out
                y1 = abs(y1)

                unknowns['y2'] = y1**.5 + z1 + z2

            def jacobian(self, params, unknowns, resids):
                """ Jacobian for Sellar discipline 2."""
                J = {}

                J['y2', 'y1'] = .5*params['y1']**-.5
                J['y2', 'z'] = np.array([[1.0, 1.0]])

                return J

In building these discplines, we gave default values to all of the `params`
and `unknowns` so that OpenMDAO can allocate the correct size in the vectors.
The global design variables `z1` and `z1` were combined into a 2-element `ndarray`.

``Discipline2`` contains a square root of variable *y1* in its calculation. For negative values
of *y1,* the result would be imaginary, so the absolute value is taken before the square root
is applied. This component is clearly not valid for ``y1 < 0``, but some solvers could
occasionally force *y1* to go slightly negative while trying to converge the two disciplines . The inclusion
of the absolute value solves the problem without impacting the final converged solution.

Now that you have defined the components for the Sellar Problem for yourself, let's take a momement to
consider what we have really accomplished. Firstly, we have written two (very simple) analysis components.
If you were working on a real problem, these would likely come in the form of some much more complex tools
that you wrapped in the framework. But keep in mind that from an optimization point of view, whether they
are simple tools or wrappers for real analyses, OpenMDAO still views them as components with `params`, `unknowns`,
a `solve_nonlinear` function, and optionally a `jacobian` function.

We have talked about the problem formulation and specified that certain variables will be
design variables, while others are coupling variables. But none of the code we have written has told
OpenMDAO about those details. That's what we'll get to next!

**Reference:**

Sellar, R. S., Batill, S. M., and Renaud, J. E., "Response Surface Based,
Concurrent Subspace Optimization for Multidisciplinary System Design,"
*Proceedings References 79 of the 34th AIAA Aerospace Sciences Meeting and
Exhibit,* Reno, NV, January 1996.



Setting up the Optimization Problem
===================================

.. testcode:: Disciplines

    from openmdao.components.execcomp import ExecComp
    from openmdao.components.paramcomp import ParamComp
    from openmdao.core.group import Group
    from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel

    class SellarDerivatives(Group):
        """ Group containing the Sellar MDA. This version uses the disciplines
        with derivatives."""

        def __init__(self):
            super(SellarDerivatives, self).__init__()

            self.add('px', ParamComp('x', 1.0), promotes=['*'])
            self.add('pz', ParamComp('z', np.array([5.0, 2.0])), promotes=['*'])

            self.add('d1', SellarDis1(), promotes=['*'])
            self.add('d2', SellarDis2(), promotes=['*'])

            self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                         z=np.array([0.0, 0.0]), x=0.0, d1=0.0, d2=0.0),
                     promotes=['*'])

            self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['*'])
            self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['*'])

            self.nl_solver = NLGaussSeidel()
            self.nl_solver.options['atol'] = 1.0e-12



.. testcode:: Disciplines

        top = Problem()
        top.root = SellarDerivatives()

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['tol'] = 1.0e-8

        top.driver.add_param('z', low=np.array([-10.0, 0.0]),
                             high=np.array([10.0, 10.0]))
        top.driver.add_param('x', low=0.0, high=10.0)

        top.driver.add_objective('obj')
        top.driver.add_constraint('con1')
        top.driver.add_constraint('con2')

        top.setup()
        top.run()

        assert_rel_error(self, top['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, top['z'][1], 0.0, 1e-3)
        assert_rel_error(self, top['x'], 0.0, 1e-3)