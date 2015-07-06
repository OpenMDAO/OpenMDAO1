.. _OpenMDAO-Examples:

============
Examples
============

Now that you have some sense of how OpenMDAO's pieces relate to one another,
let's take a look a couple of concrete examples at work, Paraboloid, and the
Sellar Problem.

[This document is walking through these classes without any reference to the past
OpenMDAO syntax/structure.]

Paraboloid
----------

::

    """ paraboloid.py - Evaluates the equation (x-3)^2 + xy + (y+4)^2 = 3
    """
    from openmdao.core.component import Component


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

            J['f_xy','x'] = 2.0*x - 6.0 + y
            J['f_xy','y'] = 2.0*y + 8.0 + x
            return J


Sellar
------

[Sellar intro taken from existing docs, will need to be edited based on new framework/
new code.]
We will cover some of the more advanced capabilities of OpenMDAO. You should
read and understand Simple Optimization and MetaModel before starting this one.

This tutorial illustrates the features of OpenMDAO that support the use of
decomposition-based MDAO architectures, such as:

Multidisciplinary Design Feasible (MDF)
Independent Design Feasible (IDF)
Collaborative Optimization (CO)

First we’ll walk through the manual implementation of these architectures on a simple
  example problem. This will introduce you to using iteration hierarchy, metamodeling,
  and Design of Experiments (DOE) to construct different kinds of optimization processes.
  Understanding this section is important if you want to implement a new MDAO architecture
  or an existing one that is not currently available within OpenMDAO.

Once you understand how to construct an MDAO architecture by hand, you’ll see
that it can take a good amount of work to set up. That’s why we’ll show you how
to set up your problem so you can automatically apply a number of different MDAO
architectures. Using the automatic implementation of an architecture will dramatically
simplify your input files.

All of these tutorials use the Sellar Problem, which consists of two disciplines as follows:

[insert sellar graphic here after moving it to the _static dir]

Variables z1, z2, and x1 are the design variables over which we’d like to minimize
  the objective. Both disciplines are functions of z1 and z2, so they are called
  the global design variables, while only the first discipline is a function of x1,
  so it is called the local design variable. The two disciplines are coupled by the
  coupling variables y1 and y2. Discipline 1 takes y2 as an input, and computes y1
  as an output, while Discipline 2 takes y1 as an input and computes y2 as an output.
  As such, the two disciplines depend on each other’s output, so iteration is required
  to find a set of coupling variables that satisfies both equations.

Disciplines 1 and 2 were implemented in OpenMDAO as components.
[note, old intro, but the code that follows is 1.0 code]

::


    """ Test objects for the sellar two discipline problem.
    From Sellar's analytic problem.

        Sellar, R. S., Batill, S. M., and Renaud, J. E., "Response Surface Based, Concurrent Subspace
        Optimization for Multidisciplinary System Design," Proceedings References 79 of the 34th AIAA
        Aerospace Sciences Meeting and Exhibit, Reno, NV, January 1996.
    """

    import numpy as np

    from openmdao.components.execcomp import ExecComp
    from openmdao.components.paramcomp import ParamComp
    from openmdao.core.component import Component
    from openmdao.core.group import Group
    from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel


    class SellarDis1(Component):
        """Component containing Discipline 1 -- no derivatives version."""

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


    class SellarDis1withDerivatives(SellarDis1):
        """Component containing Discipline 1 -- derivatives version."""

        def jacobian(self, params, unknowns, resids):
            """ Jacobian for Sellar discipline 1."""
            J = {}

            J['y1','y2'] = -0.2
            J['y1','z'] = np.array([[2*params['z'][0], 1.0]])
            J['y1','x'] = 1.0

            return J


    class SellarDis2(Component):
        """Component containing Discipline 2 -- no derivatives version."""

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


    class SellarDis2withDerivatives(SellarDis2):
        """Component containing Discipline 2 -- derivatives version."""

        def jacobian(self, params, unknowns, resids):
            """ Jacobian for Sellar discipline 2."""
            J = {}

            J['y2', 'y1'] = .5*params['y1']**-.5
            J['y2', 'z'] = np.array([[1.0, 1.0]])

            return J


    class SellarNoDerivatives(Group):
        """ Group containing the Sellar MDA. This version uses the disciplines
        without derivatives."""

        def __init__(self):
            super(SellarNoDerivatives, self).__init__()

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
            self.d1.fd_options['force_fd'] = True
            self.d2.fd_options['force_fd'] = True


    class SellarDerivatives(Group):
        """ Group containing the Sellar MDA. This version uses the disciplines
        with derivatives."""

        def __init__(self):
            super(SellarDerivatives, self).__init__()

            self.add('px', ParamComp('x', 1.0), promotes=['*'])
            self.add('pz', ParamComp('z', np.array([5.0, 2.0])), promotes=['*'])

            self.add('d1', SellarDis1withDerivatives(), promotes=['*'])
            self.add('d2', SellarDis2withDerivatives(), promotes=['*'])

            self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                         z=np.array([0.0, 0.0]), x=0.0, d1=0.0, d2=0.0),
                     promotes=['*'])

            self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['*'])
            self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['*'])

            self.nl_solver = NLGaussSeidel()


    class SellarDerivativesGrouped(Group):
        """ Group containing the Sellar MDA. This version uses the disciplines
        without derivatives."""

        def __init__(self):
            super(SellarDerivativesGrouped, self).__init__()

            self.add('px', ParamComp('x', 1.0), promotes=['*'])
            self.add('pz', ParamComp('z', np.array([5.0, 2.0])), promotes=['*'])
            sub = self.add('mda', Group(), promotes=['*'])

            sub.add('d1', SellarDis1withDerivatives(), promotes=['*'])
            sub.add('d2', SellarDis2withDerivatives(), promotes=['*'])

            self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                         z=np.array([0.0, 0.0]), x=0.0, d1=0.0, d2=0.0),
                     promotes=['*'])

            self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['*'])
            self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['*'])

            sub.nl_solver = NLGaussSeidel()
            sub.d1.fd_options['force_fd'] = True
            sub.d2.fd_options['force_fd'] = True


    class StateConnection(Component):
        """ Define connection with an explicit equation"""

        def __init__(self):
            super(StateConnection, self).__init__()

            # Inputs
            self.add_param('y2_actual', 1.0)

            # States
            self.add_state('y2_command', val=1.0)

        def apply_nonlinear(self, params, unknowns, resids):
            """ Don't solve; just calculate the residual."""

            y2_actual = params['y2_actual']
            y2_command = unknowns['y2_command']

            resids['y2_command'] = y2_actual - y2_command

        def solve_nonlinear(self, params, unknowns, resids):
            """ This is a dummy comp that doesn't modify its state."""
            pass

        def jacobian(self, params, unknowns, resids):
            """Analytical derivatives."""

            J = {}

            # State equation
            J[('y2_command', 'y2_command')] = -1.0
            J[('y2_command', 'y2_actual')] = 1.0

            return J

    class SellarStateConnection(Group):
        """ Group containing the Sellar MDA. This version uses the disciplines
        with derivatives."""

        def __init__(self):
            super(SellarStateConnection, self).__init__()

            self.add('px', ParamComp('x', 1.0), promotes=['*'])
            self.add('pz', ParamComp('z', np.array([5.0, 2.0])), promotes=['*'])

            self.add('state_eq', StateConnection())
            self.add('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1'])
            self.add('d2', SellarDis2withDerivatives(), promotes=['z', 'y1'])

            self.connect('state_eq.y2_command', 'd1.y2')
            self.connect('d2.y2', 'state_eq.y2_actual')

            self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                         z=np.array([0.0, 0.0]), x=0.0, d1=0.0, d2=0.0),
                     promotes=['x', 'z', 'y1'])
            self.connect('d2.y2', 'obj_cmp.y2')

            self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['*'])
            self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2'])
            self.connect('d2.y2', 'con_cmp2.y2')

            self.nl_solver = NLGaussSeidel()
