""" Test objects for the sellar two discipline problem.
From Sellar's analytic problem.

    Sellar, R. S., Batill, S. M., and Renaud, J. E., "Response Surface Based, Concurrent Subspace
    Optimization for Multidisciplinary System Design," Proceedings References 79 of the 34th AIAA
    Aerospace Sciences Meeting and Exhibit, Reno, NV, January 1996.
"""

import numpy as np

from openmdao.components.exec_comp import ExecComp
from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.solvers.newton import Newton


class SellarDis1(Component):
    """Component containing Discipline 1 -- no derivatives version."""

    def __init__(self):
        super(SellarDis1, self).__init__()

        # Global Design Variable
        self.add_param('z', val=np.zeros(2))

        # Local Design Variable
        self.add_param('x', val=0.)

        # Coupling parameter
        self.add_param('y2', val=1.0)

        # Coupling output
        self.add_output('y1', val=1.0)

        self.execution_count = 0

    def solve_nonlinear(self, params, unknowns, resids):
        """Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2"""

        z1 = params['z'][0]
        z2 = params['z'][1]
        x1 = params['x']
        y2 = params['y2']

        unknowns['y1'] = z1**2 + z2 + x1 - 0.2*y2

        self.execution_count += 1


class SellarDis1withDerivatives(SellarDis1):
    """Component containing Discipline 1 -- derivatives version."""

    def linearize(self, params, unknowns, resids):
        """ Jacobian for Sellar discipline 1."""
        J = {}

        J['y1','y2'] = -0.2
        J['y1','z'] = np.array([[2.0*params['z'][0], 1.0]])
        J['y1','x'] = 1.0

        return J


class SellarDis2(Component):
    """Component containing Discipline 2 -- no derivatives version."""

    def __init__(self):
        super(SellarDis2, self).__init__()

        # Global Design Variable
        self.add_param('z', val=np.zeros(2))

        # Coupling parameter
        self.add_param('y1', val=1.0)

        # Coupling output
        self.add_output('y2', val=1.0)

        self.execution_count = 0

    def solve_nonlinear(self, params, unknowns, resids):
        """Evaluates the equation
        y2 = y1**(.5) + z1 + z2"""

        z1 = params['z'][0]
        z2 = params['z'][1]
        y1 = params['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        unknowns['y2'] = y1**.5 + z1 + z2

        self.execution_count += 1


class SellarDis2withDerivatives(SellarDis2):
    """Component containing Discipline 2 -- derivatives version."""

    def linearize(self, params, unknowns, resids):
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

        self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        cycle = self.add('cycle', Group(), promotes=['x', 'z', 'y1', 'y2'])
        cycle.ln_solver = ScipyGMRES()
        cycle.add('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        cycle.add('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0),
                 promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.nl_solver = NLGaussSeidel()
        self.cycle.d1.fd_options['force_fd'] = True
        self.cycle.d2.fd_options['force_fd'] = True


class SellarDerivatives(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    with derivatives."""

    def __init__(self):
        super(SellarDerivatives, self).__init__()

        self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        self.add('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        self.add('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0),
                 promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.nl_solver = NLGaussSeidel()
        self.ln_solver = ScipyGMRES()

class SellarDerivativesGrouped(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    without derivatives."""

    def __init__(self):
        super(SellarDerivativesGrouped, self).__init__()

        self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        mda = self.add('mda', Group(), promotes=['x', 'z', 'y1', 'y2'])
        mda.ln_solver = ScipyGMRES()
        mda.add('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                 promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        mda.nl_solver = NLGaussSeidel()
        mda.d1.fd_options['force_fd'] = True
        mda.d2.fd_options['force_fd'] = True

        self.ln_solver = ScipyGMRES()

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

    def linearize(self, params, unknowns, resids):
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

        self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub = self.add('sub', Group(), promotes=['x', 'z', 'y1', 'state_eq.y2_actual', 'state_eq.y2_command', 'd1.y2', 'd2.y2'])
        sub.ln_solver = ScipyGMRES()

        subgrp = sub.add('state_eq_group', Group(), promotes=['state_eq.y2_actual', 'state_eq.y2_command'])
        subgrp.ln_solver = ScipyGMRES()
        subgrp.add('state_eq', StateConnection())

        sub.add('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1'])
        sub.add('d2', SellarDis2withDerivatives(), promotes=['z', 'y1'])

        self.connect('state_eq.y2_command', 'd1.y2')
        self.connect('d2.y2', 'state_eq.y2_actual')

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                 promotes=['x', 'z', 'y1', 'obj'])
        self.connect('d2.y2', 'obj_cmp.y2')

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2'])
        self.connect('d2.y2', 'con_cmp2.y2')

        self.nl_solver = Newton()
