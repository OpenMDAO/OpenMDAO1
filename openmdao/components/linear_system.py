""" A component that solves a linear system. """

import numpy as np
from scipy import linalg

from openmdao.core.component import Component


class LinearSystem(Component):
    """
    A component that solves a linear system Ax=b where A and b are params
    and x is a state.

    Options
    -------
    deriv_options['type'] :  str('user')
        Derivative calculation type ('user', 'fd', 'cs')
        Default is 'user', where derivative is calculated from
        user-supplied derivatives. Set to 'fd' to finite difference
        this system. Set to 'cs' to perform the complex step
        if your components support it.
    deriv_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central)
    deriv_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    deriv_options['step_calc'] :  str('absolute')
        Set to absolute, relative
    deriv_options['check_type'] :  str('fd')
        Type of derivative check for check_partial_derivatives. Set
        to 'fd' to finite difference this system. Set to
        'cs' to perform the complex step method if
        your components support it.
    deriv_options['check_form'] :  str('forward')
        Finite difference mode: ("forward", "backward", "central")
        During check_partial_derivatives, the difference form that is used
        for the check.
    deriv_options['check_step_calc'] : str('absolute',)
        Set to 'absolute' or 'relative'. Default finite difference
        step calculation for the finite difference check in check_partial_derivatives.
    deriv_options['check_step_size'] :  float(1e-06)
        Default finite difference stepsize for the finite difference check
        in check_partial_derivatives"
    deriv_options['linearize'] : bool(False)
        Set to True if you want linearize to be called even though you are using FD.
    """

    def __init__(self, size):
        super(LinearSystem, self).__init__()
        self.size = size

        self.add_param("A", val=np.eye(size))
        self.add_param("b", val=np.ones(size))

        self.add_state("x", val=np.zeros(size))

        # cache
        self.lup = None
        self.rhs_cache = None

    def solve_nonlinear(self, params, unknowns, resids):
        """ Use numpy to solve Ax=b for x.
        """

        # lu factorization for use with solve_linear
        self.lup = linalg.lu_factor(params['A'])

        unknowns['x'] = linalg.lu_solve(self.lup, params['b'])
        resids['x'] = params['A'].dot(unknowns['x']) - params['b']

    def apply_nonlinear(self, params, unknowns, resids):
        """Evaluating residual for given state."""

        resids['x'] = params['A'].dot(unknowns['x']) - params['b']

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Apply the derivative of state variable with respect to
        everything."""

        if mode == 'fwd':

            if 'x' in dunknowns:
                dresids['x'] += params['A'].dot(dunknowns['x'])
            if 'A' in dparams:
                dresids['x'] += dparams['A'].dot(unknowns['x'])
            if 'b' in dparams:
                dresids['x'] -= dparams['b']

        elif mode == 'rev':

            if 'x' in dunknowns:
                dunknowns['x'] += params['A'].T.dot(dresids['x'])
            if 'A' in dparams:
                dparams['A'] += np.outer(unknowns['x'], dresids['x']).T
            if 'b' in dparams:
                dparams['b'] -= dresids['x']

    def solve_linear(self, dumat, drmat, vois, mode=None):
        """ LU backsubstitution to solve the derivatives of the linear system."""

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t=0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t=1

        if self.rhs_cache is None:
            self.rhs_cache = np.zeros((self.size, ))
        rhs = self.rhs_cache

        for voi in vois:
            rhs[:] = rhs_vec[voi]['x']

            sol = linalg.lu_solve(self.lup, rhs, trans=t)

            sol_vec[voi]['x'] = sol[:]
