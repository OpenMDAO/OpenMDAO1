""" Brent Nonlinear solver."""

from six import iteritems

from math import isnan

import numpy as np
from scipy.optimize import brentq

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.util.record_util import update_local_meta, create_local_meta


class Brent(NonLinearSolver):
    """Root finding using Brent's method. This is a specialized solver that
    can only converge a single scalar residual. You must specify the name
    of the state-variable/residual via the `state_var` option. You must also
    give either `lower_bound` and `upper_bound` or `var_lower_bound`
    and `var_upper_bound`.


    Options
    -------
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout each iteration, set to 2 to print subiteration residuals as well.
    options['maxiter'] :  int(100)
        if convergence is not achieved in maxiter iterations, and error is raised. Must be >= 0.
    options['rtol'] :  float64(4.4408920985e-16)
        The routine converges when a root is known to lie within rtol times the value returned of the value returned. Should be >= 0. Defaults to np.finfo(float).eps * 2.
    options['state_var'] :  str('')
        name of the state-variable/residual the solver should with
    options['state_var_idx'] :  int(0)
        Index into state_var if it is a vector
    options['upper_bound'] :  float(100.0)
        upper bound for the root search
    options['lower_bound'] :  float(0.0)
        lower bound for the root search
    options['var_lower_bound'] :  str('')
        if given, name of the variable to pull the lower bound value from.This variable must be a parameter on of of the child components of the containing system
    options['var_upper_bound'] :  str('')
        if given, name of the variable to pull the upper bound value from.This variable must be a parameter on of of the child components of the containing system
    options['xtol'] :  int(0)
        The routine converges when a root is known to lie within xtol of the value return. Should be >= 0. The routine modifies this to take into account the relative precision of doubles.
    """

    def __init__(self):
        super(Brent, self).__init__()

        opt = self.options
        opt.add_option('xtol', 0,
            desc='The routine converges when a root is known to lie within xtol of the value return. Should be >= 0. '
                 'The routine modifies this to take into account the relative precision of doubles.')

        opt.add_option('rtol', np.finfo(float).eps * 2.,
            desc='The routine converges when a root is known to lie within rtol times the value returned of '
                 'the value returned. Should be >= 0. Defaults to np.finfo(float).eps * 2.')

        opt.add_option('maxiter', 100,
            desc='if convergence is not achieved in maxiter iterations, and error is raised. Must be >= 0.')

        opt.add_option('state_var', '', desc="name of the state-variable/residual the solver should with")
        opt.add_option('state_var_idx', 0, desc="Index into state_var if it is a vector.")

        opt.add_option('lower_bound', 0., desc="lower bound for the root search")
        opt.add_option('upper_bound', 100., desc="upper bound for the root search")

        opt.add_option('var_lower_bound', '', desc='if given, name of the variable to pull the lower bound value from.'
            'This variable must be a parameter on of of the child components of the containing system')
        opt.add_option('var_upper_bound', '', desc='if given, name of the variable to pull the upper bound value from.'
            'This variable must be a parameter on of of the child components of the containing system')

        # we renamed max_iter to maxiter to match all the other solvers
        opt._add_deprecation('max_iter', 'maxiter')

        self.xstar = None

        self.print_name = 'BRENT'
        self.basenorm = 0.0

    def setup(self, sub):
        """ Initialization

        Args
        ----
        sub: `System`
            System that owns this solver.
        """

        if self.options['state_var'].strip() == '':
            pathname = 'root' if sub.pathname=='' else sub.pathname
            msg = "'state_var' option in Brent solver of %s must be specified" % pathname
            raise ValueError(msg)

        # TODO: check to make sure that variable is a scalar

        self.s_var_name = self.options['state_var']

        self.var_lower_bound = None
        var_lower_bound = self.options['var_lower_bound']
        if var_lower_bound.strip() != '':
            for var_name, meta in iteritems(sub.params):
                if meta['top_promoted_name'] == var_lower_bound:
                    self.var_lower_bound = var_name
                    break
            if self.var_lower_bound is None:
                raise(ValueError("'var_lower_bound' variable '%s' was not found as a parameter on any component in %s"%(var_lower_bound, sub.pathname)))

        self.var_upper_bound = None
        var_upper_bound = self.options['var_upper_bound']
        if var_upper_bound.strip() != '':
            for var_name, meta in iteritems(sub.params):
                if meta['top_promoted_name'] == var_upper_bound:
                    self.var_upper_bound = var_name
                    break
            if self.var_upper_bound is None:
                raise(ValueError("'var_lower_bound' variable '%s' was not found as a parameter on any component in %s"%(var_upper_bound, sub.pathname)))

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Solves the system using the Brent Method.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        self.sys = system
        self.metadata = metadata
        self.local_meta = create_local_meta(self.metadata, self.sys.pathname)
        self.sys.ln_solver.local_meta = self.local_meta
        idx = self.options['state_var_idx']

        if self.var_lower_bound is not None:
            lower = params[self.var_lower_bound]
        else:
            lower = self.options['lower_bound']

        if self.var_upper_bound is not None:
            upper = params[self.var_upper_bound]
        else:
            upper = self.options['upper_bound']

        kwargs = {'maxiter': self.options['maxiter'],
                  'a': lower,
                  'b': upper,
                  'full_output': False, # False, because we don't use the info, so just wastes operations
                  'args': (params, unknowns, resids)
                  }

        if self.options['xtol']:
            kwargs['xtol'] = self.options['xtol']
        if self.options['rtol'] > 0:
            kwargs['rtol'] = self.options['rtol']

        # Brent's method
        self.iter_count = 0

        # initial run to compute initial_norm
        self.sys.children_solve_nonlinear(self.local_meta)
        self.recorders.record_iteration(system, self.local_meta)

        # Evaluate Norm
        self.sys.apply_nonlinear(params, unknowns, resids)
        self.basenorm = resid_norm_0 = abs(resids._dat[self.s_var_name].val[idx])

        failed = False
        try:
            xstar = brentq(self._eval, **kwargs)
        except RuntimeError as err:
            msg = str(err)
            if 'different signs' in msg:
                raise
            failed = True

        self.sys = None

        resid_norm = abs(resids._dat[self.s_var_name].val[idx])

        if self.options['iprint'] > 0:

            if not failed:
                msg = 'converged'

            self.print_norm(self.print_name, system.pathname, self.iter_count,
                            resid_norm, resid_norm_0, msg=msg)

        if failed and self.options['err_on_maxiter']:
            raise AnalysisError(msg)

    def _eval(self, x, params, unknowns, resids):
        """Callback function for evaluating f(x)"""

        idx = self.options['state_var_idx']
        self.iter_count += 1
        update_local_meta(self.local_meta, (self.iter_count, ))

        unknowns._dat[self.s_var_name].val[idx] = x

        self.sys.children_solve_nonlinear(self.local_meta)
        self.sys.apply_nonlinear(params, unknowns, resids)

        self.recorders.record_iteration(self.sys, self.local_meta)

        if self.options['iprint'] > 0:
            normval = abs(resids._dat[self.s_var_name].val[idx])
            self.print_norm(self.print_name, self.sys.pathname, self.iter_count, normval,
                            self.basenorm)

        return resids._dat[self.s_var_name].val[idx]
