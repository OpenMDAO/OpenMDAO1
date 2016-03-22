"""
OpenMDAO Wrapper for the scipy.optimize.minimize family of local optimizers.
"""

from __future__ import print_function

from six import itervalues, iteritems
from six.moves import range

import numpy as np
from scipy.optimize import minimize

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta
from collections import OrderedDict

_optimizers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP']
_gradient_optimizers = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC',
                        'SLSQP', 'dogleg', 'trust-ncg']
_bounds_optimizers = ['L-BFGS-B', 'TNC', 'SLSQP']
_constraint_optimizers = ['COBYLA', 'SLSQP']
_constraint_grad_optimizers = ['SLSQP']

# These require Hessian or Hessian-vector product, so they are unsupported
# right now.
_unsupported_optimizers = ['dogleg', 'trust-ncg']


class ScipyOptimizer(Driver):
    """ Driver wrapper for the scipy.optimize.minimize family of local
    optimizers. Inequality constraints are supported by COBYLA and SLSQP,
    but equality constraints are only supported by COBYLA. None of the other
    optimizers support constraints.

    ScipyOptimizer supports the following:
        equality_constraints

        inequality_constraints

    Options
    -------
    options['disp'] :  bool(True)
        Set to False to prevent printing of Scipy convergence messages
    options['maxiter'] : int(200)
        Maximum number of iterations.
    options['optimizer'] : str('SLSQP')
        Name of optimizer to use
    options['tol'] :  float(1e-06)
        Tolerance for termination. For detailed control, use solver-specific options.

    """

    def __init__(self):
        """Initialize the ScipyOptimizer."""

        super(ScipyOptimizer, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = False

        # User Options
        self.options.add_option('optimizer', 'SLSQP', values=_optimizers,
                                desc='Name of optimizer to use')
        self.options.add_option('tol', 1.0e-6, lower=0.0,
                                desc='Tolerance for termination. For detailed '
                                'control, use solver-specific options.')
        self.options.add_option('maxiter', 200, lower=0,
                                desc='Maximum number of iterations.')
        self.options.add_option('disp', True,
                                desc='Set to False to prevent printing of Scipy '
                                'convergence messages')

        # The user places optimizer-specific settings in here.
        self.opt_settings = OrderedDict()

        self.metadata = None
        self._problem = None
        self.result = None
        self.exit_flag = 0
        self.grad_cache = None
        self.con_cache = None
        self.con_idx = OrderedDict()
        self.cons = None
        self.objs = None

    def _setup(self):
        self.supports['gradients'] = self.options['optimizer'] in _gradient_optimizers
        super(ScipyOptimizer, self)._setup()

    def run(self, problem):
        """Optimize the problem using your choice of Scipy optimizer.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """

        # Metadata Setup
        opt = self.options['optimizer']
        self.metadata = create_local_meta(None, opt)
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count,))

        # Initial Run
        with problem.root._dircontext:
            problem.root.solve_nonlinear(metadata=self.metadata)

        pmeta = self.get_desvar_metadata()
        self.params = list(pmeta)
        self.objs = list(self.get_objectives())
        con_meta = self.get_constraint_metadata()
        self.cons = list(con_meta)
        self.con_cache = self.get_constraints()

        self.opt_settings['maxiter'] = self.options['maxiter']
        self.opt_settings['disp'] = self.options['disp']

        # Size Problem
        nparam = 0
        for param in itervalues(pmeta):
            nparam += param['size']
        x_init = np.empty(nparam)

        # Initial Parameters
        i = 0
        use_bounds = (opt in _bounds_optimizers)
        if use_bounds:
            bounds = []
        else:
            bounds = None

        for name, val in iteritems(self.get_desvars()):
            size = pmeta[name]['size']
            x_init[i:i+size] = val
            i += size

            # Bounds if our optimizer supports them
            if use_bounds:
                meta_low = pmeta[name]['lower']
                meta_high = pmeta[name]['upper']
                for j in range(0, size):

                    if isinstance(meta_low, np.ndarray):
                        p_low = meta_low[j]
                    else:
                        p_low = meta_low

                    if isinstance(meta_high, np.ndarray):
                        p_high = meta_high[j]
                    else:
                        p_high = meta_high

                    bounds.append((p_low, p_high))

        # Constraints
        constraints = []
        i = 0
        if opt in _constraint_optimizers:
            for name, meta in con_meta.items():
                size = meta['size']
                dblcon = meta['upper'] is not None and meta['lower'] is not None
                for j in range(0, size):
                    con_dict = OrderedDict()
                    if meta['equals'] is not None:
                        con_dict['type'] = 'eq'
                    else:
                        con_dict['type'] = 'ineq'
                    con_dict['fun'] = self._confunc
                    if opt in _constraint_grad_optimizers:
                        con_dict['jac'] = self._congradfunc
                    con_dict['args'] = [name, j]
                    constraints.append(con_dict)
                self.con_idx[name] = i
                i += size

                # Add extra constraint if double-sided
                if dblcon:
                    name = '2bl-' + name
                    for j in range(0, size):
                        con_dict = OrderedDict()
                        con_dict['type'] = 'ineq'
                        con_dict['fun'] = self._confunc
                        if opt in _constraint_grad_optimizers:
                            con_dict['jac'] = self._congradfunc
                        con_dict['args'] = [name, j]
                        constraints.append(con_dict)

        # Provide gradients for optimizers that support it
        if opt in _gradient_optimizers:
            jac = self._gradfunc
        else:
            jac = None

        # optimize
        self._problem = problem
        result = minimize(self._objfunc, x_init,
                          #args=(),
                          method=opt,
                          jac=jac,
                          #hess=None,
                          #hessp=None,
                          bounds=bounds,
                          constraints=constraints,
                          tol=self.options['tol'],
                          #callback=None,
                          options=self.opt_settings)

        self._problem = None
        self.result = result
        self.exit_flag = 1 if self.result.success else 0

        if self.options['disp']:
            print('Optimization Complete')
            print('-'*35)

    def _objfunc(self, x_new):
        """ Function that evaluates and returns the objective function. Model
        is executed here.

        Args
        ----
        x_new : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """

        system = self.root
        metadata = self.metadata

        # Pass in new parameters
        i = 0
        for name, meta in self.get_desvar_metadata().items():
            size = meta['size']
            self.set_desvar(name, x_new[i:i+size])
            i += size

        self.iter_count += 1
        update_local_meta(metadata, (self.iter_count,))

        with system._dircontext:
            system.solve_nonlinear(metadata=metadata)

        # Get the objective function evaluations
        for name, obj in self.get_objectives().items():
            f_new = obj
            break

        self.con_cache = self.get_constraints()

        # Record after getting obj and constraints to assure it has been
        # gathered in MPI.
        self.recorders.record_iteration(system, metadata)

        #print("Functions calculated")
        #print(x_new)
        #print(f_new)

        return f_new

    def _confunc(self, x_new, name, idx):
        """ Function that returns the value of the constraint function
        requested in args. Note that this function is called for each
        constraint, so the model is only run when the objective is evaluated.

        Args
        ----
        x_new : ndarray
            Array containing parameter values at new design point.
        name : string
            Name of the constraint to be evaluated.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """

        if name.startswith('2bl-'):
            name = name[4:]
            dbl_side = True
        else:
            dbl_side = False

        cons = self.con_cache
        meta = self._cons[name]

        # Equality constraints
        bound = meta['equals']
        if bound is not None:
            if isinstance(bound, np.ndarray):
                bound = bound[idx]
            return bound - cons[name][idx]

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        upper = meta['upper']
        lower = meta['lower']
        if lower is None or dbl_side:
            if isinstance(upper, np.ndarray):
                upper = upper[idx]
            return upper - cons[name][idx]
        else:
            if isinstance(lower, np.ndarray):
                lower = lower[idx]
            return cons[name][idx] - lower

    def _gradfunc(self, x_new):
        """ Function that evaluates and returns the objective function.
        Gradients for the constraints are also calculated and cached here.

        Args
        ----
        x_new : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        ndarray
            Gradient of objective with respect to parameter array.
        """

        grad = self.calc_gradient(self.params, self.objs+self.cons,
                                  return_format='array')
        self.grad_cache = grad

        #print("Gradients calculated")
        #print(x_new)
        #print(grad[0, :])

        return grad[0, :]

    def _congradfunc(self, x_new, name, idx):
        """ Function that returns the cached gradient of the constraint
        function. Note, scipy calls the constraints one at a time, so the
        gradient is cached when the objective gradient is called.

        Args
        ----
        x_new : ndarray
            Array containing parameter values at new design point.
        name : string
            Name of the constraint to be evaluated.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Gradient of the constraint function wrt all params.
        """

        if name.startswith('2bl-'):
            name = name[4:]
            dbl_side = True
        else:
            dbl_side = False

        grad = self.grad_cache
        meta = self._cons[name]
        grad_idx = self.con_idx[name] + idx + 1

        #print("Constraint Gradient returned")
        #print(x_new)
        #print(name, idx, grad[grad_idx, :])

        # Equality constraints
        if meta['equals'] is not None:
            return -grad[grad_idx, :]

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        if meta['lower'] is None or dbl_side:
            return -grad[grad_idx, :]
        else:
            return grad[grad_idx, :]
