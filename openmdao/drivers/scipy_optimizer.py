"""
OpenMDAO Wrapper for the scipy.optimize.minimize family of local optimizers.
"""

from __future__ import print_function

# pylint: disable=E0611,F0401
import numpy as np
from scipy.optimize import minimize

from openmdao.core.driver import Driver
from openmdao.util.recordutil import create_local_meta, update_local_meta

_optimizers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']
_gradient_optimizers = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC',
                        'SLSQP', 'dogleg', 'trust-ncg']
_bounds_optimizers = ['L-BFGS-B', 'TNC', 'SLSQP']


class ScipyOptimizer(Driver):
    """ Driver wrapper for the scipy.optimize.minimize family of local
    optimizers. Inequality constraints are supported by COBYLA and SLSQP,
    but equality constraints are only supported by COBYLA. None of the other
    optimizers support constraints.
    """

    def __init__(self):
        """Initialize the ScipyOptimizer."""

        super(ScipyOptimizer, self).__init__()

        # What we support
        self.supports['Inequality Constraints'] = True
        self.supports['Equality Constraints'] = True
        self.supports['Multiple Objectives'] = False

        # User Options
        self.options.add_option('optimizer', 'SLSQP', values=_optimizers,
                                desc='Name of optimizer to use')
        self.options.add_option('tol', 1.0e-6,
                                desc='Tolerance for termination. For detailed '
                                'control, use solver-specific options.')
        self.options.add_option('maxiter', 200,
                                desc='Maximum number of iterations.')
        self.options.add_option('disp', False,
                                desc='Set to True to print Scipy convergence '
                                'messages')

        # The user places optimizer-specific settings in here.
        self.opt_settings = {}

        self.metadata = None
        self._problem = None
        self.result = None

    def run(self, problem):
        """Optimize the problem using our choice of Scipy optimizer.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """

        # Metadata Setup
        self.metadata = create_local_meta(None, self.options['optimizer'])

        # Initial Run
        problem.root.solve_nonlinear()

        pmeta = self.get_param_metadata()
        self.objs = self.get_objectives().keys()

        self.opt_settings['maxiter'] = self.options['maxiter']
        self.opt_settings['disp'] = self.options['disp']

        # Size Problem
        nparam = 0
        for param in pmeta.values():
            nparam += param['size']
        x_init = np.zeros(nparam)

        # Initial Parameters
        i = 0
        for name, val in self.get_params().items():
            size = pmeta[name]['size']
            x_init[i:i+size] = val
            i += size

        # Provide gradients for optimizers that support it
        if self.options['optimizer'] in _gradient_optimizers:
            jac = self.gradfunc
        else:
            jac=None

        # optimize
        self._problem = problem
        result = minimize(self.objfunc, x_init,
                          #args=(),
                          method=self.options['optimizer'],
                          jac=jac,
                          #hess=None,
                          #hessp=None,
                          #bounds=None,
                          #constraints=(),
                          tol=self.options['tol'],
                          #callback=None,
                          options=self.opt_settings)

        self._problem = None
        self.result = result

    def objfunc(self, x_new):
        """ Function that evaluates and returns the objective function.

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
        for name, meta in self.get_param_metadata().items():
            size = meta['size']
            self.set_param(name, x_new[i:i+size])
            i += size

        self.iter_count += 1
        update_local_meta(metadata, (self.iter_count,))

        system.solve_nonlinear(metadata=metadata)
        for recorder in self.recorders:
            recorder.raw_record(system.params, system.unknowns,
                                system.resids, metadata)

        # Get the objective function evaluations
        for name, obj in self.get_objectives().items():
            f_new = obj
            break

        #print("Functions calculated")
        #print(x_new)
        #print(f_new)
        return f_new

    def gradfunc(self, x_new):
        """ Function that evaluates and returns the objective function.

        Args
        ----
        x_new : dict
            Dictionary of design variable values

        Returns
        -------
        ndarray
            Gradient of objective with respect to parameter array.
        """

        params = self.get_param_metadata().keys()
        grad = self._problem.calc_gradient(params, self.objs,
                                           return_format='array')

        #print("Gradients calculated")
        #print(grad)
        return grad