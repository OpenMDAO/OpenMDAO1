"""
OpenMDAO Wrapper for pyoptsparse.

pyoptsparse is based on pyOpt, which is an object-oriented framework for formulating and solving nonlinear
constrained optimization problems, with additional MPI capability. Note: only SNOPT is supported right now.
"""
from __future__ import print_function

# pylint: disable=E0611,F0401
import numpy as np

from pyoptsparse import Optimization

from openmdao.core.driver import Driver


class pyOptSparseDriver(Driver):
    """ Driver wrapper for pyoptsparse. pyoptsparse is based on pyOpt, which
    is an object-oriented framework for formulating and solving nonlinear
    constrained optimization problems, with additional MPI capability. Note:
    only SNOPT is supported right now.
    """

    def __init__(self):
        """Initialize pyopt"""

        super(pyOptSparseDriver, self).__init__()

        # What we support
        self.supports['Inequality Constraints'] = True
        self.supports['Equality Constraints'] = True
        self.supports['Multiple Objectives'] = False

        # TODO: Support these
        self.supports['Linear Constraints'] = False
        self.supports['2-Sided Constraints'] = False
        self.supports['Integer Parameters'] = False

        # User Options
        self.options.add_option('optimizer', 'SNOPT', values=['SNOPT'],
                                desc='Name of optimizers to use')
        self.options.add_option('title', 'Optimization using pyOpt_sparse',
                                desc='Title of this optimization run')
        self.options.add_option('print_results', True,
                                 desc='Print pyOpt results if True')
        self.options.add_option('pyopt_diff', False,
                                 desc='Set to True to let pyOpt calculate the gradient')
        self.options.add_option('exit_flag', 0,
                                 desc='0 for fail, 1 for ok')

        self.pyopt_excludes = ['optimizer', 'title', 'print_results',
                               'pyopt_diff', 'exit_flag']

        self.pyOpt_solution = None

        self.lin_jacs = {}
        self.quantities = []

    def run(self, problem):
        """pyOpt execution. Note that pyOpt controls the execution, and the
        individual optimizers (i.e., SNOPT) control the iteration.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """

        self.pyOpt_solution = None

        # Initial Run
        problem.root.solve_nonlinear()

        opt_prob = Optimization(self.options['title'], self.objfunc)

        # Add all parameters
        param_meta = self.get_param_metadata()
        param_list = param_meta.keys()
        for name, meta in param_meta.items():

            vartype = 'c'
            lower_bounds = meta['low']
            upper_bounds = meta['high']
            n_vals = meta['size']

            opt_prob.addVarGroup(name, n_vals, type=vartype,
                                 lower=lower_bounds, upper=upper_bounds)
            param_list.append(name)

        # Add all objectives
        objs = self.get_objectives()
        self.quantities = objs.keys()
        for name, obj in objs.items():
            opt_prob.addObj(name)

        # Calculate and save gradient for any linear constraints.
        lcons = self.get_constraints(lintype='linear').values()
        if len(lcons) > 0:
            self.lin_jacs = problem.calc_gradient(param_list, lcons,
                                                  return_format='dict')
            #print("Linear Gradient")
            #print(self.lin_jacs)

        # Add all equality constraints
        econs = self.get_constraints(ctype='eq', lintype='nonlinear')
        con_meta = self.get_constraint_metadata()
        self.quantities += econs.keys()
        for name, con in econs.items():
            size = con_meta[name]['size']
            lower = np.zeros((size))
            upper = np.zeros((size))
            if con_meta[name]['linear'] is True:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     linear=True, wrt=param_list,
                                     jac=self.lin_jacs[name])
            else:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper)

        # Add all inequality constraints
        incons = self.get_constraints(ctype='ineq', lintype='nonlinear')
        self.quantities += incons.keys()
        for name, con in incons.items():
            size = con_meta[name]['size']
            upper = np.zeros((size))
            if con_meta[name]['linear'] is True:
                opt_prob.addConGroup(name, size, upper=upper, linear=True,
                wrt=param_list, jac=self.lin_jacs[name])
            else:
                opt_prob.addConGroup(name, size, upper=upper)

        # TODO: Support double-sided constraints in openMDAO
        # Add all double_sided constraints
        #for name, con in self.get_2sided_constraints().items():
            #size = con_meta[name]['size']
            #upper = con.high * np.ones((size))
            #lower = con.low * np.ones((size))
            #name = '%s.out0' % con.pcomp_name
            #if con.linear is True:
                #opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     #linear=True, wrt=param_list,
                                     #jac=self.lin_jacs[name])
            #else:
                #opt_prob.addConGroup(name, size, upper=upper, lower=lower)

        # Instantiate the requested optimizer
        optimizer = self.options['optimizer']
        try:
            exec('from pyoptsparse import %s' % optimizer)
        except ImportError:
            msg = "Optimizer %s is not available in this installation." % \
                   optimizer
            self.raise_exception(msg, ImportError)

        optname = vars()[optimizer]
        opt = optname()

        # Set optimization options
        for option, value in self.options.items():
            if option in self.pyopt_excludes:
                continue
            opt.setOption(option, value)

        self._problem = problem

        # Execute the optimization problem
        if self.options['pyopt_diff'] is True:
            # Use pyOpt's internal finite difference
            sol = opt(opt_prob, sens='FD', sensStep=self.gradient_options.fd_step)
        else:
            # Use OpenMDAO's differentiator for the gradient
            sol = opt(opt_prob, sens=self.gradfunc)

        self._problem = None

        # Print results
        if self.options['print_results'] is True:
            print(sol)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dv_dict = sol.getDVs()
        for name, param in self.get_params().items():
            val = dv_dict[name]
            self.set_param(name, val)

        self.root.solve_nonlinear()

        # Save the most recent solution.
        self.pyOpt_solution = sol
        try:
            exit_status = sol.optInform['value']
            self.exit_flag = 1
            if exit_status > 2: # bad
                self.exit_flag = 0
        except KeyError: #nothing is here, so something bad happened!
            self.exit_flag = 0

    def objfunc(self, dv_dict):
        """ Function that evaluates and returns the objective function and
        constraints. This function is passed to pyOpt's Optimization object
        and is called from its optimizers.

        dv_dict: dict
            Dictionary of design variable values

        Returns

        func_dict: dict
            Dictionary of all functional variables evaluated at design point

        fail: int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """

        fail = 1
        func_dict = {}

        try:

            for name, param in self.get_params().items():
                self.set_param(name, dv_dict[name])

            # Execute the model
            #print("Setting DV")
            #print(dv_dict)
            self.root.solve_nonlinear()

            # Get the objective function evaluations
            for name, obj in self.get_objectives().items():
                func_dict[name] = obj

            # Get the constraint evaluations
            for name, con in self.get_constraints().items():
                func_dict[name] = con

            # Get the double-sided constraint evaluations
            #for key, con in self.get_2sided_constraints().items():
            #    func_dict[name] = np.array(con.evaluate(self.parent))

            fail = 0

        except Exception as msg:

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70*"=")
            import traceback
            traceback.print_exc()
            print(70*"=")

        #print("Functions calculated")
        #print(func_dict)
        return func_dict, fail

    def gradfunc(self, dv_dict, func_dict):
        """ Function that evaluates and returns the gradient of the objective
        function and constraints. This function is passed to pyOpt's
        Optimization object and is called from its optimizers.

        dv_dict: dict
            Dictionary of design variable values

        func_dict: dict
            Dictionary of all functional variables evaluated at design point

        Returns

        sens_dict: dict
            Dictionary of dictionaries for gradient of each dv/func pair

        fail: int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """

        fail = 1
        sens_dict = {}

        try:
            sens_dict = self._problem.calc_gradient(dv_dict.keys(), self.quantities,
                                                    return_format='dict')
            #for key, value in self.lin_jacs.items():
            #    sens_dict[key] = value

            fail = 0

        except Exception as msg:

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70*"=")
            import traceback
            traceback.print_exc()
            print(70*"=")

        #print("Derivatives calculated")
        #print(dv_dict)
        #print(sens_dict)
        return sens_dict, fail
