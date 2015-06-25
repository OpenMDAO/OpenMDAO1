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
from openmdao.core.options import OptionsDictionary

_snopt_options = {
        # SNOPT Printing Options
        'Major print level':(1, 'Majors Print (1 - line major iteration log)'),
        'Minor print level':(1, 'Minors Print (1 - line minor iteration log)'),
        'Print file':('SNOPT_print.out','Print File Name (specified by subroutine snInit)'),
        'iPrint':(18,                  'Print File Output Unit (override internally in snopt?)'),
        'Summary file':('SNOPT_summary.out', 'Summary File Name (specified by subroutine snInit)'),
        'iSumm':(19,             'Summary File Output Unit (override internally in snopt?)'),
        'Print frequency':(100,   'Minors Log Frequency on Print File'),
        'Summary frequency':(100, 'Minors Log Frequency on Summary File'),
        'Solution':('Yes',  'Print Solution on the Print File'),
        'Suppress options listing':(None,  '(options are normally listed)'),
        'System information':('No', 'Print System Information on the Print File'),
        # SNOPT Problem Specification Options
        'Problem Type':('Minimize', " 'Minimize', 'Maximize', or 'Feasible point'"),
        'Objective row':(1, '(has precedence over ObjRow (snOptA))'),
        'Infinite bound':(1.0e+20, 'Infinite Bound Value'),
        # # SNOPT Convergence Tolerances Options
        'Major feasibility tolerance':(1.0e-6, 'Target Nonlinear Constraint Violation'),
        #TODO setting Major optimality tolerance here results in the optimization getting "lost", despite this being the default value
        'Major optimality tolerance':(1.0e-4,  'Target Complementarity Gap'),
        'Minor feasibility tolerance':(1.0e-6, 'For Satisfying the QP Bounds'),
        # # SNOPT Derivative Checking Options
        'Verify level':(0, 'Gradients Check Flag'),
        'Start objective check at column':(1, 'Start the gradient verification at this column'),
        'Start constraint check at column':(1,'Start the jacobian verification at this column'),
        # SNOPT Scaling Options
        'Scale option':(1, 'Scaling (1 - linear constraints and variables)'),
        'Scale tolerance':(0.9, 'Scaling Tolerance'),
        'Scale Print':(None, 'Default: scales are not printed'),
        # SNOPT Other Tolerances Options
        'Crash tolerance':(0.1,'Crash tolerance'),
        'Linesearch tolerance':(0.9, 'smaller for more accurate search'),
        'Pivot tolerance':(3.7e-11,  'epsilon^(2/3)'),
        # # SNOPT QP subproblems Options
        'QPSolver':('Cholesky','Default: Cholesky'),
        'Crash option':(3,  '(3 - first basis is essentially triangular)'),
        'Elastic mode':('No', '(start with elastic mode until necessary)'),
        'Elastic weight':(1.0e+4, '(used only during elastic mode)'),
        'Iterations limit':(10000,  '(or 20*ncons if that is more)'),
        'Partial price':(1,         '(10 for large LPs)'),
        'Start':('Cold',  "has precedence over argument start, ('Warm':( alternative to a cold start)"),
        # # SNOPT SQP method Options'
        'Major iterations limit':(1000, 'or ncons if that is more'),
        'Minor iterations limit':(500,  'or 3*ncons if that is more'),
        'Major step limit':(2.0,'Limit to the change in x during a linesearch'),
        'Superbasics limit':(None, '(n1 + 1, n1 = number of nonlinear variables)'),
        'Derivative level':(3,     '(NOT ALLOWED IN snOptA)'),
        'Derivative option':(1,    '(ONLY FOR snOptA)'),
        'Derivative linesearch':(None,''),
        'Nonderivative linesearch':(None,''),
        'Function precision':(3.0e-13, 'epsilon^0.8 (almost full accuracy)'),
        'Difference interval':(5.5e-7, 'Function precision^(1/2)'),
        'Central difference interval':(6.7e-5, 'Function precision^(1/3)'),
        'New superbasics limit':(99, 'controls early termination of QPs'),
        'Objective row':(1, 'row number of objective in F(x)'),
        'Penalty parameter':(0.0, 'initial penalty parameter'),
        'Proximal point method':(1, '(1 - satisfies linear constraints near x0)'),
        'Reduced Hessian dimension':(2000, '(or Superbasics limit if that is less)'),
        'Violation limit':(10.0,  '(unscaled constraint violation limit)'),
        'Unbounded step size':(1.0e+18,''),
        'Unbounded objective':(1.0e+15,''),
        # SNOPT Hessian approximation Options
        'Hessian full memory':(None, 'default if n1 <= 75'),
        'Hessian limited memory':(None, 'default if n1 > 75'),
        'Hessian frequency':(999999,'for full Hessian (never reset)'),
        'Hessian updates':(10, 'for limited memory Hessian'),
        'Hessian flush':(999999, 'no flushing'),
        # 'SNOPT Frequencies Options'
        'Check frequency':(60,   'test row residuals ||Ax - sk||'),
        'Expand frequency':(10000, 'for anti-cycling procedure'),
        'Factorization frequency':(50, '100 for LPs'),
        'Save frequency':(100, 'save basis map'),
        # SNOPT LUSOL Options'
        'LU factor tolerance':(3.99, 'for NP (100.0 for LP)'),
        'LU update tolerance':(3.99, 'for NP ( 10.0 for LP)'),
        'LU singularity tolerance':(3.2e-11,''),
        'LU partial pivoting':(None, 'default threshold pivoting strategy'),
        'LU rook pivoting':(None, 'threshold rook pivoting'),
        'LU complete pivoting':(None, 'threshold complete pivoting'),
        #SNOPT Basis files Options
        'Old basis file':(0, 'input basis map'),
        'New basis file':(0, 'output basis map'),
        'Backup basis file':(0, 'output extra basis map'),
        'Insert file':(0, 'input in industry format'),
        'Punch file':(0, 'output Insert data'),
        'Load file':(0, 'input names and values'),
        'Dump file':(0, 'output Load data'),
        'Solution file':(0, 'different from printed solution'),
        #SNOPT Partitions of cw, iw, rw Options
        # 'Total character workspace':(500, 'lencw:( 500'),
        # 'Total integer workspace':(None, 'leniw:( 500 + 100 * (m+n)'),
        # 'Total real workspace':(None, 'lenrw:( 500 + 200 * (m+n)'),
        # 'User character workspace':(500,''),
        # 'User integer workspace':(500,''),
        # 'User real workspace':(500,''),
        #SNOPT Miscellaneous Options')
        # 'Debug level':(1, '(0 - Normal, 1 - for developers)'),
        #'Timing level':(3, '(3 - print cpu times')
        }

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
        # self.options.add_option('SNOPT', OptionsDictionary(),
        #                          desc='SNOPT-specific options')

        self.opt_settings = {}

        # for key,value in _snopt_options.items():
        #     default_val, description = value
        #     self.options['SNOPT'].add_option(key,default_val,desc=description)

        self.pyopt_excludes = ['optimizer', 'title', 'print_results',
                               'pyopt_diff', 'exit_flag', 'SNOPT']

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

        #Set optimization options
        for option, value in self.opt_settings.items():
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
