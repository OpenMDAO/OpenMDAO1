"""
OpenMDAO Wrapper for pyoptsparse.

pyoptsparse is based on pyOpt, which is an object-oriented framework for formulating and solving nonlinear
constrained optimization problems, with additional MPI capability. Note: only SNOPT is supported right now.
"""

# pylint: disable=E0611,F0401
from numpy import array, zeros, ones, float32, float64, int32, int64
import numpy as np

from pyoptsparse import Optimization

from openmdao.main.api import Driver


#def _check_imports():
    #""" Dynamically remove optimizers we don't have
    #"""

    #optlist = ['ALPSO', 'CONMIN', 'FSQP', 'IPOPT',
               #'NLPQL', 'NLPY_AUGLAG', 'NSGA2', 'PSQP', 'SLSQP',
               #'SNOPT']

    #for optimizer in optlist[:]:
        #try:
            #exec('from pyoptsparse import %s' % optimizer)
        #except ImportError:
            #optlist.remove(optimizer)

    #return optlist


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

        # User Options
        self.options.add_option('optimizer', 'SNOPt', values=['SNOPT'],
                                desc='Name of optimizers to use')
        self.options.add_option('title', 'Optimization using pyOpt_sparse',
                                desc='Title of this optimization run')
        self.options.add_options('print_results', True,
                                 desc='Print pyOpt results if True')
        self.options.add_options('pyopt_diff', False,
                                 desc='Set to True to let pyOpt calculate the gradient')
        self.options.add_options('exit_flag', 0,
                                 desc='0 for fail, 1 for ok')

        self.pyOpt_solution = None
        self.param_type = {}
        self.nparam = None

        self.objs = None
        self.nlcons = None
        self.lin_jacs = {}

    def execute(self, problem):
        """pyOpt execution. Note that pyOpt controls the execution, and the
        individual optimizers control the iteration.

        Parameters
        ----------
        problem : `Problem`
            Our parent `Problem`.
        """

        self.pyOpt_solution = None

        # Initial Run
        problem.root.solve_nonlinear()

        opt_prob = Optimization(self.title, self.objfunc)

        # Add all parameters
        self.param_type = {}
        self.nparam = self.total_parameters()
        param_list = []

        #need a counter for lb and ub arrays
        i_param = 0


        for name, param in self.get_parameters().iteritems():

            if isinstance(name, tuple):
                name = name[0]

            # We need to identify Enums, Lists, Dicts
            metadata = param.get_metadata()[1]
            values = param.evaluate()

            # Assuming uniform enumerated, discrete, or continuous for now.
            val = values[0]
            n_vals = len(values)
            choices = []
            if 'values' in metadata and \
               isinstance(metadata['values'], (list, tuple, array, set)):
                vartype = 'd'
                choices = metadata['values']
            elif isinstance(val, bool):
                vartype = 'd'
                choices = [True, False]
            elif isinstance(val, (int, int32, int64)):
                vartype = 'i'
            elif isinstance(val, (float, float32, float64)):
                vartype = 'c'
            else:
                msg = 'Only continuous, discrete, or enumerated variables' \
                      ' are supported. %s is %s.' % (name, type(val))
                self.raise_exception(msg, ValueError)
            self.param_type[name] = vartype

            if self.n_x is None:
                lower_bounds = param.get_low()
                upper_bounds = param.get_high()
            else:
                lower_bounds = self.lb[i_param:i_param+n_vals]
                upper_bounds = self.ub[i_param:i_param+n_vals]

            i_param += n_vals
            opt_prob.addVarGroup(name, n_vals, type=vartype,
                                 lower=lower_bounds, upper=upper_bounds,
                                 value=values, choices=choices)
            param_list.append(name)
        # Add all objectives
        for name, obj in self.get_objectives().iteritems():
            name = '%s.out0' % obj.pcomp_name
            opt_prob.addObj(name)

        # Calculate and save gradient for any linear constraints.
        lcons = self.get_constraints(linear=True).values() + \
                self.get_2sided_constraints(linear=True).values()
        if len(lcons) > 0:
            lcon_names = ['%s.out0' % obj.pcomp_name for obj in lcons]
            self.lin_jacs = self.workflow.calc_gradient(param_list, lcon_names,
                                                   return_format='dict')
            #print "Linear Gradient"
            #print self.lin_jacs

        # Add all equality constraints
        nlcons = []
        for name, con in self.get_eq_constraints().iteritems():
            size = con.size
            lower = zeros((size))
            upper = zeros((size))
            name = '%s.out0' % con.pcomp_name
            if con.linear is True:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     linear=True, wrt=param_list,
                                     jac=self.lin_jacs[name])
            else:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper)
                nlcons.append(name)

        # Add all inequality constraints
        for name, con in self.get_ineq_constraints().iteritems():
            size = con.size
            upper = zeros((size))
            name = '%s.out0' % con.pcomp_name
            if con.linear is True:
                opt_prob.addConGroup(name, size, upper=upper, linear=True,
                wrt=param_list, jac=self.lin_jacs[name])
            else:
                opt_prob.addConGroup(name, size, upper=upper)
                nlcons.append(name)

        # Add all double_sided constraints
        for name, con in self.get_2sided_constraints().iteritems():
            size = con.size
            upper = con.high * ones((size))
            lower = con.low * ones((size))
            name = '%s.out0' % con.pcomp_name
            if con.linear is True:
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     linear=True, wrt=param_list,
                                     jac=self.lin_jacs[name])
            else:
                opt_prob.addConGroup(name, size, upper=upper, lower=lower)
                nlcons.append(name)

        self.objs = self.list_objective_targets()
        self.nlcons = nlcons

        # Instantiate the requested optimizer
        optimizer = self.optimizer
        try:
            exec('from pyoptsparse import %s' % optimizer)
        except ImportError:
            msg = "Optimizer %s is not available in this installation." % \
                   optimizer
            self.raise_exception(msg, ImportError)

        optname = vars()[optimizer]
        opt = optname()

        # Set optimization options
        for option, value in self.options.iteritems():
            opt.setOption(option, value)

        # Execute the optimization problem
        if self.pyopt_diff:
            # Use pyOpt's internal finite difference
            sol = opt(opt_prob, sens='FD', sensStep=self.gradient_options.fd_step)
        else:
            # Use OpenMDAO's differentiator for the gradient
            sol = opt(opt_prob, sens=self.gradfunc)

        # Print results
        if self.print_results:
            print sol


        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dv_dict = sol.getDVs()
        param_types = self.param_type
        for name, param in self.get_parameters().iteritems():

            full_name = name
            if isinstance(name, tuple):
                name = name[0]

            val = dv_dict[name]
            if param_types[name] == 'i':
                val = int(round(val))

            self.set_parameter_by_name(full_name, val)

        self.run_iteration()

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

            # Integer parameters come back as floats, so we need to round them
            # and turn them into python integers before setting.
            param_types = self.param_type
            for name, param in self.get_parameters().iteritems():

                tup_name = name
                if isinstance(name, tuple):
                    name = name[0]

                val = dv_dict[name]
                if param_types[name] == 'i':
                    val = int(round(val))

                self.set_parameter_by_name(tup_name, val)

            # Execute the model
            #print "Setting DV"
            #print dv_dict
            self.run_iteration()

            # Get the objective function evaluations
            for key, obj in self.get_objectives().iteritems():
                name = '%s.out0' % obj.pcomp_name
                func_dict[name] = array(obj.evaluate())

            # Get the constraint evaluations
            for key, con in self.get_constraints().iteritems():
                name = '%s.out0' % con.pcomp_name
                func_dict[name] = array(con.evaluate(self.parent))

            # Get the double-sided constraint evaluations
            for key, con in self.get_2sided_constraints().iteritems():
                name = '%s.out0' % con.pcomp_name
                func_dict[name] = array(con.evaluate(self.parent))

            fail = 0

        except Exception as msg:

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print "Exception: %s" % str(msg)
            print 70*"="
            import traceback
            traceback.print_exc()
            print 70*"="

        #print "Functions calculated"
        #print func_dict
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
            sens_dict = self.workflow.calc_gradient(dv_dict.keys(), self.objs + self.nlcons,
                                                    return_format='dict')
            #for key, value in self.lin_jacs.iteritems():
            #    sens_dict[key] = value

            fail = 0

        except Exception as msg:

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print "Exception: %s" % str(msg)
            print 70*"="
            import traceback
            traceback.print_exc()
            print 70*"="

        #print "Derivatives calculated"
        #print dv_dict
        #print sens_dict
        return sens_dict, fail

