"""
OpenMDAO Wrapper for pyoptsparse.
pyoptsparse is based on pyOpt, which is an object-oriented framework for
formulating and solving nonlinear constrained optimization problems, with
additional MPI capability. Note: only SNOPT and SLSQP are currently supported.
"""

from __future__ import print_function

import traceback
from six import iteritems
from six.moves import range

import scipy as sp
import numpy as np

from pyoptsparse import Optimization

from openmdao.core.driver import Driver
from openmdao.core.system import AnalysisError
from openmdao.util.record_util import create_local_meta, update_local_meta
from collections import OrderedDict

# names of optimizers that use gradients
grad_drivers = set(['CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
                    'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG'])

# names of optimizers that allow multiple objectives
multi_obj_drivers = set(['NSGA2'])

def _check_imports():
    """ Dynamically remove optimizers we don't have
    """

    optlist = ['ALPSO', 'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
               'NSGA2', 'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG', 'NOMAD']

    for optimizer in optlist[:]:
        try:
            exec('from pyoptsparse import %s' % optimizer)
        except ImportError:
            optlist.remove(optimizer)

    return optlist


class pyOptSparseDriver(Driver):
    """ Driver wrapper for pyoptsparse. pyoptsparse is based on pyOpt, which
    is an object-oriented framework for formulating and solving nonlinear
    constrained optimization problems, with additional MPI capability.
    pypptsparse has interfaces to the following optimizers:
    ALPSO, CONMIN, FSQP, IPOPT, NLPQLP, NSGA2, PSQP, SLSQP,
    SNOPT, NLPY_AUGLAG, NOMAD.
    Note that some of these are not open source and therefore not included
    in the pyoptsparse source code.

    pyOptSparseDriver supports the following:
        equality_constraints

        inequality_constraints

        two_sided_constraints

    Options
    -------
    options['exit_flag'] :  int(0)
        0 for fail, 1 for ok
    options['optimizer'] :  str('SLSQP')
        Name of optimizers to use
    options['print_results'] :  bool(True)
        Print pyOpt results if True
    options['gradient method'] :  str('openmdao', 'pyopt_fd', 'snopt_fd')
        Finite difference implementation to use ('snopt_fd' may only be used with SNOPT)
    options['title'] :  str('Optimization using pyOpt_sparse')
        Title of this optimization run
    """

    def __init__(self):
        """Initialize pyopt"""

        super(pyOptSparseDriver, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = True
        self.supports['two_sided_constraints'] = True

        # TODO: Support these
        self.supports['linear_constraints'] = False
        self.supports['integer_design_vars'] = False

        # User Options
        self.options.add_option('optimizer', 'SLSQP', values=_check_imports(),
                                desc='Name of optimizers to use')
        self.options.add_option('title', 'Optimization using pyOpt_sparse',
                                desc='Title of this optimization run')
        self.options.add_option('print_results', True,
                                desc='Print pyOpt results if True')
        self.options.add_option('gradient method', 'openmdao', 
                                values={'openmdao', 'pyopt_fd', 'snopt_fd'},
                                desc='Finite difference implementation to use')
       
        # The user places optimizer-specific settings in here.
        self.opt_settings = {}

        # The user can set a file name here to store history
        self.hist_file = None

        self.pyopt_solution = None

        self.lin_jacs = OrderedDict()
        self.quantities = []
        self.metadata = None
        self.exit_flag = 0
        self._problem = None
        self.sparsity = OrderedDict()
        self.sub_sparsity = OrderedDict()

    def _setup(self):
        self.supports['gradients'] = self.options['optimizer'] in grad_drivers
        if len(self._objs) > 1 and self.options['optimizer'] not in multi_obj_drivers:
            raise RuntimeError('Multiple objectives have been added to pyOptSparseDriver'
                               ' but the selected optimizer ({0}) does not support'
                               ' multiple objectives.'.format(self.options['optimizer']))
        super(pyOptSparseDriver, self)._setup()

    def run(self, problem):
        """pyOpt execution. Note that pyOpt controls the execution, and the
        individual optimizers (i.e., SNOPT) control the iteration.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """

        self.pyopt_solution = None
        rel = problem.root._probdata.relevance

        # Metadata Setup
        self.metadata = create_local_meta(None, self.options['optimizer'])
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count,))

        # Initial Run
        with problem.root._dircontext:
            problem.root.solve_nonlinear(metadata=self.metadata)

        opt_prob = Optimization(self.options['title'], self._objfunc)

        # Add all parameters
        param_meta = self.get_desvar_metadata()
        self.indep_list = indep_list = list(param_meta)
        param_vals = self.get_desvars()

        for name, meta in iteritems(param_meta):
            opt_prob.addVarGroup(name, meta['size'], type='c',
                                 value=param_vals[name],
                                 lower=meta['lower'], upper=meta['upper'])

        opt_prob.finalizeDesignVariables()

        # Figure out parameter subsparsity for paramcomp index connections.
        # sub_param_conns is empty unless there are some index conns.
        # full_param_conns gets filled with the connections to the entire
        # parameter so that those params can be filtered out of the sparse
        # set if the full path is also relevant
        sub_param_conns = {}
        full_param_conns = {}
        for name in indep_list:
            pathname = problem.root.unknowns.metadata(name)['pathname']
            sub_param_conns[name] = {}
            full_param_conns[name] = set()
            for target, info in iteritems(problem.root.connections):
                src, indices = info
                if src == pathname:
                    if indices is not None:
                        # Need to map the connection indices onto the desvar
                        # indices if both are declared.
                        dv_idx = param_meta[name].get('indices')
                        indices = set(indices)
                        if dv_idx is not None:
                            indices.intersection_update(dv_idx)
                            ldv_idx = list(dv_idx)
                            mapped_idx = [ldv_idx.index(item) for item in indices]
                            sub_param_conns[name][target] = mapped_idx
                        else:
                            sub_param_conns[name][target] = indices
                    else:
                        full_param_conns[name].add(target)

        # Add all objectives
        objs = self.get_objectives()
        self.quantities = list(objs)
        self.sparsity = OrderedDict()
        self.sub_sparsity = OrderedDict()
        for name in objs:
            opt_prob.addObj(name)
            self.sparsity[name] = self.indep_list

        # Calculate and save gradient for any linear constraints.
        lcons = self.get_constraints(lintype='linear').keys()
        if len(lcons) > 0:
            self.lin_jacs = problem.calc_gradient(indep_list, lcons,
                                                  return_format='dict')
            #print("Linear Gradient")
            #print(self.lin_jacs)

        # Add all equality constraints
        econs = self.get_constraints(ctype='eq', lintype='nonlinear')
        con_meta = self.get_constraint_metadata()
        self.quantities += list(econs)

        for name in self.get_constraints(ctype='eq'):
            meta = con_meta[name]
            size = meta['size']
            lower = upper = meta['equals']

            # Sparsify Jacobian via relevance
            rels = rel.relevant[name]
            wrt = rels.intersection(indep_list)
            self.sparsity[name] = wrt

            if meta['linear']:
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     linear=True, wrt=wrt,
                                     jac=self.lin_jacs[name])
            else:

                jac = self._build_sparse(name, wrt, size, param_vals,
                                         sub_param_conns, full_param_conns, rels)
                opt_prob.addConGroup(name, size, lower=lower, upper=upper,
                                     wrt=wrt, jac=jac)

        # Add all inequality constraints
        incons = self.get_constraints(ctype='ineq', lintype='nonlinear')
        self.quantities += list(incons)

        for name in self.get_constraints(ctype='ineq'):
            meta = con_meta[name]
            size = meta['size']

            # Bounds - double sided is supported
            lower = meta['lower']
            upper = meta['upper']

            # Sparsify Jacobian via relevance
            rels = rel.relevant[name]
            wrt = rels.intersection(indep_list)
            self.sparsity[name] = wrt

            if meta['linear']:
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     linear=True, wrt=wrt,
                                     jac=self.lin_jacs[name])
            else:

                jac = self._build_sparse(name, wrt, size, param_vals,
                                         sub_param_conns, full_param_conns, rels)
                opt_prob.addConGroup(name, size, upper=upper, lower=lower,
                                     wrt=wrt, jac=jac)

        # Instantiate the requested optimizer
        optimizer = self.options['optimizer']
        try:
            exec('from pyoptsparse import %s' % optimizer)
        except ImportError:
            msg = "Optimizer %s is not available in this installation." % \
                   optimizer
            raise ImportError(msg)

        optname = vars()[optimizer]
        opt = optname()

        #Set optimization options
        for option, value in self.opt_settings.items():
            opt.setOption(option, value)

        self._problem = problem

        # Execute the optimization problem
        if self.options['gradient method'] == 'pyopt_fd':  
              
            # Use pyOpt's internal finite difference
            fd_step = problem.root.fd_options['step_size']
            sol = opt(opt_prob, sens='FD', sensStep=fd_step, storeHistory=self.hist_file) 
                       
        elif self.options['gradient method'] == 'snopt_fd':        
            if self.options['optimizer']=='SNOPT':            
            
                # Use SNOPT's internal finite difference
                fd_step = problem.root.fd_options['step_size']
                sol = opt(opt_prob, sens=None, sensStep=fd_step, storeHistory=self.hist_file)
                                
            else:
                msg = "SNOPT's internal finite difference can only be used with SNOPT"
                raise Exception(msg)                
        else:
        
            # Use OpenMDAO's differentiator for the gradient
            sol = opt(opt_prob, sens=self._gradfunc, storeHistory=self.hist_file)          
            
        self._problem = None

        # Print results
        if self.options['print_results']:
            print(sol)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dv_dict = sol.getDVs()
        for name in indep_list:
            val = dv_dict[name]
            self.set_desvar(name, val)

        with self.root._dircontext:
            self.root.solve_nonlinear(metadata=self.metadata)

        # Save the most recent solution.
        self.pyopt_solution = sol
        try:
            exit_status = sol.optInform['value']
            self.exit_flag = 1
            if exit_status > 2: # bad
                self.exit_flag = 0
        except KeyError: #nothing is here, so something bad happened!
            self.exit_flag = 0

    def _build_sparse(self, name, wrt, consize, param_vals, sub_param_conns,
                      full_param_conns, rels):
        """ Build up the data structures that define a sparse Jacobian
        matrix. Called separately on each nonlinear constraint.

        Args
        ----
        name : str
            Constraint name.
        wrt : list
            List of relevant param names.
        consize : int
            Width of this constraint.
        param_vals : dict
            Dictionary of parameter values; used for sizing.
        sub_param_conns : dict
            Parameter subindex connection info.
        full_param_conns : dict
            Parameter full connection info.
        rels : set
            Set of relevant nodes for this connstraint.

        Returns
        -------
        pyoptsparse coo matrix or None
        """

        jac = None

        # Additional sparsity for index connections
        for param in wrt:

            sub_conns = sub_param_conns.get(param)
            if not sub_conns:
                continue

            # If we have a simultaneous full connection, then we move on
            full_conns = full_param_conns.get(param)
            if full_conns.intersection(rels):
                continue

            rel_idx = set()
            for target, idx in iteritems(sub_conns):

                # If a target of the indexed desvar connection is
                # in the relevant path for this constraint, then
                # those indices are relevant.
                if target in rels:
                    rel_idx.update(idx)

            nrel = len(rel_idx)
            if nrel > 0:

                if jac is None:
                    jac = {}

                if param not in jac:
                    # A coo matrix for the Jacobian
                    # mat = {'coo':[row, col, data],
                    #        'shape':[nrow, ncols]}
                    coo = {}
                    coo['shape'] = [consize, len(param_vals[param])]
                    jac[param] = coo

                row = []
                col = []
                for i in range(consize):
                    row.extend([i]*nrel)
                    col.extend(rel_idx)
                data = np.ones((len(row), ))

                jac[param]['coo'] = [np.array(row), np.array(col), data]

                if name not in self.sub_sparsity:
                    self.sub_sparsity[name] = {}
                self.sub_sparsity[name][param] = np.array(list(rel_idx))

        return jac

    def _objfunc(self, dv_dict):
        """ Function that evaluates and returns the objective function and
        constraints. This function is passed to pyOpt's Optimization object
        and is called from its optimizers.

        Args
        ----
        dv_dict : dict
            Dictionary of design variable values.

        Returns
        -------
        func_dict : dict
            Dictionary of all functional variables evaluated at design point.

        fail : int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """

        fail = 0
        metadata = self.metadata
        system = self.root

        try:
            for name in self.indep_list:
                self.set_desvar(name, dv_dict[name])

            # Execute the model
            #print("Setting DV")
            #print(dv_dict)

            self.iter_count += 1
            update_local_meta(metadata, (self.iter_count,))

            try:
                with self.root._dircontext:
                    system.solve_nonlinear(metadata=metadata)

            # Let the optimizer try to handle the error
            except AnalysisError:
                fail = 1

            func_dict = self.get_objectives() # this returns a new OrderedDict
            func_dict.update(self.get_constraints())

            # Record after getting obj and constraint to assure they have
            # been gathered in MPI.
            self.recorders.record_iteration(system, metadata)

            # Get the double-sided constraint evaluations
            #for key, con in iteritems(self.get_2sided_constraints()):
            #    func_dict[name] = np.array(con.evaluate(self.parent))

        except Exception as msg:
            tb = traceback.format_exc()

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70*"=",tb,70*"=")
            fail = 1
            func_dict = {}

        #print("Functions calculated")
        #print(func_dict)
        return func_dict, fail

    def _gradfunc(self, dv_dict, func_dict):
        """ Function that evaluates and returns the gradient of the objective
        function and constraints. This function is passed to pyOpt's
        Optimization object and is called from its optimizers.

        Args
        ----
        dv_dict : dict
            Dictionary of design variable values.

        func_dict : dict
            Dictionary of all functional variables evaluated at design point.

        Returns
        -------
        sens_dict : dict
            Dictionary of dictionaries for gradient of each dv/func pair

        fail : int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """

        fail = 0

        try:

            try:
                sens_dict = self.calc_gradient(dv_dict, self.quantities,
                                               return_format='dict',
                                               sparsity=self.sparsity)

            # Let the optimizer try to handle the error
            except AnalysisError:
                fail = 1

                # We need to cobble together a sens_dict of the correct size.
                # Best we can do is return zeros.

                sens_dict = OrderedDict()
                for okey, oval in iteritems(func_dict):
                    sens_dict[okey] = OrderedDict()
                    osize = len(oval)
                    for ikey, ival in iteritems(dv_dict):
                        isize = len(ival)
                        sens_dict[okey][ikey] = np.zeros((osize, isize))

            # Support for sub-index sparsity by returning the Jacobian in a
            # pyopt sparse format.
            for con, val1 in iteritems(self.sub_sparsity):
                for desvar, rel_idx in iteritems(val1):
                    coo = {}
                    jac = sens_dict[con][desvar]
                    nrow, ncol = jac.shape
                    coo['shape'] = [nrow, ncol]

                    row = []
                    col = []
                    data = []
                    ncol = len(rel_idx)
                    for i in range(nrow):
                        row.extend([i]*ncol)
                        col.extend(rel_idx)
                        data.extend(jac[i][rel_idx])

                    coo['coo'] = [np.array(row), np.array(col), np.array(data)]
                    sens_dict[con][desvar] = coo

        except Exception as msg:
            tb = traceback.format_exc()

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print("Exception: %s" % str(msg))
            print(70*"=",tb,70*"=")
            sens_dict = {}

        #print("Derivatives calculated")
        #print(dv_dict)
        #print(sens_dict)
        return sens_dict, fail
