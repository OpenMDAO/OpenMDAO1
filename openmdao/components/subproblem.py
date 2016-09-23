
from __future__ import print_function

import sys
import os
from collections import OrderedDict
from itertools import chain

import numpy

from six import iteritems, itervalues, reraise

from openmdao.core.component import Component
from openmdao.util.dict_util import _jac_to_flat_dict
from openmdao.core.mpi_wrap import MPI


def _reraise(pathname, exc):
    """
    Rather than adding the sub-Problem pathname to every system and variable
    in the sub-Problem (and causing more complication w.r.t. promoted names),
    just put a try block around all of the calls to the sub-Problem and
    preface any exception messages with "In subproblem 'x' ..."
    """
    new_err = exc[0]("In subproblem '%s': %s" % (pathname, str(exc[1])))
    reraise(exc[0], new_err, exc[2])

class SubProblem(Component):
    """A Component that wraps a sub-Problem.

    Args
    ----

    problem : Problem
        The Problem to be wrapped by this component.

    params : iter of str
        Names of variables that are to be visible as parameters to
        this component.  Note that these are allowed to be unknowns in
        the sub-problem.

    unknowns : iter of str
        Names of variables that are to be visible as unknowns in this
        component.
    """

    def __init__(self, problem, params=(), unknowns=()):
        super(SubProblem, self).__init__()
        self._problem = problem
        self._prob_params = list(params)
        self._prob_unknowns = list(unknowns)

    def check_setup(self, out_stream=sys.stdout):
        """Write a report to the given stream indicating any potential problems
        found with the current configuration of this ``System``.

        Args
        ----
        out_stream : a file-like object, optional
            Stream where report will be written.
        """
        try:
            self._problem.check_setup(out_stream)
        except:
            _reraise(self.pathname,  sys.exc_info())

    def cleanup(self):
        """ Clean up resources prior to exit. """
        try:
            self._problem.cleanup()
        except:
            _reraise(self.pathname,  sys.exc_info())

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the min and max
            processors usable by this `System`.
        """

        # because this is called before self._problem.setup, we need to go
        # ahead and set the problem's driver's root explicitly here.
        self._problem.driver.root = self._problem.root

        try:
            return self._problem.get_req_procs()
        except:
            _reraise(self.pathname,  sys.exc_info())

    def _get_relname_map(self, parent_proms):
        """
        Args
        ----
        parent_proms : `dict`
            A dict mapping absolute names to promoted names in the parent
            system.

        Returns
        -------
        dict
            Maps promoted name in parent (owner of unknowns) to
            the corresponding promoted name in the child.
        """
        # use an ordered dict here so we can use this smaller dict when looping
        # during get_view.
        #   (the order of this one matches the order in the parent)
        umap = OrderedDict()

        for key in self._prob_unknowns:
            pkey = '.'.join((self.name, key))
            if pkey in parent_proms:
                umap[parent_proms[pkey]] = key

        return umap

    def _rec_get_param(self, name):
        return self.params[name]

    def _rec_get_param_meta(self, name):
        try:
            return self._problem.root._rec_get_param_meta(name)
        except:
            _reraise(self.pathname,  sys.exc_info())

    def _rec_set_param(self, name, value):
        self.params[name] = value

    def _setup_communicators(self, comm, parent_dir):
        """
        Assign communicator to this `System` and run full setup on its
        subproblem.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.

        parent_dir : str
            The absolute directory of the parent, or '' if unspecified. Used to
            determine the absolute directory of all subsystems.

        """

        self._problem.comm = comm

        # do full setup on our subproblem now that we have what we need
        # check_setup will be called later if specified from the top level
        # Problem so always set check=False here.
        try:
            self._problem.setup(check=False)
        except:
            _reraise(self.pathname,  sys.exc_info())

        super(SubProblem, self)._setup_communicators(comm, parent_dir)

        self._problem.pathname = self.pathname
        self._problem._parent_dir = self._sysdata.absdir


        for p in self._prob_params:
            if not (p in self._problem._dangling or p in self._problem.root.unknowns):
                raise RuntimeError("Param '%s' cannot be set. Either it will "
                                   "be overwritten by a connected output or it "
                                   "doesn't exist." % p)

    def _setup_variables(self):
        """
        Returns copies of our params and unknowns dictionaries,
        re-keyed to use absolute variable names.

        """
        to_prom_name = self._sysdata.to_prom_name = {}
        to_abs_uname = self._sysdata.to_abs_uname = {}
        to_abs_pnames = self._sysdata.to_abs_pnames = OrderedDict()
        to_prom_uname = self._sysdata.to_prom_uname = OrderedDict()
        to_prom_pname = self._sysdata.to_prom_pname = OrderedDict()

        # Our subproblem has been completely set up. We now just pull
        # variable metadata from our subproblem
        subparams = self._problem.root.params
        subunknowns = self._problem.root.unknowns

        # keep track of params that are actually unknowns in the subproblem
        self._unknowns_as_params = []

        self._params_dict = self._init_params_dict = OrderedDict()
        for name in self._prob_params:
            pathname = self._get_var_pathname(name)
            if name in subparams:
                meta = subparams._dat[name].meta
            elif name in self._problem._dangling:
                meta = self._rec_get_param_meta(name)
            else:
                meta = subunknowns._dat[name].meta
                if not meta.get('_canset_'):
                    raise TypeError("SubProblem param '%s' is mapped to the output of an internal component."
                                    " This is illegal because a value set into the param will be overwritten"
                                    " by the internal component." % name)
                self._unknowns_as_params.append(name)

            meta = meta.copy() # don't mess with subproblem's metadata!

            self._params_dict[pathname] = meta
            meta['pathname'] = pathname
            del meta['top_promoted_name']
            to_prom_pname[pathname] = name
            to_abs_pnames[name] = (pathname,)

        self._unknowns_dict = self._init_unknowns_dict = OrderedDict()

        # if we have params that are really unknowns in the subproblem, we
        # also add them as unknowns so we can take derivatives
        for name in self._prob_unknowns:
            pathname = self._get_var_pathname(name)
            meta = subunknowns._dat[name].meta.copy()
            self._unknowns_dict[pathname] = meta
            meta['pathname'] = pathname
            del meta['top_promoted_name']
            to_prom_uname[pathname] = name
            to_abs_uname[name] = pathname

        to_prom_name.update(to_prom_uname)
        to_prom_name.update(to_prom_pname)

        self._post_setup_vars = True

        self._sysdata._params_dict = self._params_dict
        self._sysdata._unknowns_dict = self._unknowns_dict

        return self._params_dict, self._unknowns_dict

    def solve_nonlinear(self, params, unknowns, resids):
        """Sets params into the sub-problem, runs the
        sub-problem, and updates our unknowns with values
        from the sub-problem.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)
        """
        if not self.is_active():
            return

        try:
            # set params into the subproblem
            prob = self._problem
            for name in self._prob_params:
                prob[name] = params[name]

            self._problem.run()

            # update our unknowns from subproblem
            for name in self._sysdata.to_abs_uname:
                unknowns[name] = prob.root.unknowns[name]
                resids[name] = prob.root.resids[name]

            # if params are really unknowns, they may have changed, so update
            for name in self._unknowns_as_params:
                params[name] = prob.root.unknowns[name]
        except:
            _reraise(self.pathname,  sys.exc_info())

    def linearize(self, params, unknowns, resids):
        """
        Returns Jacobian. J is a dictionary whose keys are tuples
        of the form ('unknown', 'param') and whose values are ndarrays.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays.
        """
        try:
            prob = self._problem

            # set params into the subproblem
            for name in self._prob_params:
                prob[name] = params[name]

            # have to convert jacobian returned from calc_gradient from a
            # nested dict to a flat dict with tuple keys.
            return _jac_to_flat_dict(prob.calc_gradient(self.params.keys(),
                                                        self.unknowns.keys(),
                                                        return_format='dict'))
        except:
            _reraise(self.pathname,  sys.exc_info())

    def add_param(self, name, **kwargs):
        raise NotImplementedError("Can't add '%s' to SubProblem. "
                                  "add_param is not supported." % name)

    def add_output(self, name, **kwargs):
        raise NotImplementedError("Can't add '%s' to SubProblem. "
                                  "add_output is not supported." % name)

    def add_state(self, name, **kwargs):
        raise NotImplementedError("Can't add '%s' to SubProblem. "
                                  "add_state is not supported." % name)
