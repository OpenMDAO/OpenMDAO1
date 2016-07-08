""" Base class for all systems in OpenMDAO."""

from __future__ import print_function

import sys
import os
from collections import OrderedDict
from itertools import chain

import numpy

from six import iteritems, itervalues

from openmdao.core.component import Component
from openmdao.core.problem import _jac_to_flat_dict


class ProblemComponent(Component):
    """A System that contains a Problem."""

    def __init__(self, problem, params=(), unknowns=()):
        super(ProblemComponent, self).__init__()
        self._problem = problem
        self._prob_params = params[:]
        self._prob_unknowns = unknowns[:]

    def check_setup(self, out_stream=sys.stdout):
        """Write a report to the given stream indicating any potential problems
        found with the current configuration of this ``System``.

        Args
        ----
        out_stream : a file-like object, optional
            Stream where report will be written.
        """
        self._problem.check_setup(out_stream)

    def cleanup(self):
        """ Clean up resources prior to exit. """
        self._problem.cleanup()

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the min and max
            processors usable by this `System`.
        """
        return self._problem.root.get_req_procs()

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

        chop = len(self.name)+1
        for key in self._prob_unknowns:
            pkey = '.'.join((self.name, key))
            if pkey in parent_proms:
                umap[parent_proms[pkey]] = key

        return umap

    def _rec_get_param(self, name):
        return self.params[name]

    def _rec_get_param_meta(self, name):
        return self._problem.root._rec_get_param_meta(name)

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
        super(ProblemComponent, self)._setup_communicators(comm, parent_dir)

        self._problem.comm = comm
        #self._problem.pathname = self.pathname
        self._problem._parent_dir = self._sysdata.absdir

        # now do full setup on our subproblem now that we have what we need
        # check_setup will be called later if specified from the top level Problem
        self._problem.setup(check=False)

        # only set params that are either dangling params in the subproblem or are
        # unknowns in the subproblem.
        self._params_to_set = [p for p in self._prob_params
                                       if p in self._problem._dangling or
                                       p in self._problem.root.unknowns]

    def _setup_variables(self, compute_indices=False):
        """
        Returns copies of our params and unknowns dictionaries,
        re-keyed to use absolute variable names.

        Args
        ----

        compute_indices : bool, optional
            If True, call setup_distrib() to set values of
            'src_indices' metadata.

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
                meta = subparams._dat[name].meta.copy()
            elif name in self._problem._dangling:
                meta = self._rec_get_param_meta(name)
            else:
                meta = subunknowns._dat[name].meta.copy()
                self._unknowns_as_params.append(name)

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
        # set params into the subproblem
        prob = self._problem
        for name in self._params_to_set:
            prob[name] = params[name]

        for name in self._unknowns_as_params:
            self._problem.root.unknowns[name] = params[name]

        self._problem.run()

        # update our unknowns from subproblem
        for name in self._sysdata.to_abs_uname:
            unknowns[name] = prob[name]

        # TODO: do we need to copy subproblem resids?


    def linearize(self, params, unknowns, resids):
        """
        Returns Jacobian. Returns None unless component overides this method
        and returns something. J should be a dictionary whose keys are tuples
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
        # set params into the subproblem
        prob = self._problem
        for name in self._params_to_set:
            prob[name] = params[name]

        for name in self._unknowns_as_params:
            self._problem.root.unknowns[name] = params[name]

        indep_list = self.params.keys()
        unknowns_list = self.unknowns.keys()
        ret = self._problem.calc_gradient(indep_list, unknowns_list, return_format='dict')

        ## NOT sure if we need to update our unknowns in this case...
        # update our unknowns from subproblem
        for name in self._sysdata.to_abs_uname:
            unknowns[name] = prob[name]

        # TODO: do we need to copy subproblem resids?

        # have to convert jacobian returned from calc_gradient from a nested dict to
        # a flat dict with tuple keys.
        return _jac_to_flat_dict(ret)
