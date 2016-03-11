"""
Various functions to make it easier to build test models.
"""
from __future__ import print_function

import time
import numpy

from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.component import Component
from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.test.exec_comp_for_test import ExecComp4Test

class DynComp(Component):
    """
    A component with a settable number of params, outputs, and states.
    """
    def __init__(self, nparams, noutputs, nstates=0,
                 nl_sleep=0.001, ln_sleep=0.001,
                 var_factory=float, vf_args=()):
        super(DynComp, self).__init__()

        self.nl_sleep = nl_sleep
        self.ln_sleep = ln_sleep

        for i in range(nparams):
            self.add_param('p%d'%i, var_factory(*vf_args))

        for i in range(noutputs):
            self.add_output("o%d"%i, var_factory(*vf_args))

        for i in range(nstates):
            self.add_state("s%d"%i, var_factory(*vf_args))

    def solve_nonlinear(self, params, unknowns, resids):
        time.sleep(self.nl_sleep)

    def solve_linear(self, dumat, drmat, vois, mode=None):
        time.sleep(self.ln_sleep)

def make_subtree(parent, nsubgroups, levels,
                 ncomps, nparams, noutputs, nconns, var_factory=float):
    """Construct a system subtree under the given parent group."""

    if levels <= 0:
        return

    if levels == 1:  # add leaf nodes
        create_dyncomps(parent, ncomps, nparams, noutputs, nconns,
                        var_factory=var_factory)
    else:  # add more subgroup levels
        for i in range(nsubgroups):
            g = parent.add("G%d"%i, Group())
            make_subtree(g, nsubgroups, levels-1,
                         ncomps, nparams, noutputs, nconns,
                         var_factory=var_factory)

def create_dyncomps(parent, ncomps, nparams, noutputs, nconns,
                    var_factory=float):
    """Create a specified number of DynComps with a specified number
    of variables (nparams and noutputs), and add them to the given parent
    and add the number of specified connections.
    """
    for i in range(ncomps):
        parent.add("C%d" % i, DynComp(nparams, noutputs, var_factory=var_factory))

        if i > 0:
            for j in range(nconns):
                parent.connect("C%d.o%d" % (i-1,j), "C%d.p%d" % (i, j))


if __name__ == '__main__':
    import sys
    from openmdao.core.problem import Problem
    from openmdao.devtools.debug import stats
    vec_size = 100000
    num_comps = 50
    pts = 2

    if 'petsc' in sys.argv:
        from openmdao.core.petsc_impl import PetscImpl
        impl = PetscImpl
    else:
        from openmdao.core.basic_impl import BasicImpl
        impl = BasicImpl

    g = Group()
    p = Problem(impl=impl, root=g)

    if 'gmres' in sys.argv:
        from openmdao.solvers.scipy_gmres import ScipyGMRES
        p.root.ln_solver = ScipyGMRES()

    g.add("P", IndepVarComp('x', numpy.ones(vec_size)))

    p.driver.add_desvar("P.x")

    par = g.add("par", ParallelGroup())
    for pt in range(pts):
        ptname = "G%d"%pt
        ptg = par.add(ptname, Group())
        create_dyncomps(ptg, num_comps, 2, 2, 2,
                            var_factory=lambda: numpy.zeros(vec_size))
        g.connect("P.x", "par.%s.C0.p0" % ptname)

        cname = ptname + '.' + "C%d"%(num_comps-1)
        p.driver.add_objective("par.%s.o0" % cname)
        p.driver.add_constraint("par.%s.o1" % cname, lower=0.0)

    p.setup()
    p.run()
    #g.dump(verbose=True)
    #p.root.list_connections()

    stats(p)
