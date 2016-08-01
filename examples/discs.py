""" Position discs along a line so that they are not overlapping."""

from __future__ import print_function
from six.moves import range

import numpy as np

from openmdao.api import Problem, Group, ExecComp, IndepVarComp

class ExecComp2(ExecComp):
    """ Same as ExecComp except we count the number of times apply_linear is
    called in the class."""

    def __init__(self, exprs):
        super(ExecComp2, self).__init__(exprs)
        self.total_calls = 0

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                    mode):
        """ Override this just to count the total number of calls."""
        super(ExecComp2, self).apply_linear(params, unknowns, dparams, dunknowns, dresids,
                                            mode)
        self.total_calls += 1

if __name__ == '__main__':

    # So we compare the same starting locations.
    np.random.seed(123)

    diam = 1.0
    pin = 15.0
    n_disc = 7

    prob = Problem()
    prob.root = root = Group()

    from openmdao.api import pyOptSparseDriver
    driver = prob.driver = pyOptSparseDriver()
    driver.options['optimizer'] = 'SNOPT'
    driver.options['print_results'] = False

    # Note, active tolerance requires relevance reduction to work.
    root.ln_solver.options['single_voi_relevance_reduction'] = True

    # Also, need to be in adjoint
    root.ln_solver.options['mode'] = 'rev'

    obj_expr = 'obj = '
    sep = ''
    for i in range(n_disc):

        dist = "dist_%d" % i
        x1var = 'x_%d' % i

        # First disc is pinned
        if i == 0:
            root.add('p_%d' % i, IndepVarComp(x1var, pin), promotes=(x1var, ))

        # The rest are design variables for the optimizer.
        else:
            init_val = 5.0*np.random.random() - 5.0 + pin
            root.add('p_%d' % i, IndepVarComp(x1var, init_val), promotes=(x1var, ))
            driver.add_desvar(x1var)

        for j in range(i):

            x2var = 'x_%d' % j
            yvar = 'y_%d_%d' % (i, j)
            name = dist + "_%d" % j
            expr = '%s= (%s - %s)**2' % (yvar, x1var, x2var)
            root.add(name, ExecComp2(expr), promotes = (x1var, x2var, yvar))

            # Constraint (you can experiment with turning on/off the active_tol)
            #driver.add_constraint(yvar, lower=diam)
            driver.add_constraint(yvar, lower=diam, active_tol=diam*2.0)

            # This pair's contribution to objective
            obj_expr += sep + yvar
            sep = ' + '

    root.add('sum_dist', ExecComp(obj_expr), promotes=('*', ))
    driver.add_objective('obj')

    prob.setup()

    print("Initial Locations")
    for i in range(n_disc):
        xvar = 'x_%d' % i
        print(prob[xvar])

    prob.run()

    print("\nFinal Locations")
    for i in range(n_disc):
        xvar = 'x_%d' % i
        print(prob[xvar])

    total_apply = 0
    for syst in root.subsystems(recurse=True):
        if 'dist_' in syst.name:
            total_apply += syst.total_calls
    print("\ntotal apply_linear calls:", total_apply)

    pass

