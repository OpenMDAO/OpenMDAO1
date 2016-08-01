""" Position discs along a line so that they are not overlapping."""

from __future__ import print_function
from six.moves import range

import numpy as np

from openmdao.api import Problem, Group, ExecComp, IndepVarComp

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
            root.add(name, ExecComp(expr), promotes = (x1var, x2var, yvar))

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

    # Run with profiling turned on so that we can count the total derivative
    # component calls.
    from openmdao.api import profile, Component
    profile.setup(prob, methods={'apply_linear' : (Component, )})
    profile.start()    
    prob.run()
    profile.stop()

    print("\nFinal Locations")
    for i in range(n_disc):
        xvar = 'x_%d' % i
        print(prob[xvar])

