.. index:: active_tol Example

Improving Performance with active_tol
--------------------------------------

.. note::

  This is an experimental tweak to performance. Using it requires a high
  level of understanding of optimization and the structure of your problem.

Some optimizers use an active set method, whereby their constraints are marked
as active or inactive depending on their proximity to the feasible region. If
a constraint is far enough from the feasible region, it is essentially
redundant to the set, so an optimizer can ignore it and mark it as inactive.
When this occurs, it no longer needs functional or gradient evaluations for
that constraint.

Since gradient calculation can be a major source of computation, some
performance can be gained if we can omit calculating the derivatives for
inactive constraints. The ideal way to do this would be to gain access to the
optimizer internals and promote that information to OpenMDAO. This was not
conveniently available, so we have instead provided an argument to
`add_constraint` that lets you specify how far from your constraint boundary
you need to go before you consider it to be inactive. This is most easily
used on geometric problems where you can clearly visualize when a constraint
is completely occluded by other constraints.

The following restrictions apply to using the active tolerance.

- Optimizer must support active set methods (only SNOPT in ``pyoptsparse`` at present)
- Only works for adjoint mode, so `mode` in `root` linear solver must be set to "rev"
- Relevance reduction must be enabled ("single_voi_relevance_reduction" set to True in root linear solver)

Let's consider a problem where we have 7 discs with a 1 cm diameter, and we
would like to arrange them on a line as closely together as possible without
overlapping. We can do this by minimizing the sum of the distances between
each disc and its 6 neighbors. Now, we don't want any of our discs to
overlap, so we need to constrain each of them so that the distance to every
other disc is greater than 1 diameter.

The code for this is below. We used an `ExecComp` because the equation for
distance is simple to write. To make a point about derivative calculation, we
created our own `ExecComp2` that inherits from `ExecComp` and increments a
counter every time `apply_linear` (which is the workhorse derivatives
function) is called.

.. testcode:: active_tol_example

    from __future__ import print_function
    from six.moves import range

    import numpy as np

    from openmdao.api import Problem, Group, pyOptSparseDriver, ExecComp, IndepVarComp

    class ExecComp2(ExecComp):
        """ Same as ExecComp except we count the number of times apply_linear is
        called in the class."""

        def __init__(self, exprs):
            super(ExecComp2, self).__init__(exprs)
            self.total_calls = 0

        def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
            """ Override this just to count the total number of calls."""

            super(ExecComp2, self).apply_linear(params, unknowns, dparams, dunknowns, dresids, mode)
            self.total_calls += 1

    if __name__ == '__main__':

        # So we compare the same starting locations.
        np.random.seed(123)

        diam = 1.0
        pin = 15.0
        n_disc = 7

        prob = Problem()
        prob.root = root = Group()

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


Note that we defined the variable "n_disc" for the number of discs, so
component and variable names such as "dist_1_2" and "y_2" had to be created
with some string operations.

::

        Initial Locations
        15.0
        13.482345928
        11.4306966748
        11.1342572678
        12.7565738454
        13.5973448489
        12.1155323006

        Final Locations
        15.0
        12.9999999413
        9.99999993369
        8.99999991376
        11.9999999405
        13.9999999687
        10.9999999358

        total apply_linear calls: 177

Note that this lines our discs up neatly so that they are touching each other
with their centers ranging from 9 to 15. Note that we chose a distance of 2.0
times the disc diameter as our "active_tol". When we do this, and have 3
discs in a row, then the distance constraint between disc1 and disc3 is
inactive, so its gradient is not calculated.

So, did our active tolerance really do anything? If we turn it off and check
out the number of `apply_linear` calls:

::

        Initial Locations
        15.0
        13.482345928
        11.4306966748
        11.1342572678
        12.7565738454
        13.5973448489
        12.1155323006

        Final Locations
        15.0
        12.9999998135
        9.99999980586
        8.99999978593
        11.9999998126
        13.9999999687
        10.999999808

        total apply_linear calls: 336

So almost half of the `apply_linear` calls turn out to be unneeded.

This would normally be a pretty bad case to run in adjoint mode because the
number of constraints varies with n_disc by (n_disc**2)/2 - n_disc, while the
number of design variables only varies by n_disc. However, a good choice for
"active_tol" cuts out a significant number of the extra gradient
calculations.
