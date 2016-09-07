.. index:: Inspecting Solver Convergence

Inspecting Solver Convergence
------------------------------

OpenMDAO offers a lot of flexibility in selecting and customizing the set of
linear solvers, nonlinear solvers, and preconditioners for solving your
problems. Getting all of your solver settings just right can be tricky
though. To help you with this, we've given every solver an `iprint` option to
control the output of useful convergence information.

::

    lp_spool.ln_solver.preconditioner.options['iprint'] = 2

There are 4 levels of output:

=======       ==========================================================================
iprint        What it does
=======       ==========================================================================
-1            Suppress all output, including convergence failures (not recommended.)
0             Only print convergence failures.
1             Also print the number of iterations taken at each solver level.
2             Also print the norm of the residual at each iteration.
=======       ==========================================================================

While this gives you fine control, when debugging a large model, it is more
convenient to set all the iprints globally. The `problem` has a function
`print_all_convergence` that can set all of the solver iprint settings to the
desired value (the default with no argument is 2.) You can also optionally
give it a `depth` to tell it only go the given number of levels down the
hierarchy.

For example, if we look at the Sellar problem where the disciplines reside in
a sub-group called "cycle", and the Newton solver is at the top, we can look
at just the root level.

::

    prob.print_all_convergence(level=2, depth=0)

When it runs:

::

   [root] NL: NEWTON   0 | 2.25451411 1
      [root] LN: LN_GS   1 | 0 0
      [root] LN: LN_GS   1 | Converged in 1 iterations
   [root] NL: NEWTON   1 | 0.000869924533 0.000385858989 (3.84419971931)
      [root] LN: LN_GS   1 | 0 0
      [root] LN: LN_GS   1 | Converged in 1 iterations
   [root] NL: NEWTON   2 | 1.40587986e-10 6.23584414e-11 (0.00148234770486)
   [root] NL: NEWTON   2 | Converged in 2 iterations

We see here the Newton solver is converging in 2 iterations. The format for
the residual line is as follows:

::

  [pathname] SOLVER_TYPE: SOLVER_STRING   iteration_number | abs_norm_of_resids  rel_norm_of_resids (norm_of_unknowns)

The relative norm of the residuals is normalized by the initial value of the residual at the start of iteration.

If we increase the depth to one, we will see printout from the GMRES solver in `root.cycle` as well.

::

    prob.print_all_convergence(level=2, depth=1)

::

   [root] NL: NEWTON   0 | 2.25451411 1
         [root.cycle] LN: GMRES   1 | 0.0944068636 1
         [root.cycle] LN: GMRES   2 | 0 0
         [root.cycle] LN: GMRES   2 | Converged in 2 iterations
      [root] LN: LN_GS   1 | 0 0
      [root] LN: LN_GS   1 | Converged in 1 iterations
   [root] NL: NEWTON   1 | 0.000869924533 0.000385858989 (3.84419971931)
         [root.cycle] LN: GMRES   1 | 0.0983660396 1
         [root.cycle] LN: GMRES   2 | 0 0
         [root.cycle] LN: GMRES   2 | Converged in 2 iterations
      [root] LN: LN_GS   1 | 0 0
      [root] LN: LN_GS   1 | Converged in 1 iterations
   [root] NL: NEWTON   2 | 1.40587986e-10 6.23584414e-11 (0.00148234770486)
   [root] NL: NEWTON   2 | Converged in 2 iterations

Some models can get quite complicated, so it is useful to have precise
control over the messages that are printed.

.. tags:: Solver, Output, Examples
