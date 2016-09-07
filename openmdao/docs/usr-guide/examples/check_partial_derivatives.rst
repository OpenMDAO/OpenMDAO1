.. index:: Check Partial Differences Example

Error Checking Partial Derivatives
------------------------------------

Simple Example Using Problem.check_partial_derivatives
======================================================

OpenMDAO provides a way for a `Component` developer to verify that their
partial derivatives, for each individual component, are correct.

`Problem` has a method, `check_partial_derivatives`, that checks partial
derivatives comprehensively for all `Components` in your model (as long as
you didn't set *deriv_options['type'] = 'fd' on that component*). To do this
check, the framework compares the analytic result against a finite
difference result. This means that the `check_partial_derivatives` function
can be quite computationally expensive. So use it to check your work, but
don't leave the call in your production run scripts.


.. note::

  `check_partial_derivatives` has three optional arguments,

  **out_stream**, lets you define where the results of the check are written
  to. The default is `sys.stdout`. This example explicitly sets the value of
  out_stream to `sys.stdout` to make our automated doc tests work correctly.
  You would only set this argument if you wanted to pipe it to a file or some
  other stream.

  **comps** is a list of component pathnames, which you can specify if you
  want to only check the gradients on a subset of the components as a time
  saver while debugging

  **compact_print** can be set to True for more compact results (essentially
  each input/output pair is summarized on one line.) You should be careful
  with this, particuarly for arrays where an important difference in a single
  array element might not be noticed when looking purely at the norm of the
  difference between the arrays.

Here is example code for a model that consists of a single `Component`,
`SimpleArrayComp. After setting up the model, it runs `check_partial_derivatives` on the `Problem`.

.. testcode:: check_partial_derivatives_example

   import numpy as np
   import sys

   from openmdao.api import IndepVarComp, Problem, Group
   from openmdao.test.simple_comps import SimpleArrayComp

   prob = Problem()
   prob.root = Group()
   prob.root.add('comp', SimpleArrayComp())
   prob.root.add('p1', IndepVarComp('x', np.ones([2])))

   prob.root.connect('p1.x', 'comp.x')

   prob.setup(check=False)
   prob.run()

   data = prob.check_partial_derivatives(out_stream=sys.stdout)

This code generates output that looks like this:

.. testoutput:: check_partial_derivatives_example
   :options: +ELLIPSIS, +REPORT_NDIFF

   Partial Derivatives Check

   -------------------
   Component: 'comp'
   -------------------
     comp: 'y' wrt 'x'

       Forward Magnitude : 9.327379e+00
       Reverse Magnitude : 9.327379e+00
            Fd Magnitude : 9.327379e+00 (fd:forward)

       Absolute Error (Jfor - Jfd) : 1.769949e-09
       Absolute Error (Jrev - Jfd) : 1.769949e-09
       Absolute Error (Jfor - Jrev): 0.000000e+00

       Relative Error (Jfor - Jfd) : 1.897585e-10
       Relative Error (Jrev - Jfd) : 1.897585e-10
       Relative Error (Jfor - Jrev): 0.000000e+00

       Raw Forward Derivative (Jfor)

   [[ 2.  7.]
    [ 5. -3.]]

       Raw Reverse Derivative (Jrev)

   [[ 2.  7.]
    [ 5. -3.]]

       Raw FD Derivative (Jfd)

   [[ 2.  7.]
    [ 5. -3.]]
   ...

You can control the finite difference used in the check by setting some
additional options in the deriv_options dictionary.

.. testcode:: fd_example
    :hide:

    # Setup and run the model.
    import numpy as np
    import sys

    from openmdao.api import IndepVarComp, Problem, Group
    from openmdao.test.simple_comps import SimpleArrayComp

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', SimpleArrayComp())
    prob.root.add('p1', IndepVarComp('x', np.ones([2])))

    prob.root.connect('p1.x', 'comp.x')

.. testcode:: fd_example

    # Set form to 'central', 'forward', or 'reverse'
    prob.root.comp.deriv_options['check_form'] = 'central'

    # Can check with 'fd' (finite difference) or 'cs' (complex step)
    prob.root.comp.deriv_options['check_type'] = 'fd'

    # Can be 'relative' or 'absolute'
    prob.root.comp.deriv_options['check_step_calc'] = 'relative'

    # Set a step size
    prob.root.comp.deriv_options['check_step_size'] = 1.0e-5


You can also use the `check_partial_derivatives` method to compare two
different kinds of finite difference executions (e.g., forward and central) with
each other or to complex step. Do this by setting 'type' in your component to
'fd' or 'cs'. The options for this second check are the regular 'fd' options
'step_size', 'form', 'type', and 'step_calc'.

.. testcode:: check_partial_derivatives_example2

   import numpy as np
   import sys

   from openmdao.api import IndepVarComp, Problem, Group
   from openmdao.test.simple_comps import SimpleArrayComp

   prob = Problem()
   prob.root = Group()
   prob.root.add('comp', SimpleArrayComp())
   prob.root.add('p1', IndepVarComp('x', np.ones([2])))

   prob.root.connect('p1.x', 'comp.x')

   # Turn on fd in comp using forward difference
   prob.root.comp.deriv_options['type'] = 'fd'
   prob.root.comp.deriv_options['form'] = 'forward'

   # Compare the fd with central difference
   prob.root.comp.deriv_options['check_form'] = 'central'

   prob.setup(check=False)
   prob.run()

   data = prob.check_partial_derivatives(out_stream=sys.stdout)

This code generates output that looks like this:

.. testoutput:: check_partial_derivatives_example2
   :options: +ELLIPSIS, +REPORT_NDIFF

   Partial Derivatives Check

   -------------------
   Component: 'comp'
   -------------------
     comp: 'y' wrt 'x'

       Fwd/Rev Magnitude : Component supplies no analytic derivatives.
            Fd Magnitude : 9.327379e+00 (fd:central)
           Fd2 Magnitude : 9.327379e+00 (fd:forward)

       Absolute Error (Jfd2 - Jfd): 2.551098e-09

       Relative Error (Jfd2 - Jfd) : 2.735064e-10

       Raw FD Derivative (Jfd)

   [[ 2.  7.]
    [ 5. -3.]]

       Raw FD Check Derivative (Jfd2)

   [[ 2.  7.]
    [ 5. -3.]]
   ...

Return Value of check_partial_derivatives
=================================================

The method check_partial_derivatives returns a dict of dicts of dicts with
comprehensive information about the check of the partial derivatives. You can use
this data to write scripts to interact with the derivatives check information if
you want.

The keys of the nested dicts are:

===========================================================          ======================
Dict Key                                                             Example
===========================================================          ======================
Component name                                                       'subcomp'
A tuple of strings indicating the (output, input) variables          ('y1', 'x2')
One of ['rel error', 'abs error', 'magnitude', 'fdstep']             'rel error'
===========================================================          ======================

The type of the values depends on key of the innermost dict.

=========================================================          ======================
Key of Innermost Dict                                              Type of value
=========================================================          ======================
'rel error', 'abs error', 'magnitude'                              A tuple containing norms for (forward - finite differences), ( adjoint - finite differences), (forward - adjoint) using the best case fdstep
'J_fd', 'J_fwd', 'J_rev', 'J_fd2'*                                 A numpy array representing the computed Jacobian for the three different methods of computation
=========================================================          ======================

.. tags:: Derivatives, Examples
