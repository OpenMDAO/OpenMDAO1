.. index:: Check Partial Differences Example

Error Checking Partial Derivatives
------------------------------------

Simple Example Using Problem.check_partial_derivatives
======================================================

OpenMDAO provides a way for a `Component` developer to verify that their
partial derivatives, for each individual component, are correct. 

`Problem` has a method, `check_partial_derivatives`, that checks partial
derivatives comprehensively for all `Components` in your model (as long as
you didn't set *fd_options['force_fd'] = True*). To do this check, the framework
uses compares the analytic result against a finite difference result. This means
that the `check_partial_derivatives` function can be quite computationally expensive.
So use it to check your work, but don't leave the call in your production run scripts.


.. note::

  `check_partial_derivatives` has one optional argument, `out_stream`, which lets
  you define where the results of the check are written to. The default is
  `sys.stdout`. This example explicitly sets the value of out_stream to
  `sys.stdout` to make our automated doc tests work correctly. You would only
  set this argument if you wanted to pipe it to a file or some other stream.

Here is example code for a model that consists of a single `Component`,
`SimpleArrayComp. After setting up the model, it runs `check_partial_derivatives` on the `Problem`.

.. testcode:: check_partial_derivatives_example

   import numpy as np
   import sys

   from openmdao.components import ParamComp
   from openmdao.core import Problem, Group
   from openmdao.test.simple_comps import SimpleArrayComp

   prob = Problem()
   prob.root = Group()
   prob.root.add('comp', SimpleArrayComp())
   prob.root.add('p1', ParamComp('x', np.ones([2])))

   prob.root.connect('p1.x', 'comp.x')

   prob.setup(check=False)
   prob.run()

   data = prob.check_partial_derivatives(out_stream=sys.stdout)

This code generates output that looks like this:

.. testoutput:: check_partial_derivatives_example
   :options: +REPORT_NDIFF

   Partial Derivatives Check

   -------------------
   Component: 'comp'
   -------------------
     comp: 'y' wrt 'x'

       Forward Magnitude : 9.327379e+00
       Reverse Magnitude : 9.327379e+00
            Fd Magnitude : 9.327379e+00

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

       Raw FD Derivative (Jfor)

   [[ 2.  7.]
    [ 5. -3.]]


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
'J_fd', 'J_fwd', 'J_rev'                                           A numpy array representing the computed Jacobian for the three different methods of computation
=========================================================          ======================
