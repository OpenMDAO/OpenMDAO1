.. index:: Check Partial Differences Example

Check Partial Differences
-------------------------

Simple Example of Using check_partial_derivatives
=================================================

OpenMDAO provides a way to check to see if the partial derivative calculation of `Components` via the `jacobian` method of your `Component` is working correctly. 

Here is a simple example of some code that checks partial derivative calculations for one of the simple `Components` provided with OpenMDAO, `SimpleArrayComp`.

`Problem` has a method, `check_partial_derivatives`, that the user can call to check the partial derivatives. It has one optional argument, `out_stream`, which lets you define where the results of the check are written to. The default is `sys.stdout`. This what this example uses. 

.. testcode:: check_partial_derivatives_example

    from six import iteritems
    import numpy as np

    from openmdao.components.param_comp import ParamComp
    from openmdao.core.group import Group
    from openmdao.core.problem import Problem
    from openmdao.test.simple_comps import SimpleArrayComp
    from openmdao.test.util import assert_rel_error

    prob = Problem()
    prob.root = Group()
    prob.root.add('comp', SimpleArrayComp())
    prob.root.add('p1', ParamComp('x', np.ones([2])))

    prob.root.connect('p1.x', 'comp.x')

    prob.setup(check=False)
    prob.run()

    data = prob.check_partial_derivatives()


.. testoutput:: check_partial_derivatives_example
   :options: +ELLIPSIS

   ...
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

Please note that check_partial_derivatives only does its checks for `Components` that compute derivatives by providing a `jacobian` method.

Return Value of check_partial_derivatives
=================================================

The method check_partial_derivatives returns a dict of dicts of dicts. The keys of the nested dicts are:

=========================================================          ======================
Dict Key                                                           Example
=========================================================          ======================
Component Name                                                     'subcomp'
A tuple of strings indicating the (output, input)                  ('y1', 'x2')
One of ['rel error', 'abs error', 'magnitude', 'fdstep']           'rel error'
=========================================================          ======================

The type of the values depends on the final key.

=========================================================          ======================
Third Key                                                          Type of value
=========================================================          ======================
'rel error', 'abs error', 'magnitude'                              A tuple containing norms for forward - fd, adjoint - fd, forward - adjoint using the best case fdstep
'J_fd', 'J_fwd', 'J_rev'                                           A numpy array representing the computed Jacobian for the three different methods of computation
=========================================================          ======================


