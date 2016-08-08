.. index:: Simultaneous ANalysis and Design (SAND) architecture on OpenMDAO with Sellar Problem

Simultaneous ANalysis and Design (SAND) architecture on OpenMDAO with Sellar Problem
------------------------------------------------------------------------------------

OpenMDAO allows you to do optimization on problems using a variety of MDAO archectures. One example of that is `Simultaneous ANalysis and Design (SAND) architecture <http://arc.aiaa.org/doi/abs/10.2514/3.9043>`_.

In SAND, the optimizer minimizes the problem by varying the design variables simultaneously with the coupling variables to achieve feasibility and drive the residual constraint to zero. This means the residual needs to be expressed explicitly so we don't need any implicit components or a solver. The optimizer does it all.

Here is the code for solving the Sellar problem using the SAND architecture.

.. testcode:: sand_example

    from __future__ import print_function

    import time

    import numpy as np

    from openmdao.api import Component, Group, Problem, IndepVarComp, ExecComp, NLGaussSeidel, \
        ScipyGMRES, ScipyOptimizer


    class SellarDis1(Component):

        def __init__(self):
            super(SellarDis1, self).__init__()

            self.add_param('z', val=np.zeros(2))
            self.add_param('x', val=0.0)
            self.add_param('y2', val=1.0)
            self.add_param('y1', val=1.0)

            self.add_output('resid1', val=1.0)

        def solve_nonlinear(self, params, unknowns, resids):

            z1 = params['z'][0]
            z2 = params['z'][1]
            x1 = params['x']
            y2 = params['y2']
            y1 = params['y1']

            unknowns['resid1'] = z1**2 + z2 + x1 - 0.2*y2 - y1

        def linearize(self, params, unknowns, resids):
            J = {}

            J['resid1','y1'] = -1.0
            J['resid1','y2'] = -0.2
            J['resid1','z'] = np.array([[2*params['z'][0], 1.0]])
            J['resid1','x'] = 1.0

            return J


    class SellarDis2(Component):

        def __init__(self):
            super(SellarDis2, self).__init__()

            self.add_param('z', val=np.zeros(2))
            self.add_param('y1', val=1.0)
            self.add_param('y2', val=1.0)

            self.add_output('resid2', val=1.0)

        def solve_nonlinear(self, params, unknowns, resids):

            z1 = params['z'][0]
            z2 = params['z'][1]
            y1 = params['y1']
            y1 = abs(y1)
            y2 = params['y2']

            unknowns['resid2'] = y1**.5 + z1 + z2 - y2

        def linearize(self, params, unknowns, resids):
            J = {}

            J['resid2', 'y2'] = -1.0
            J['resid2', 'y1'] = 0.5*params['y1']**-0.5
            J['resid2', 'z'] = np.array([[1.0, 1.0]])

            return J

    class SellarSAND(Group):

        def __init__(self):
            super(SellarSAND, self).__init__()

            self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
            self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
            self.add('py1', IndepVarComp('y1', 1.0), promotes=['y1'])
            self.add('py2', IndepVarComp('y2', 1.0), promotes=['y2'])

            self.add('d1', SellarDis1(), promotes=['resid1', 'z', 'x', 'y1', 'y2'])
            self.add('d2', SellarDis2(), promotes=['resid2','z', 'y1', 'y2'])

            self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                         z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                     promotes=['obj', 'z', 'x', 'y1', 'y2'])

            self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
            self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

    top = Problem()
    top.root = SellarSAND()

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'
    top.driver.options['tol'] = 1.0e-12

    top.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),upper=np.array([10.0, 10.0]))
    top.driver.add_desvar('x', lower=0.0, upper=10.0)
    top.driver.add_desvar('y1', lower=-10.0, upper=10.0)
    top.driver.add_desvar('y2', lower=-10.0, upper=10.0)

    top.driver.add_objective('obj')
    top.driver.add_constraint('con1', upper=0.0)
    top.driver.add_constraint('con2', upper=0.0)
    top.driver.add_constraint('resid1', equals=0.0)
    top.driver.add_constraint('resid2', equals=0.0)

    top.setup()
    tt = time.time()
    top.run()


    print("\n")
    print( "Minimum found at (z1,z2,x) = (%3.4f, %3.4f, %3.4f)" % (top['z'][0], \
                                             top['z'][1], \
                                             top['x']))
    print("Coupling vars: %3.4f, %3.4f" % (top['d1.y1'], top['d1.y2']))
    print("Minimum objective: %3.4f" % top['obj'])


The output should look like this:

.. testoutput:: sand_example
   :options: +ELLIPSIS

   ...
   Minimum found at (z1,z2,x) = (1.9776, ...0.0000, 0.0000)
   Coupling vars: 3.1600, 3.7553
   Minimum objective: 3.1834


.. note::

    You might ask what would be different about the implementation if you used `AAO (All At Once) <https://www.researchgate.net/profile/J_Dennis/publication/2649710_Problem_Formulation_for_Multidisciplinary_Optimization/links/09e4150ca739b888af000000.pdf>`_ instead of SAND for this problem. They are similar because both AAO and SAND architectures directly deal with state variables and residuals. In other architectures, an additional solver needs to be added to drive the disciplines to consistency.

    For AAO, you would make separate components to house the residuals, which are kept in the data transfer between d1 and d2, and the code for the disciplines d1 and d2 is the same as in the MDF examples. So, the differences are subtle but amount to a little more storage.

