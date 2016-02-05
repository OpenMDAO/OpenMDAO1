.. index:: Serial and Parallel Multi-Point Example

Requirements
----------------------------------------
This example runs OpenMDAO in parallel which requires `petsc4py` and `mpi4py`.
You must have these packages installed in order to proceed.
To get these packages set up on Linux, see `MPI on Linux`_.
To get these packages set up on Windows, see `MPI on Windows`_.

.. _MPI on Linux: ../../getting-started/mpi_linux.html

.. _MPI on Windows: ../../getting-started/mpi_windows.html


Setting Up Serial Multi-Point Problems
----------------------------------------
A multi-point problem is when you want to analyze a single design at a number
of different conditions. For example, you might model aircraft performance at
five different flight conditions or predict solar power generation at ten
different times of the year. One approach to capture this kind of problem
structure is to define a `Group` that models your design, then stamp out as many
copies as you need.  This approach can use a large amount of memory, depending
on the size of your model and the number of copies you make, but it has the
advantage of supporting the calculation of derivatives across the entire
multi-point group.

.. testcode:: serial_multi_point

    from __future__ import print_function
    from six.moves import range
    import time
    import numpy as np

    from openmdao.api import Component, Group, ParallelGroup, IndepVarComp, ExecComp


    class Plus(Component):
        """
        adder: float
            value that is added to every element of the x array parameter
        """

        def __init__(self, adder):
            super(Plus, self).__init__()
            self.add_param('x', 0.0)
            self.add_output('f1', shape=1)
            self.adder = float(adder)

        def solve_nonlinear(self, params, unknowns, resids):
            unknowns['f1'] = params['x'] + self.adder

            #sleep to slow things down a bit, to show the parallelism better
            time.sleep(.1)

    class Times(Component):
        """
        scalar: float
            every element of the x array parameter is multiplied by this value
        """

        def __init__(self, scalar):
            super(Times, self).__init__()
            self.add_param('f1', 0.0)
            self.add_output('f2', shape=1)
            self.scalar = float(scalar)

        def solve_nonlinear(self, params, unknowns, resids):
            unknowns['f2'] = params['f1'] * self.scalar

    class Point(Group):
        """
        Single point combining Plus and Times. Multiple copies will be made for
        a multi-point problem
        """

        def __init__(self, adder, scalar):
            super(Point, self).__init__()

            self.add('plus', Plus(adder), promotes=['x', 'f1'])
            self.add('times', Times(scalar), promotes=['f1', 'f2'])


    class Summer(Component):
        """
        Aggregating component that takes all the multi-point values
        and adds them together
        """

        def __init__(self, size):
            super(Summer, self).__init__()
            self.size = size
            self.vars = []
            for i in range(size):
                v_name = 'f2_%d'%i
                self.add_param(v_name, 0.)
                self.vars.append(v_name)

            self.add_output('total', shape=1)

        def solve_nonlinear(self, params, unknowns, resids):
            tot = 0
            for v_name in self.vars:
                tot += params[v_name]
            unknowns['total'] = tot

    class ParallelMultiPoint(Group):

        def __init__(self, adders, scalars):
            super(ParallelMultiPoint, self).__init__()

            size = len(adders)
            self.add('desvar', IndepVarComp('x', val=np.zeros(size)),
                                            promotes=['x'])

            self.add('aggregate', Summer(size))

            # a ParallelGroup works just like a Group if it's run in serial,
            # so using a ParallelGroup here will make our ParallelMultipoint
            # class work well in serial and under MPI.
            pg = self.add('multi_point', ParallelGroup())

            #This is where you stamp out all the points you need
            for i,(a,s) in enumerate(zip(adders, scalars)):
                c_name = 'p%d'%i
                pg.add(c_name, Point(a,s))
                self.connect('x', 'multi_point.%s.x'%c_name, src_indices=[i])
                self.connect('multi_point.%s.f2'%c_name,'aggregate.f2_%d'%i)


    from openmdao.api import Problem


    prob = Problem()

    size = 10 #number of points

    adders = np.arange(size)/10.
    scalars = np.arange(size, 2*size)/10.

    prob.root = ParallelMultiPoint(adders, scalars)

    prob.setup()

    st = time.time()

    prob['x'] = np.ones(size) * 0.7
    st = time.time()
    print("run started")
    prob.run()
    print("run finished", time.time() - st)

    print(prob['aggregate.total'])



If you run this script, you should see output that looks like this:

.. testoutput:: serial_multi_point
    :hide:
    :options: +ELLIPSIS

    run started
    run finished ...
    17.5

::

    ##############################################
    Setup: Checking for potential issues...

    No recorders have been specified, so no data will be saved.

    Found ParallelGroup 'multi_point', but not running under MPI.

    Setup: Check complete.
    ##############################################

    run started
    run finished 1.03730106354
    17.5


Running Multi-Point in Parallel
-------------------------------

In many multi-point problems, all of the points can be run independently of
each other, which provides an opportunity to run things in parallel. Your serial
multi-point problem needs only a few minor modifications in order to run in parallel.

.. note::

     You'll need to make sure you have mpi, mpi4py, petsc, and petsc4py installed
     in order to do anything in parallel.

All of the changes you're going to make are in the run-script itself.
No changes are needed to the `Component` or `Group` classes.
You'll need to import the PETSc based data passing implementation,
and then to avoid getting a lot of extra print-out use a small
helper function that only prints on the rank 0 processor.
We also turned off the check-setup just to avoid getting
lots of extra output to the screen.

.. code-block:: python


    if __name__ == "__main__":
        from openmdao.api import Problem

        from openmdao.core.mpi_wrap import MPI

        if MPI: # pragma: no cover
            # if you called this script with 'mpirun', then use the
            # petsc data passing
            from openmdao.core.petsc_impl import PetscImpl as impl
        else:
            # if you didn't use `mpirun`, then use the numpy data passing
            from openmdao.api import BasicImpl as impl

        def mpi_print(prob, *args):
            """ helper function to only print on rank 0"""
            if prob.root.comm.rank == 0:
                print(*args)

        prob = Problem(impl=impl) #set the implementation

        size = 10 #number of points

        adders = np.arange(size)/10.
        scalars = np.arange(size, 2*size)/10.

        prob.root = ParallelMultiPoint(adders, scalars)

        #turning off setup checking to avoid getting 10 sets of printouts to the screen
        prob.setup(check=False)

        st = time.time()

        prob['x'] = np.ones(size) * 0.7
        st = time.time()
        mpi_print(prob, "run started")
        prob.run()
        mpi_print(prob, "run finished", time.time() - st)

        mpi_print(prob, prob['aggregate.total'])


You can save the new run-script to a second file, called
*parallel_multi_point.py* Then you run this code,
and you should see a significant reduction in the run-time.


::

    mpirun -n 10 python parallel_multi_point.py

We have to allocate 10 processes, because we have 10 points in `ParallelGroup`.

::

    run started
    run finished 0.14165687561
    17.5
