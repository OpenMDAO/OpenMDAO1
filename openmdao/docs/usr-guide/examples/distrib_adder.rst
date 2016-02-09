.. index:: Distributed Component Example

Requirements
----------------------------------------
This example runs OpenMDAO in parallel which requires `petsc4py` and `mpi4py`.
You must have these packages installed in order to proceed.
To get these packages set up on Linux, see `MPI on Linux`_.
To get these packages set up on Windows, see `MPI on Windows`_.

.. _MPI on Linux: ../../getting-started/mpi_linux.html

.. _MPI on Windows: ../../getting-started/mpi_windows.html


Distributed Components
------------------------
OpenMDAO can work with components that are actually distributed themselves.
This is useful for dealing with complex tools, like PDE solver (CFD or FEA).
But it can also be used to speed up any calculations you're implementing yourself
directly in OpenMDAO using our MPI-based parallel data passing.

Why should you use OpenMDAO to build your own distributed components? Because
OpenMDAO lets you build distributed components without writing any significant
MPI code yourself. Here is a simple example where we break up the job of adding
a value to a large float array (1,000,000 elements).


.. testcode :: dist_adder

    from __future__ import print_function
    import numpy as np
    from six.moves import range

    from openmdao.api import Component
    from openmdao.util.array_util import evenly_distrib_idxs

    class DistributedAdder(Component):
        """
        Distributes the work of adding 10 to every item in the param vector
        """

        def __init__(self, size=100):
            super(DistributedAdder, self).__init__()

            self.local_size = self.size = int(size)

            #NOTE: we declare the variables at full size so that the component will work in serial too
            self.add_param('x', shape=size)
            self.add_output('y', shape=size)

        def get_req_procs(self):
            """
            min/max number of procs that this component can use
            """
            return (1,self.size)

        def setup_distrib(self):
            """
            specify the local sizes of the variables and which specific indices this specific
            distributed component will handle. Indices do NOT need to be sequential or
            contiguous!
            """

            comm = self.comm
            rank = comm.rank

            # NOTE: evenly_distrib_idxs is a helper function to split the array
            #       up as evenly as possible
            sizes, offsets = evenly_distrib_idxs(comm.size, self.size)
            local_size, local_offset = sizes[rank], offsets[rank]
            self.local_size = int(local_size)

            start = local_offset
            end = local_offset + local_size

            self.set_var_indices('x', val=np.zeros(local_size, float),
                src_indices=np.arange(start, end, dtype=int))
            self.set_var_indices('y', val=np.zeros(local_size, float),
                src_indices=np.arange(start, end, dtype=int))

        def solve_nonlinear(self, params, unknowns, resids):

            #NOTE: Each process will get just its local part of the vector
            print('process {0:d}: {1}'.format(self.comm.rank, params['x'].shape))

            unknowns['y'] = params['x'] + 10


    class Summer(Component):
        """
        Agreggation component that collects all the values from the distributed
        vector addition and computes a total
        """

        def __init__(self, size=100):
            super(Summer, self).__init__()

            #NOTE: this component depends on the full y array, so OpenMDAO
            #      will automatically gather all the values for it
            self.add_param('y', val=np.zeros(size))
            self.add_output('sum', shape=1)

        def solve_nonlinear(self, params, unknowns, resids):

            unknowns['sum'] = np.sum(params['y'])

The distributed component magic happens in the `setup_distrib` method of
the `DistributedAdder` class. This is where we tell the framework how to split
up the the big array into smaller chunks handled separately by each distributed
process. In this case, we just split the array up one chuck at a time in order
as we go from process to process. But OpenMDAO does not require that the `src_indices`
be ordered or sequential!

.. note::

    Only the `DistributedAdder` class is a distributed component. The `Summer`
    is class is a normal component that aggregates the whole array to sum it up.

Next we'll use these components to build an actual distributed model:


.. testcode :: dist_adder

    import time

    from openmdao.api import Problem, Group, IndepVarComp

    from openmdao.core.mpi_wrap import MPI

    if MPI:
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl
    else:
        # if you didn't use `mpirun`, then use the numpy data passing
        from openmdao.api import BasicImpl as impl

    #how many items in the array
    size = 1000000

    prob = Problem(impl=impl)
    prob.root = Group()

    prob.root.add('des_vars', IndepVarComp('x', np.ones(size)), promotes=['x'])
    prob.root.add('plus', DistributedAdder(size), promotes=['x', 'y'])
    prob.root.add('summer', Summer(size), promotes=['y', 'sum'])

    prob.setup(check=False)

    prob['x'] = np.ones(size)

    st = time.time()
    prob.run()

    #only print from the rank 0 process
    if prob.root.comm.rank == 0:
        print("run time:", time.time() - st)
        #expected answer is 11
        print("answer: ", prob['sum']/size)

.. testoutput:: dist_adder
    :hide:
    :options: +ELLIPSIS

    process 0: (1000000...
    run time: ...
    answer:  11.0


You can run this model in either serial or parallel, depending on how you call the script.
Lets say you put the above code into a python script called *dist_adder.py*. Then to run it in
serial you would call it just like any other python script:

::

    python dist_adder.py


In that case, you'll expect to see some output that looks like this:

::

    process 0: (30000000,)
    run time: 1.76785802841
    answer:  11.0


To run the model in parallel you need to have an MPI library (e.g. OpenMPI),
mpi4py, PETSc, and petsc4py installed. Then you can call the script like this:

::

    mpirun -n 2 python dist_adder.py


And you can expect to see some output as follows:

::

    process 0: (15000000,)
    process 1: (15000000,)
    run time: 1.00080680847
    answer:  11.0

With two processes running, you get a decent speed up. You can see that each process took
half the array. Why don't we get a full 2x speedup? Two reasons. The first, and more
significant factor is that we don't have a fully parallel model. The `DistributedAdder`
component is distributed, but the `Summer` component is not. This introduces a bottleneck
because we have to wait for the serial operation to complete.
