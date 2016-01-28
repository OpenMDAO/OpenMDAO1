.. _MPI on Linux:

MPI on Linux
============

This document provides the setup and usage of MPI (Message Passing Interface) in
OpenMDAO on Linux. We start with installing the necessary packages and test them
to make sure they work. Then we will look at a simple example of how to take
advantage of MPI in OpenMDAO.


Setup
------

The first package that must be installed is `mpi4py` which provides Python
bindings for MPI. This package requires that your system have an implementation
of MPI installed.  A quick way to check is to see if you can execute `mpirun`
or `mpiexec` from your path.  You will also need to make sure the MPI C and C++
compiler wrappers `mpicc` and `mpic++` are also in your path.  If so,
you are ready to install `mpi4py` with the following command:

::

    pip install mpi4py

The next packages you will want to install are `petsc` and `petsc4py`.  PETSc
stands for "Portable, Extensible Toolkit for Scientific Computation."
It is built on MPI.  The package `petsc4py` is the Python bindings for `petsc`.
To install these packages, first make sure you have a fortran
compiler installed, such as the GNU `gfortran` and make sure it is in your path.
Then, run the following command to install both `petsc` and `petsc4py`.

::

    pip install --allow-all-external petsc4py

Verify Installed Packages
---------------------------

To make sure MPI and `petsc` are working in your environment, you can use this
small `petsc4py` script:

::

    from petsc4py import PETSc
    rank = PETSc.COMM_WORLD.getRank()
    num_ranks = PETSc.COMM_WORLD.getSize()

    x = PETSc.Vec().createMPI(4) # VecCreateMPI: Creates a parallel vector.  size=4
    x.setValues([0,1,2,3], [10,20,30,40]) # VecSetValues: Inserts or adds values into certain locations of a vector.  x[0]=10, x[1]=20, x[2]=30, x[3]=40

    print ('Rank',rank,'has this portion of the MPI vector:', x.getArray() ) # VecGetArray: Returns a pointer to a contiguous array that contains this processor's portion of the vector data.

    vec_sum = x.sum() # VecSum: Computes the sum of all the components of a vector. 10+20+30+40=100

    if rank == 0:
        print ('Sum of all elements of vector x is',vec_sum,'and was computed using',num_ranks,'MPI processes.')


This script creates a PETSc MPI/parallel vector with four elements, sets the
value of those elements, and then computes the total sum of all the elements.
You can run the script with two processes
using `mpirun` (or `mpiexec`):

::

    mpirun -np 2 python petsc_test.py

The output will look something like this:

::

    Rank  1  has this portion of the MPI vector:  [ 30.  40.]
    Rank  0  has this portion of the MPI vector:  [ 10.  20.]
    Sum of all elements of vector x is 100.0 and was computed using 2 MPI processes.

As you can see, because we had a four element vector and two MPI processes,
PETSc automatically and evenly divided the vector in half across the two
processes.  If we tried three processes,
PETSc would not be able to split our four element vector up nicely across those
processes, yet it would still compute (inefficiently) the correct result:

::

    Rank  1  has this portion of the MPI vector:  [ 30.]
    Rank  2  has this portion of the MPI vector:  [ 40.]
    Rank  0  has this portion of the MPI vector:  [ 10.  20.]
    Sum of all elements of vector x is 100.0 and was computed using 3 MPI processes.
