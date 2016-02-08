.. _MPI on Windows:


MPI on Windows
===============

This document provides the setup and usage of MPI (Message Passing Interface) in OpenMDAO on Windows (64-bit) using Python3 (64-bit).
We start with installing the necessary packages and test them to make sure they work.
Then we will look at a simple example of how to take advantage of MPI in OpenMDAO.

Python 3
------------

The Python 3 Environment should be installed from either the `official Python page`_ or Anaconda_.  Python 3.5 and later use Microsoft Visual Studio 2015.  Python 3.3 and 3.4 use Microsoft Visual Studio 2010.  The version you install must match the Visual Studio environment you install.  Make sure you also install `numpy` and `scipy` with your environment.

.. _official Python page: https://www.python.org/downloads/windows/
.. _Anaconda: https://www.continuum.io/downloads


Microsoft Visual Studio
-------------------------

Visual Studio 2010 Professional was used during this tutorial as the C/C++ compiler, because the official Python 3.4 build was compiled with that version.  Python2 cannot be used
because it was built with Visual Studio 2008 which no longer exists.


Microsoft MPI
---------------

You will need to install is MSMPI_ (Microsoft MPI).  You will need both the SDK (msmpisdk.msi) and the runtime (MSMpiSetup.exe).  As of this writing, version 7.0.12437.6 was used.

.. _MSMPI: https://www.microsoft.com/en-us/download/details.aspx?id=49926


Git for Windows
----------------

You will need to get a copy of `Git for Windows`_ and make sure it is in your path.

.. _Git for Windows: https://git-for-windows.github.io/


Cygwin
----------
The Cygwin64_ environment is required to build PETSc.  Although the Visual Studio Compiler `CL` is used, it is needed to run configure scripts and makefiles.  Cygwin64 should be installed with the standard development packages.  Additionally, you will need to install the `diffutils` package.

.. _Cygwin64: https://cygwin.com/install.html


Mpi4py
--------

The first package that must be installed is `mpi4py`_ which provides Python bindings for MPI.  You need to download and extract the source.  Build `mpi4py` with the following command:

.. _mpi4py: https://pypi.python.org/pypi/mpi4py


::

    python setup.py build


The setup should detect that the MSMPI SDK is installed, and if so, you will see this printed to the screen in the build output:

::

    MPI configuration: [msmpi] from 'C:\Program Files (x86)\Microsoft SDKs\MPI'


After a successful build, go ahead and install with:

::

    python setup.py install


PETSc
--------

The next step is to build PETSc_ from source.  PETSc stands for "Portable, Extensible Toolkit for Scientific Computation."  It is built on MPI.  Obtain the latest `petsc-lite` source package.  The version used as of this writing was 3.6.3.

.. _PETSc: http://www.mcs.anl.gov/petsc/download/index.html

Next launch Cygwin64.  We temporarily want to disable the Cygwin linker, so rename it:

::

    mv /usr/bin/link.exe /usr/bin/link-cygwin.exe


Close the Cygwin64 terminal.  Open up a new terminal window (`cmd`) and run the following commands:

::

    "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" amd64

    C:\cygwin64\bin\mintty.exe


Now switch to the new Cygwin terminal that `mintty.exe` opened.  This new cygwin terminal will now have the Visual Studio compilers in the path.  You should make sure the `cl` command works and that the `link` command is the Microsoft link.exe and not the Cygwin link.exe.  Also, the GNU `make` command should be in the path.  Also, make sure there is an installed version of Python in the path.  The default Anaconda python 2.7 is fine.  Now run:

::

    export PATH=/usr/bin:$PATH

    cd /cygdrive/c/path/to/petsc-3.6.3/

    export PATH=$PATH:`pwd`/bin/win32fe

    ./configure --with-cc="win32fe cl" --with-fc=0 --download-f2cblaslapack --with-mpi-include="/cygdrive/c/Program Files (x86)/Microsoft SDKs/MPI/Include" --with-mpi-lib=['/cygdrive/c/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib'] --with-mpi-mpiexec="/cygdrive/c/Program Files/Microsoft MPI/Bin/mpiexec.exe" --with-debugging=0 -CFLAGS='-O2 -MD -wd4996' -CXXFLAGS='-O2 -MD -wd4996' --with-file-create-pause=1


That will take some time, and when completed you will get a make command at the end after the message "Configure stage complete. Now build PETSc libraries with (gnumake build)."  Go ahead and run that command that will look something like:

::

    make PETSC_DIR=/cygdrive/c/path/to/petsc-3.6.3 PETSC_ARCH=arch-mswin-c-opt all


After that builds, you will get another command to run tests on the build.  Go ahead and run that command that will look something like:

::

    make PETSC_DIR=/cygdrive/c/path/to/petsc-3.6.3 PETSC_ARCH=arch-mswin-c-opt test


Finally, go ahead and restore the Cygwin linker back to what it originally was:

::

    mv /usr/bin/link-cygwin.exe /usr/bin/link.exe


Also, under `petsc-3.6.3/arch-mswin-c-opt/lib` make a duplicate copy of `libpetsc.lib` and rename it to `petsc.lib` in preparation for the install of `petsc4py`.


petsc4py
---------

The next step is to install `petsc4py`_ which is the Python bindings for `PETSc`.  Obtain the latest release.  Version 3.6.0 was used as of this writing.  Open up a new terminal window (`cmd`).  If you're using Anaconda, go ahead and activate the Python3 environment you want to use `petsc4py` in.  Then, set these two environmental variables:

.. _petsc4py: https://bitbucket.org/petsc/petsc4py/downloads


::

    set PETSC_DIR=C:/path/to/petsc-3.6.3

    set PETSC_ARCH=arch-mswin-c-opt

In the petsc4py directory, we need to make a correction.  Open up conf/baseconf.py and comment out line 186, or the line that reads:

::

    petsc_lib['runtime_library_dirs'].append(self['PETSC_LIB_DIR'])



Now go ahead and `cd` to the extracted `petsc4py` directory and issue the following command to build it, making sure all the paths are correct:

::

    python setup.py build build_ext --libraries="libf2cblas libf2clapack msmpi" --library-dirs="C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64;C:/path/to/petsc-3.6.3/arch-mswin-c-opt/lib" --include-dirs="C:/path/to/petsc-3.6.3/arch-mswin-c-opt/include;C:/path/to/petsc-3.6.3/include;C:/Program Files (x86)/Microsoft SDKs/MPI/Include"


We had to manually specify where all the paths and library directories were through build_ext because the default setup.py will try to use the Cygwin paths we used to build `petsc` which will not work in our `cmd` terminal.  The setup.py will still put the Cygwin paths in, but the build process will only issue warnings and ignore those Cygwin paths.

Now go ahead and install `petsc4py` with the following command:

::

    python setup.py install


Verify Installed Packages
---------------------------

To make sure MPI and `petsc4py` are working in your environment, you can use this small `petsc4py` script:

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


This script creates a PETSc MPI/parallel vector with four elements, sets the value of those elements, and then computes the total sum of all the elements.  You can run the script with two processes
using `mpiexec`:

::

    mpiexec -np 2 python petsc_test.py


The output will look something like this:

::

    Rank  1  has this portion of the MPI vector:  [ 30.  40.]
    Rank  0  has this portion of the MPI vector:  [ 10.  20.]
    Sum of all elements of vector x is 100.0 and was computed using 2 MPI processes.


As you can see, because we had a four element vector and two MPI processes, PETSc automatically and evenly divided the vector in half across the two processes.  If we tried three processes,
PETSc would not be able to split our four element vector up nicely across those processes, yet it would still compute (inefficiently) the correct result:

::

    Rank  1  has this portion of the MPI vector:  [ 30.]
    Rank  2  has this portion of the MPI vector:  [ 40.]
    Rank  0  has this portion of the MPI vector:  [ 10.  20.]
    Sum of all elements of vector x is 100.0 and was computed using 3 MPI processes.
