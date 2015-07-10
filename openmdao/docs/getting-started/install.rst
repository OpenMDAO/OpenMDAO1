
.. warning::

        OpenMDAO 1.0 is in an ALPHA state.  The version that you are downloading
        and installing is under active development, and as such may be broken from time to time.
        Therefore, OpenMDAO 1.0 Alpha should be used at your own risk!

=====
Intro
=====

Purpose
-------

This document exists to explain what OpenMDAO is, how to get it, and how to install it
on OS X, Windows or Linux.  For a guide of examples of how to use OpenMDAO,
see the `OpenMDAO User Guide`_.

.. _OpenMDAO User Guide: ../usr-guide/design.html


`TL;DR`_
---------

.. _TL;DR: https://en.wikipedia.org/wiki/TL;DR

Install Python, Pip, Numpy, and Scipy. (`Anaconda Python <http://continuum.io/downloads>`_, comes
bundled with everything you need). Next, install OpenMDAO with pip:

::

    pip install git+http://github.com/OpenMDAO/OpenMDAO.git@master

This will at least get you started, but you should read the rest of this guide--
we worked really hard on it!

=================
What is OpenMDAO?
=================

OpenMDAO is a high-performance computing platform for systems analysis and optimization
that enables you to decompose your models, making them easier to build and
maintain, while still solving them in a tightly-coupled manner with efficient parallel
numerical methods.

We provide a library of sparse solvers and optimizers designed to work
with our MPI-based, distributed-memory data-passing scheme. But don't worry about
installing MPI when you're just getting started. We can run really efficiently in
serial using a Numpy data-passing implementation as well.

Our most unique capability is our automatic analytic multidisciplinary derivatives.
Provide analytic derivatives for each of your components, and
OpenMDAO will solve the chain-rule across your entire model, to compute system-
level derivatives for Newton solvers and/or gradient-based optimizers. This lets you
solve really large non-linear problems, like a `cubesat design <http://openmdao.org/publications/gray_hearn_moore_et_al_multidisciplinary_derivatives.pdf>`_
with over 25,000 design variables using adjoint derivatives.

You don't have to provide analytic derivatives for all of the components. OpenMDAO just
finite-differences components that are missing them and then computes semi-analytic
multidisciplinary derivatives. Semi-analytic derivatives offer a fast and easy
way to gain a lot of computational efficiency. For example, they gave us a 5x
reduction in compute cost for an `aero-structural wind turbine optimization
<http://openmdao.org/publications/gray_hearn_moore_et_al_multidisciplinary_derivatives.pdf>`_.

===========================
What's New in OpenMDAO 1.0?
===========================

If you're new to OpenMDAO, then all you need to know is that the API in 1.0 is different
than in any older version. So, if you look at older models in a forum post or something,
don't be surprised when the code doesn't look quite right.

If you're an existing OpenMDAO user trying to move your models up into this version,
then there are a bunch of API changes you need to be aware of.
OpenMDAO 1.0 Alpha, is a departure from the versions that preceded it (OpenMDAO 0.0.1 through 0.13.0).
In fact, OpenMDAO 1.0 is a complete re-write of the framework from the ground up. The new code base is
much smaller (it's now ~5000 lines of code), and will be much easier for others to work with.

We tried to follow a couple of guiding principles in the re-write:

#. The code base should have as few dependencies as possible
#. The user-facing API should be as small and self-consistent as possible
#. There should not be any "magic" happening without the user's knowledge

To that end we've made some significant changes to the framework. Some of the
most notable differences include:

  - The Component execute method is now named `solve_nonlinear`
  - There are now three types of variables: `parameter`, `output`, and `state`
  - The way you define and access framework variables (we've removed our dependency on Traits)
  - The way you group sets of components together (Assembly has been replaced with Group)
  - Solvers are not drivers any more. Optimizers and DOE are still drivers
  - There is no more workflow

If you'd like to get a more direct comparison between old and new OpenMDAO input files,
check out our `guide to converting your models, the <Pre-1.0 Conversion Guide>`_.
While a lot of the API has changed, the overall concepts are mostly the same.
You still have components, which get grouped together with connections, defining data
passing between them. The new API helps draw a sharper line between what is a framework
variable and what is a regular Python attribute. The new API also reduces the number of different
kinds of objects you have to interact with.

Since this is still an Alpha release, there is a lot of missing functionality
compared to the older versions. For example, we're not yet automatically computing
execution order for you, and we don't have full support for file-wrapped components
yet. We'll be working on adding the missing features as we go, but the Alpha is
already very capable, especially for gradient-based optimization with analytic derivatives.
We're putting it out specifically for our users to try the new API so that they can start to
play around with it. If you have any feedback, we'd love to hear it.

===================
Installation
===================

Supported Platforms
-------------------

OpenMDAO will run on specified versions of Windows, Mac OS X, and Linux.
However, we can't support every version of every OS.  So while you may very well
be able to get OpenMDAO to run on a iPhone that's running a Windows 3.1 emulator,
we're not going to be able to help you when something goes awry with that install.
Here are the systems which we will support:

Mac OS X
++++++++

 * Mountain Lion (10.8.5)

 * Mavericks (10.9.5)

 * Yosemite (10.10.4)

Linux
+++++

While we have seen successful installations using RHEL and Mint, the distribution
that we use is Ubuntu_.  The versions of Ubuntu that we will support are:

.. _Ubuntu: http://ubuntu.com

 * Trusty Tahr (14.04.2 LTS)

 * Utopic Unicorn (14.10)

 * Vivid Vervet (15.04)


Windows
+++++++

 * Windows 7

 * Windows 8


OpenMDAO Prerequisites
----------------------

In order to use OpenMDAO, you will need Python_ installed on your system.
You'll also need a few other basic scientific computing libraries for python:
Numpy and Scipy.

.. note::

    If you want a bundled Python installation that has all our prerequisites
    included, try Anaconda_.

    .. _Anaconda: http://continuum.io/downloads

Python
++++++

Currently, we are supporting two different versions of Python_:

.. _Python: http://www.python.org

 * 2.7.9_ or higher versions of 2.7.x

.. _2.7.9: https://www.python.org/downloads/release/python-279/

 * 3.4.3_ or higher versions of 3.4.x

 .. _3.4.3: https://www.python.org/downloads/release/python-343/


Numpy
+++++

Install Numpy_, unless you already have a distribution like Anaconda that
includes Numpy.

.. _Numpy: http://numpy.org

 * Version 1.9.2 or higher will be supported.

Scipy
+++++

Install Scipy_, unless you already have a distribution like Anaconda that
includes Scipy.

.. _Scipy: http://scipy.org

 * Version 0.15.1 or higher will be supported.

Git (Optional)
++++++++++++++
Git_ is a very popular open-source version control system that we use for our source code.
It tracks content such as files and directories. OpenMDAO hosts its repo on `GitHub <https://github.com/OpenMDAO/OpenMDAO>`_.
Git_ is not a hard requirement, but it's a good way to stay up to date with the latest code
updates (remember, we're still in ALPHA!).

.. _Git: http://git-scm.com/download

Compilers (Optional)
++++++++++++++++++++
OpenMDAO doesn't have a strict requirement on any compiled code, but we can optionally
make use of some compiled libraries, if they are present in your Python environment.
If you don't want to use any of these optional features, then you won't need
a compiler. You can always install the compilers later and build the libraries then,
and OpenMDAO will use them.

We can link to both the PyOpt and PyOpt-Sparse optimization libraries. Also, in
order to run things in parallel, you'll need petsc4py and mpi4py. So if you want to use those
packages, you'll either need platform-specific binaries for them, or you'll need a compiler.


Install OpenMDAO Using pip
--------------------------

To pip install OpenMDAO directly from the OpenMDAO Github repository:

::

    pip install git+http://github.com/OpenMDAO/OpenMDAO.git@master


Clone the Repo and Install From Source (Optional)
-------------------------------------------------

Since the code is in ALPHA state, and is changing daily, you might prefer to actually
clone our repository and install from that. This way you can always pull down the latest
changes without re-installing.

::

    git clone http://github.com/OpenMDAO/OpenMDAO-Framework


Then you're going to use pip to install in development mode. Change directories to
the top level of the OpenMDAO repository, and use the following command:

::

    pip install -e .


Testing
-------

You can run our test suite to see if your installation is working correctly.
Run any single test manually by simply passing the test file to python, or you can
use a test-runner, like `nosetest <https://nose.readthedocs.org/en/latest/>`_ to run
the whole OpenMDAO test suite at once. Once you've installed nosetest, go to the top of the
OpenMDAO repo and run:

::

    nosetest .
