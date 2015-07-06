
.. _Getting-Started:

_____________________________
Getting Started with OpenMDAO
_____________________________

=======
Purpose
=======

This document exists to explain what OpenMDAO is, how to get it, and to help
the reader get OpenMDAO installed on her OS X, Windows or Linux machine, and
ensure that the installation passes all the tests.  This document is not a guide
on examples of how to use OpenMDAO, for that, see the OpenMDAO User's Guide. (link)
This document is not a guide of how to contribute to the development of OpenMDAO,
for that see the OpenMDAO Developer's Guide. (link)

This document does not help you with operating system problems, problems with
bad numpy or scipy installations, nor other types of software that you may have
installed that may interfere with OpenMDAO.

=================
What is OpenMDAO?
=================

OpenMDAO is a high-performance computing platform for systems analysis and optimization
that enables you to heavily decompose your models, making them easier to build and
maintain, while solving them in a tightly-coupled manner with efficient parallel
numerical methods.

===================
Supported Platforms
===================

OpenMDAO will run on specified versions of Windows, Mac OS X, and Linux.
However, we can't support every version of every OS.  So while you may very well
be able to get OpenMDAO to run on a phone that's running a Windows 3.1 emulator,
we're not going to be able to help you when something goes awry with that install.
Here are the systems on which we will test things, and therefore, support:

Mac OS X
++++++++

The 1.0.x versions of OpenMDAO should run on:

 * Mountain Lion (10.8.5)

 * Mavericks (10.9.5)

 * Yosemite (10.10.4)

 * El Capitan (10.11 (upcoming))


Linux
+++++

While we have seen successful installations using RHEL and Mint, the distribution
that we use is Ubuntu_.  The versions of Ubuntu that we will support are:

.. _Ubuntu: http://ubuntu.com

 * Trusty Tahr (14.04.2 LTS)

 * Utopic Unicorn (14.10)

 * Vivid Vervet (15.04)

 * Wily Werewolf (15.10 (upcoming))



Windows
+++++++

 * Windows 7

 * Windows 8

 * Windows 10 (upcoming)


======================
OpenMDAO Prerequisites
======================

Python
++++++

In order to use OpenMDAO, you will need Python_ installed at the system level of
your machine.  Currently, we are supporting two different versions of Python:

.. _Python: http://www.python.org

 * 2.7.9_ or higher versions of 2.7

.. _2.7.9: https://www.python.org/downloads/release/python-279/

 * 3.4.3_ or higher versions of 3.4

 .. _3.4.3: https://www.python.org/downloads/release/python-343/

Anaconda
++++++++

OpenMDAO works best using some kind of Python environment into which you can install it and
its pre-requisistes. We here at OpenMDAO.org recommend Anaconda, for the cleanest and
easiest installation experience.  The examples that follow will use Anaconda, but feel
free to use any type of Python environment (e.g. Virtualenv, Autoenv, etc.) to do the equivalent
things.

 Anaconda_  (version 3.7.4 or higher) is a Python distribution for scientific
 computing that includes Scipy and Numpy, among many other packages.

.. _Anaconda: http://continuum.io/downloads


If you're not familiar with an isolated Python environment, Anaconda (and products like
it) helps you to create a space into which one may install OpenMDAO.
These environments are kind of like sandboxes into which you can install many different
packages without "dirtying" your machine's system level.  Or, another way of thinking of it
might be that you can install a piece of software into an isolated environment so
that changes to system-level libraries won't "break" the installation of said software.


Git
+++
Git is a very popular and efficient open-source version control system that we use for our source code.
It tracks content such as files and directories. OpenMDAO keeps its code on GitHub, the Git website.
Download Git_ for later use to grab the OpenMDAO repository during installation.

.. _Git: http://git-scm.com/download


Compilers
+++++++++
The only compiled things in OpenMDAO are petsc and mpi. So if you want to use those
packages, you'll need binaries for them for your platform, or you'll need a compiler.
We don't have hard requirements for petsc or mpi, so if you don't want to do parallel operations,
then you shouldn't need a compiler installed for OpenMDAO.


Numpy
+++++

If you have Numpy_ installed at your system level, it will be available in your conda/
virtualenv environments.  However, you can also choose to install Numpy into your environment
if for any reason you don't want it in your system level.

.. _Numpy: http://numpy.org

 * Version 1.9.2 or higher will be supported.

Scipy
+++++

If you have Scipy_ installed at your system level, it will be available in your conda/
virtualenv environments.  However, you can also choose to install Scipy into your environment
if for any reason you don't want it in your system level, or if you have an older
version at system level that you don't want to mess with.

.. _Scipy: http://scipy.org

 * Version 0.15.1 or higher will be supported.

============
Installation
============

Create an Environment
+++++++++++++++++++++

Anaconda
--------

First, you'll want to create an environment with a name you choose that has the Python that
you desire.  Then you'll need to decide if you want to install numpy and scipy
into your conda environment, or let them be used from your top-level installation.
If no `== [version]` is given, the latest version will be installed.

This example creates a conda env named "openmdao" (you can name the env whatever you'd
like, for our examples, we'll use "openmdao") with Python 2.7.9 and the latest
numpy and scipy. We will also need a pip installed within the conda env for use
in the installation of OpenMDAO:

::

    conda create --name openmdao python==2.7.9 numpy scipy pip


NOTE: Anything not installed at the time of creation can be added to the environment
later by simply doing, from and activated prompt:

::

    conda install `[item]`

Activate Environment
++++++++++++++++++++

Once you have created an Anaconda environment, you need to activate it
in order to enter into it and use it. To leave the environment, you'll need to
deactivate.  Each product has different, platform-specific ways of achieving these
things, here's how it works in Anaconda.

Anaconda
--------

Windows:
&&&&&&&&
::

    activate openmdao
    deactivate

Linux/OSX:
&&&&&&&&&&
::

    source activate openmdao
    source deactivate


Git the OpenMDAO Source Code
++++++++++++++++++++++++++++

Now that we have created an environment with all of OpenMDAO's pre-requisistes,
and entered that environment, we need OpenMDAO itself. We will use Git to obtain
the source code from Github.

::

    git clone http://github.com/OpenMDAO/OpenMDAO-Framework




Install OpenMDAO Using pip
++++++++++++++++++++++++++

From your activated environment, from the top level of the OpenMDAO repository,
you'll want to pip install OpenMDAO into your environment.  The pip that you use
to do this installation needs to be pointing at the python inside your environment.
This is why we install pip into the env above.

::

    pip install -e .


=======
Testing
=======

You can test using any python test framework, e.g. `unittest`, `nosetest`.
