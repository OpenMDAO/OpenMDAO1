.. warning::

        The following software is in an ALPHA state.  The version that you are downloading
        and installing is under active development, and as such may be broken from time to time.
        Therefore, OpenMDAO 1.0 Alpha should be used at your own risk!

=======
Purpose
=======

This document exists to explain what OpenMDAO is, how to get it, and how to install it
on OS X, Windows or Linux.  For a guide of examples of how to use OpenMDAO,
see the OpenMDAO User's Guide. (link)

=================
What is OpenMDAO?
=================

OpenMDAO is a high-performance computing platform for systems analysis and optimization
that enables you to decompose your models, making them easier to build and
maintain, while solving them in a tightly-coupled manner with efficient parallel
numerical methods.

===================
Supported Platforms
===================

OpenMDAO will run on specified versions of Windows, Mac OS X, and Linux.
However, we can't support every version of every OS.  So while you may very well
be able to get OpenMDAO to run on a iPhone that's running a Windows 3.1 emulator,
we're not going to be able to help you when something goes awry with that install.
Here are the systems on which we will test and support:

Mac OS X
++++++++

The 1.0.x versions of OpenMDAO should run on:

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

.. note::

    If you want a bundled Python installation that has all our prerequisites
    included, try :ref:`Anaconda`.  (This is the way the OpenMDAO developers do it.)


Numpy
+++++

Install Numpy_, unless you already have a distribution like Anaconda that already
includes Numpy.

.. _Numpy: http://numpy.org

 * Version 1.9.2 or higher will be supported.

Scipy
+++++

Install Scipy_, unless you already have a distribution like Anaconda that already
includes Scipy.

.. _Scipy: http://scipy.org

 * Version 0.15.1 or higher will be supported.

Compilers (Optional)
++++++++++++++++++++
OpenMDAO doesn't have a strict requirement on any compiled code, but we can optionally
make use of some compiled libraries, if they are present in your python environment.
In order to run things in parallel you'll need petsc4py and mpi4py. So if you want to use those
packages, you'll need binaries for them for your platform, or you'll need a compiler.
If you don't want to run in parallel, then you shouldn't need a compiler installed for OpenMDAO.

Git (Optional)
++++++++++++++
Git is a very popular and efficient open-source version control system that we use for our source code.
It tracks content such as files and directories. OpenMDAO keeps its code on GitHub, the Git website.
Git is not a hard requirement, though.  If you want, download Git_ for later use to grab the
OpenMDAO repository during installation.

.. _Git: http://git-scm.com/download

==========================
Install OpenMDAO Using pip
==========================

To pip install OpenMDAO directly from the OpenMDAO Github repository:

::

    pip install git+http://github.com/OpenMDAO/OpenMDAO.git@master


=======================================
Get the OpenMDAO Source Code (optional)
=======================================

Some users might prefer to have the source code, and install from that.
There are two ways to obtain the code: use Git, or download the code.

Using Git
+++++++++
 If you installed Git, use Git to obtain the OpenMDAO source code from Github:

::

    git clone http://github.com/OpenMDAO/OpenMDAO-Framework

Download File
+++++++++++++

    Download a zip file of code from the `OpenMDAO Github website <http://github.com/OpenMDAO/OpenMDAO-Framework/>`_,
    then unzip it locally.

If you get the source code, then pip installation will work differently. From
the top level of the OpenMDAO repository, you'll want to use the following command:

::

    pip install -e .



=======
Testing
=======

You can test using any python test framework, e.g. `unittest`, `nosetest` to run
the OpenMDAO test suite from the top level of the OpenMDAO repo.



pip install straight from Git, other alternate install options like python setup.py install
make some prereq OPTIONAL
