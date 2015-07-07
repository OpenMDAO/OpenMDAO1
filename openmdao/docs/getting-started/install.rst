=======
Purpose
=======

This document exists to explain what OpenMDAO is, how to get it, and to help
the reader get OpenMDAO installed on her OS X, Windows or Linux machine, and
ensure that the installation passes all the tests.  This document is not a guide
on examples of how to use OpenMDAO, for that, see the OpenMDAO User's Guide. (link)
This document is not a guide of how to contribute to the development of OpenMDAO,
for that see the OpenMDAO Developer's Guide. (link)

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
be able to get OpenMDAO to run on a iPhone that's running a Windows 3.1 emulator,
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

Compilers
+++++++++
The only compiled things in OpenMDAO are petsc and mpi. So if you want to use those
packages, you'll need binaries for them for your platform, or you'll need a compiler.
We don't have hard requirements for petsc or mpi, so if you don't want to do parallel operations,
then you shouldn't need a compiler installed for OpenMDAO.

Git
+++
Git is a very popular and efficient open-source version control system that we use for our source code.
It tracks content such as files and directories. OpenMDAO keeps its code on GitHub, the Git website.
Git is not a hard requirement, though.  If you want, download Git_ for later use to grab the
OpenMDAO repository during installation.

.. _Git: http://git-scm.com/download


============================
Get the OpenMDAO Source Code
============================

Now that we have all of OpenMDAO's pre-requisistes, we need OpenMDAO itself.
There are two ways to obtain the code: use Git, or download the code.

1.  If you installed Git, use Git to obtain the OpenMDAO source code from Github:

::

    git clone http://github.com/OpenMDAO/OpenMDAO-Framework

2.  Download a zip file of code from the `OpenMDAO Github website <http://github.com/OpenMDAO/OpenMDAO-Framework/>`_.

==========================
Install OpenMDAO Using pip
==========================

From within the top level of the OpenMDAO repository, you'll want to pip install OpenMDAO
into your environment using the following command:

::

    pip install -e .


.. note:: If you want to install things the way the OpenMDAO devs do it, check out this section on  using OpenMDAO in :ref:`Anaconda`.


=======
Testing
=======

You can test using any python test framework, e.g. `unittest`, `nosetest` to run
the OpenMDAO test suite from the top level of the OpenMDAO repo.
