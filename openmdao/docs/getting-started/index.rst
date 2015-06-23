
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

Anaconda or Virtualenv
++++++++++++++++++++++

In order to install OpenMDAO, you'll have to install either

 * Anaconda_  (version 3.7.4 or higher)

 * Virtualenv_ (version 12.1 or higher)

.. _Anaconda: http://continuum.io/downloads

.. _Virtualenv: https://pypi.python.org/pypi/virtualenv

These two programs help you to create isolated Python environments into which one may install OpenMDAO.
These environments are kind of like sandboxes into which you can install many different
packages without dirtying your machine's system level.  Or, another way of thinking of it
might be that you can install a piece of software into an isolated environment so
that changes to system-level libraries won't "break" the installation of said software.
We here at OpenMDAO.org recommend Anaconda, for the cleanest and easiest installation experience.
Choose one of these tools and install it.

Compilers
+++++++++
The only compiled things in OpenMDAO are petsc and mpi. So if you want to use those
packages, you'll need binaries for them for your platform, or you'll need a compiler.
We don't have hard requirements for petsc or mpi, so if you don't want to do parallel operations,
then you shouldn't need a compiler installed for OpenMDAO.


Numpy
+++++

If you have Numpy_ installed at your system level, it will be available in your
environments.  However, you can also choose to install Numpy into your environment
if for any reason you don't want it in your system level.

.. _Numpy: http://numpy.org

 * Version 1.9.2 or higher will be supported.

Scipy
+++++

If you have Scipy_ installed at your system level, it will be available in your
environments.  However, you can also choose to install Scipy into your environment
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

How to create an environment?

`conda create --name openmdao python=2.7.9 numpy scipy`

Virtenv
-------

How to create an installation environment?


Activate Environment
++++++++++++++++++++

Anaconda
--------

 * Windows:
    `activate [env_name]`
    `deactivate`

 * Linux/OSX:
    `source activate [env_name]`
    `source deactivate`

Virtenv
-------



Install Using pip
+++++++++++++++++

Finally, do this: `pip install -e openmdao`


=======
Testing
=======

You can test using any python test framework from the openmdao level of the repository
