========
Anaconda
========

OpenMDAO works best using some kind of isolated Python environment into which you can install it and
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


Installation
============

Create an Environment
+++++++++++++++++++++

First, you'll want to create an environment with a name you choose that has the Python that
you desire.  Then you'll want to install numpy and scipy into your conda environment.
If no `== [version]` is given, the latest version will be installed.

This example creates a conda env named "openmdao" (you can name the env whatever you'd
like, for our examples, we'll use "openmdao") with Python 2.7.9 and the latest
numpy and scipy. We will also need a pip installed within the conda env for use
in the installation of OpenMDAO:

::

    conda create --name openmdao python==2.7.9 numpy scipy pip


.. note:: Anything not installed at the time of creation can be added to the environment later by simply doing, from and activated prompt:

::

    conda install `[item]`

Activate Environment
++++++++++++++++++++

Once you have created an Anaconda environment, you need to activate it
in order to enter into it and use it. To leave the environment, you'll need to
deactivate.  Each product has different, platform-specific ways of achieving these
things, here's how it works in Anaconda.

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


Install OpenMDAO Using pip
++++++++++++++++++++++++++

From your activated environment, cd to the top level of the OpenMDAO repository,
and use pip to install OpenMDAO into your environment.  The pip that you use
to do this installation needs to be pointing at the python inside your environment.
This is why we installed pip into our environment in the above steps.

::

    pip install -e .


From there, you should be good to use OpenMDAO any time that environment is activated.
