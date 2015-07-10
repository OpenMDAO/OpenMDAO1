.. _Anaconda-OpenMDAO:

========
Anaconda
========

Introduction
============

OpenMDAO works best using some kind of isolated Python environment into which you can install it and
its prerequisites. We here at OpenMDAO.org recommend Anaconda_, for the cleanest and
easiest installation experience.  Anaconda is a Python distribution for scientific
computing that includes Scipy and Numpy, among many other packages.  It also allows users to
operate in isolated Python environments.

If you're not familiar with an isolated Python environment, Anaconda
helps you to create a space into which you can install things, like OpenMDAO.
These environments are kind of like sandboxes into which you can install many different
packages of varying version numbers without "dirtying" your machine's system level.
Another way of thinking of it is that you can install OpenMDAO into an isolated environment, and
subsequent changes to system-level installations won't break the isolated OpenMDAO install.


Installation of OpenMDAO in Anaconda
====================================

Download and Install Anaconda
+++++++++++++++++++++++++++++

Download the Anaconda_ distribution for your platform (version 3.7.4 or higher) and run the installer.

.. _Anaconda: http://continuum.io/downloads

Create an Environment
+++++++++++++++++++++

Once Anaconda is installed, you'll want to create an environment that has the Python that
you desire.  Then you'll want to install numpy and scipy into your conda environment.
If no `== [version]` is given, the latest version of a package will be installed.

This example creates a conda env named "openmdao" (you can name the env whatever you'd
like) with Python 2.7.9 and the latest numpy and scipy. We will also need ``pip``
installed within the conda env for use in the installation of OpenMDAO:

::

    conda create --name openmdao python==2.7.9 numpy scipy pip


.. note:: Anything not installed at the time of creation can be added to the environment later by simply doing, from an activated prompt:

::

    conda install `[item]`

Activate Environment
++++++++++++++++++++

Once you have created an Anaconda environment, you need to activate it
to be able to use it. To leave the environment, you'll need to
deactivate.  Each platform has specific ways of achieving these
things:

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


Clone the Repo
++++++++++++++

Since the code is in ALPHA state, and is changing daily, you might prefer to actually
clone our repository and install from that. This way you can always pull down the latest
changes without re-installing.

::

    git clone http://github.com/OpenMDAO/OpenMDAO-Framework


Install OpenMDAO Using pip
++++++++++++++++++++++++++

Then you're going to use pip to install in development mode. From your activated environment,
change directories to the top level of the OpenMDAO repository, and use the following command:

::

    pip install -e .

From there, you should be good to use OpenMDAO any time that environment is activated.
