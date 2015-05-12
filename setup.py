
import os
import sys
from distutils.core import setup

setup(name='openmdao',
      version=1.0,
      description="OpenMDAO v1 framework infrastructure",
      long_description="""\
""",
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
      ],
      keywords='optimization multidisciplinary multi-disciplinary analysis',
      author='',
      author_email='',
      url='http://openmdao.org',
      license='Apache License, Version 2.0',
      packages=[
          'openmdao',
          'openmdao.core',
          'openmdao.core.test',
          'openmdao.util',
          'openmdao.test',
          'openmdao.units',
          'openmdao.components',
          'openmdao.drivers',
          'openmdao.doegenerators',
          'openmdao.surrogatemodels',
      ],
      install_requires=[
        'six', 'Sphinx', 'numpydoc'
      ],
      entry_points= """
        [console_scripts]
        wingproj=openmdao.devtools.wingproj:run_wing
      """
    )
