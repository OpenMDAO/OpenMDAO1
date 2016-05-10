
from distutils.core import setup

setup(name='openmdao',
      version='1.6.4',
      description="OpenMDAO v1 framework infrastructure",
      long_description="""\
      """,
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: CPython',
      ],
      keywords='optimization multidisciplinary multi-disciplinary analysis',
      author='OpenMDAO Team',
      author_email='openmdao@openmdao.org',
      url='http://openmdao.org',
      download_url='http://github.com/OpenMDAO/OpenMDAO/tarball/1.6.4',
      license='Apache License, Version 2.0',
      packages=[
          'openmdao',
          'openmdao.core',
          'openmdao.core.test',
          'openmdao.util',
          'openmdao.util.test',
          'openmdao.test',
          'openmdao.test.test',
          'openmdao.units',
          'openmdao.units.test',
          'openmdao.components',
          'openmdao.components.test',
          'openmdao.drivers',
          'openmdao.drivers.test',
          'openmdao.solvers',
          'openmdao.solvers.test',
          'openmdao.recorders',
          'openmdao.recorders.test',
          'openmdao.devtools',
          'openmdao.surrogate_models',
          'openmdao.surrogate_models.nn_interpolators',
          'openmdao.surrogate_models.test'
      ],
      package_data={'openmdao.units': ['unit_library.ini']},
      install_requires=[
        'six', 'numpydoc', 'networkx==1.9.1', 'numpy>=1.9.2',
        'scipy', 'sqlitedict', 'pyparsing'
      ],
      entry_points="""
      [console_scripts]
      wingproj=openmdao.devtools.wingproj:run_wing
      webview=openmdao.devtools.d3graph:webview_argv
      profview=openmdao.util.profile:prof_view
      proftotals=openmdao.util.profile:prof_totals
      profdump=openmdao.util.profile:prof_dump
      """
)
