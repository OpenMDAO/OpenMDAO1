
import os
import sys
from setuptools import setup, find_packages

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
      packages=find_packages(),
      include_package_data=True,
      package_data={
          'openmdao': ['src/openmdao/docs/*']
      },
      zip_safe=False,
      install_requires=[
          'setuptools',
          'zope.interface',
      ],
      extras_require={
          'numpy_comps': ['numpy'],
      },
      entry_points="""
      [console_scripts]
      openmdao=openmdao.main.cli:openmdao
      """,
      )
