#generate_openmdao_srcdoc_rst.py
#generate index.rst, the openmdao.[pkg].rst, and openmdao.[pkg].[item] rst files
#in a nested, procedural technique.

#This is script is currently called from the docs/Makefile via "make html"
#The srcdocs directory is removed and regenerated with each make.

#Some formatted strings that will be needed along the way.
index_top=""".. _source_documentation:

=============================
OpenMDAO Source Documentation
=============================

.. toctree::
   :maxdepth: 3
   :glob:


"""

package_top = """
.. toctree::
    :maxdepth: 3

"""

package_bottom = """
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

ref_sheet_bottom = """
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2
"""

import os
#need to set up the srcdocs directory structure, relative to docs.
os.mkdir("srcdocs")
os.mkdir("srcdocs" + os.path.sep + "packages")


#look for directories in the openmdao level, one up from docs
#those directories will be the openmdao packages
#auto-generate the top-level index.rst file for srcdocs, based on openmdao packages:
packages = []
listings = os.listdir("..")
#Everything in listings that isn't discarded is appended as a source package.
for listing in listings:
    if os.path.isdir(".." + os.path.sep + listing):
        if listing != "docs" and listing != "test" and listing != "config"\
        and listing != "devtools":
            packages.append(listing)

#begin writing the 'srcdocs/index.rst' file at top level.
index_filename = "srcdocs" + os.path.sep + "index.rst"
index = open(index_filename, "w")
index.write(index_top)

#auto-generate package header files (e.g. 'openmdao.core.rst')
for package in packages:
    #a package is e.g. openmdao.core, that contains source files
    #a sub_package, is a src file, e.g. openmdao.core.component
    sub_packages = []
    package_filename = "srcdocs" + os.path.sep + "packages" + os.path.sep + \
        "openmdao." + package + ".rst"
    package_name = "openmdao." + package
    sub_listings = os.listdir(".." + os.path.sep + package)

    #the sub_listing is going into each package dir and listing what's in it
    for sub_listing in sub_listings:
        #don't want to catalog files twice, nor use init files nor test dir
        if not "pyc" in sub_listing and not "__init__" in sub_listing \
        and sub_listing != "test":
            #just want the name of e.g. dataxfer not dataxfer.py
            sub_packages.append(sub_listing.rsplit('.')[0])

    if len(sub_packages) > 0:
        #continue to write in the top-level index file.
        #only document non-empty packages to avoid errors
        #(e.g. at time of writing, doegenerators, drivers, are empty dirs)
        index.write("   packages/openmdao." + package + "\n")

        #make subpkg directory (e.g. srcdocs/packages/core) for ref sheets
        package_dirname = "srcdocs" + os.path.sep + "packages" + os.path.sep + \
             package
        os.mkdir(package_dirname)

        #create/write a package index file: (e.g. "srcdocs/packages/openmdao.core.rst")
        package_file = open(package_filename, "w")
        package_file.write(package_name + "\n")
        package_file.write("-" * len(package_name) + "\n")
        package_file.write(package_top)

        for sub_package in sub_packages:
            #this line writes subpackage name e.g. "core/component.py"
            #into the corresponding package index file (e.g. "openmdao.core.rst")
            package_file.write("    " + package + os.path.sep + sub_package + "\n")

            #creates and writes out one reference sheet (e.g. core/component.rst)
            ref_sheet_filename = package_dirname + os.path.sep + sub_package + ".rst"
            ref_sheet = open(ref_sheet_filename, "w")
            #get the meat of the ref sheet code done
            filename = sub_package + ".py"
            ref_sheet.write(".. index:: "+ filename + "\n\n")
            ref_sheet.write(".. _" + package_name + "." + filename + ":\n\n")
            ref_sheet.write(filename + "\n")
            ref_sheet.write("+" * len(filename) + "\n\n")
            ref_sheet.write(".. automodule:: " + package_name + "." + sub_package)
            #finish and close each reference sheet.
            ref_sheet.write(ref_sheet_bottom)
            ref_sheet.close()

        #finish and close each package file
        package_file.write(package_bottom)
        package_file.close()

#finish and close top-level index file
index.close()
