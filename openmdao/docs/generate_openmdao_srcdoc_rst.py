#generate_openmdao_srcdoc_rst.py
#generate index.rst, the openmdao.[pkg].rst and ref sheets
#in a nested technique.

#Some format strings that will be needed along the way.
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

#begin writing the "srcdocs/index.rst" file at top leve.
index_filename = "srcdocs" + os.path.sep + "index.rst"
index = open(index_filename, "w")
index.write(index_top)

#auto-generate package header files (e.g. 'openmdao.core.rst')
for package in packages:
    #a package is e.g. openmdao.core
    #a sub_package, for lack of better term, is e.g. openmdao.core.component
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

        #contine to write in the top-level index file
        #only write in packages that have something in them to avoid errors
        #e.g. at time of writing doegenerators, drivers, and surrmodels are empty dirs.
        index.write("   packages/openmdao." + package + "\n")

        #make directory (e.g.) srcdocs/packages/core in which to stick files
        package_dirname = "srcdocs" + os.path.sep + "packages" + os.path.sep + \
             package
        os.mkdir(package_dirname)

        #write a package index file: "openmdao.core.rst,"" for instance
        package_file = open(package_filename, "w")
        package_file.write(package_name + "\n")
        package_file.write("-" * len(package_name) + "\n")
        package_file.write(package_top)

        for sub_package in sub_packages:
            #this line writes subpackage name e.g. "core/component.py"
            #into the corresponding package index file e.g. "openmdao.core.rst"
            package_file.write("    " + package + os.path.sep + sub_package + "\n")

            #writes out the reference sheet for one item, e.g. core/component.py
            ref_sheet_filename = package_dirname + os.path.sep + sub_package + ".rst"
            ref_sheet = open(ref_sheet_filename, "w")
            ref_sheet.write(".. index:: "+ sub_package + ".py\n\n")
            ref_sheet.write(".. _" + package_name + "." + sub_package + ".py:\n\n")
            filename = sub_package + ".py"
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
