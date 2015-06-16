import os
os.mkdir("srcdocs")
os.mkdir("srcdocs" + os.path.sep + "packages")

#auto-generate the top-level index.rst file for srcdocs, based on openmdao packages:
index_top=""".. _source_documentation:

=============================
OpenMDAO Source Documentation
=============================

.. toctree::
   :maxdepth: 3
   :glob:


"""
packages = []
#look for directories in the openmdao level, one up from docs
#those directories will be the openmdao packages
listings = os.listdir("..")
for listing in listings:
    if os.path.isdir(".." + os.path.sep + listing):
        if listing != "docs" and listing != "test" and listing != "config"\
        and listing != "devtools":
            packages.append(listing)
index = open("srcdocs/index.rst", "w")
index.write(index_top)

for package in packages:
    index.write("   packages/openmdao." + package + "\n")
    #index.write(package)
    #index.write("\n")

index.close()

#auto-generate package header files (e.g. 'openmdao.core.rst')
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

#a package is e.g. openmdao.core
for package in packages:
    #a sub_package, for lack of better term, is e.g. openmdao.core.whatever
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
        #make directory (e.g.) srcdocs/packages/core in which to stick files
        package_dirname = "srcdocs" + os.path.sep + "packages" + os.path.sep + \
             package
        os.mkdir(package_dirname)

        #write the openmdao.core.rst, for instance
        package_file = open(package_filename, "w")
        package_file.write(package_name + "\n")
        package_file.write("-" * len(package_name) + "\n")
        package_file.write(package_top)

        for sub_package in sub_packages:
            #this line writes subpackage name e.g. core/component.py
            #into package's dir file e.g. openmdao.core.rst
            package_file.write("    " + package + os.path.sep + sub_package + "\n")

            #this segment writes out the subfile to autodoc e.g. core/component.py
            #ref_sheet_filename = "srcdocs"+ os.path.sep + "packages" + os.path.sep \
            #    + package + os.path.sep + sub_package + ".rst"
            ref_sheet_filename = package_dirname + os.path.sep + sub_package + ".rst"
            ref_sheet = open(ref_sheet_filename, "w")
            ref_sheet.write(".. index:: "+ sub_package + ".py\n\n")
            ref_sheet.write(".. _" + package_name + "." + sub_package + ".py:\n\n")
            filename = sub_package + ".py"
            ref_sheet.write(filename + "\n")
            ref_sheet.write("+" * len(filename) + "\n\n")
            ref_sheet.write(".. automodule:: " + package_name + "." + sub_package)
            ref_sheet_bottom = """
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2
"""

            ref_sheet.write(ref_sheet_bottom)
            ref_sheet.close()

        package_file.write(package_bottom)
        package_file.close()
