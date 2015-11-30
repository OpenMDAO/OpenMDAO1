"""
Testing the namelist writer.
"""

import os.path
import sys
import unittest
import tempfile
import shutil

from numpy import float32 as numpy_float32
from numpy import int32 as numpy_int32
from numpy import array, zeros

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.util.namelist_util import Namelist, ToBool


class VarComponent(Component):
    """Contains some vars"""

    def __init__(self):
        super(VarComponent, self).__init__()

        self.add_param('boolvar', False, pass_by_obj=True)
        self.add_param('intvar', 333, pass_by_obj=True)
        self.add_param('floatvar', -16.54)
        self.add_param('expvar1', 1.2)
        self.add_param('expvar2', 1.2)
        self.add_param('textvar', "This", pass_by_obj=True)
        self.add_param('arrayvar', zeros((3, )))
        self.add_param('arrayvarsplit', zeros((5, )))
        self.add_param('arrayvarsplit2', zeros((3, )))
        self.add_param('arrayvarzerod', zeros((0, 0)), pass_by_obj=True)
        self.add_param('arrayvartwod', zeros((1, 3)))
        self.add_param('arraysmall', zeros((1, )))
        self.add_param('arrayshorthand', zeros((5, )))
        self.add_param('single', zeros((1, ), dtype=numpy_int32), pass_by_obj=True)
        self.add_param('singleint', 0, pass_by_obj=True)
        self.add_param('singlebool', zeros((0, 0)), pass_by_obj=True)
        self.add_param('stringarray', [], pass_by_obj=True)

        # Formerly the VarTree `varcontainer`
        self.add_param('varcontainer:boolvar', True, pass_by_obj=True)
        self.add_param('varcontainer:intvar', 7777, pass_by_obj=True)
        self.add_param('varcontainer:floatvar', 2.14543)
        self.add_param('varcontainer:textvar', "Hey", pass_by_obj=True)
        self.add_param('varcontainer_1:boolvar_1',True,pass_by_obj=True)
        self.add_param('varcontainer_1:intvar_1', 4444, pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):
        pass


class TestCase(unittest.TestCase):
    """ Test namelist writer functions. """

    def setUp(self):
        self.filename = 'test_namelist.dat'
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='omdao-')
        os.chdir(self.tempdir)

    def tearDown(self):
        # if os.path.exists(self.filename):
        #     os.remove(self.filename)
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_forgot_to_read(self):

        my_comp = VarComponent()
        sb = Namelist(my_comp)
        try:
            sb.load_model()
        except RuntimeError as err:
            msg = "Input file must be read with parse_file before " \
                  "load_model can be executed."
            self.assertEqual(str(err), msg)
        else:
            self.fail('RuntimeError expected')

    def test_writes(self):

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)

        sb.set_filename(self.filename)
        sb.set_title("Testing")

        sb.add_group('FREEFORM')
        sb.add_group('OPTION')
        sb.add_comment("This is a comment")
        sb.add_var("boolvar")
        sb.add_var("intvar")
        sb.add_var("floatvar")
        sb.add_var("textvar")

        sb.add_newvar("newcard", "new value")

        sb.generate()

        f = open(self.filename, 'r')
        contents = f.read()

        compare = "Testing\n" + \
                  "FREEFORM\n" + \
                  "&OPTION\n" + \
                  "  This is a comment\n" + \
                  "  boolvar = F\n" + \
                  "  intvar = 333\n" + \
                  "  floatvar = -16.54\n" + \
                  "  textvar = 'This'\n" + \
                  "  newcard = 'new value'\n" + \
                  "/\n"

        self.assertEqual(contents, compare)

    def test_read1(self):
        # Put variables in top container, so no rules_dict

        namelist1 = "Testing\n" + \
                    "  \n" + \
                    "&OPTION\n" + \
                    "  This is a comment\n" + \
                    "  INTVAR = 777, single(1) = 15.0, floatvar = -3.14\n" + \
                    "  singleint(2) = 3,4,5\n" + \
                    "  stringarray(3) = 'xyz'\n" + \
                    "  boolvar = T\n" + \
                    "  textvar = 'That'\n" + \
                    "  ! This is a comment too\n" + \
                    "  arrayvar = 3.5, 7.76, 1.23\n" + \
                    "  arrayvarsplit = 3.5, 7.76\n" + \
                    "                  5.45, 22.0\n" + \
                    "                  1.23\n" + \
                    "  arrayvarsplit2 = 1\n" + \
                    "                   2\n" + \
                    "                   3\n" + \
                    "  arraysmall = 1.75\n" + \
                    "  arrayshorthand = 3.456*8\n" + \
                    "  expvar1 = 1.5e-12\n" + \
                    "  expvar2 = -1.5D12\n" + \
                    "/\n"

        outfile = open(self.filename, 'w')
        outfile.write(namelist1)
        outfile.close()

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)
        sb.set_filename(self.filename)

        sb.parse_file()

        sb.load_model()

        self.assertEqual(sb.title, 'Testing')
        self.assertEqual(top['my_comp.intvar'], 777)
        self.assertEqual(top['my_comp.boolvar'], True)
        self.assertEqual(top['my_comp.floatvar'], -3.14)
        self.assertEqual(top['my_comp.expvar1'], 1.5e-12)
        self.assertEqual(top['my_comp.expvar2'], -1.5e12)
        self.assertEqual(top['my_comp.textvar'], 'That')
        self.assertEqual(top['my_comp.arrayvar'][0], 3.5)
        self.assertEqual(top['my_comp.arrayvar'][1], 7.76)
        self.assertEqual(top['my_comp.arrayvar'][2], 1.23)
        self.assertEqual(top['my_comp.arrayvarsplit'][0], 3.5)
        self.assertEqual(top['my_comp.arrayvarsplit'][1], 7.76)
        self.assertEqual(top['my_comp.arrayvarsplit'][2], 5.45)
        self.assertEqual(top['my_comp.arrayvarsplit'][3], 22.0)
        self.assertEqual(top['my_comp.arrayvarsplit'][4], 1.23)
        self.assertEqual(top['my_comp.arrayvarsplit2'][0], 1)
        self.assertEqual(top['my_comp.arrayvarsplit2'][1], 2)
        self.assertEqual(top['my_comp.arrayvarsplit2'][2], 3)
        self.assertEqual(top['my_comp.arraysmall'][0], 1.75)
        self.assertEqual(len(top['my_comp.arraysmall']), 1)
        self.assertEqual(top['my_comp.arrayshorthand'][4], 3.456)
        self.assertEqual(len(top['my_comp.arrayshorthand']), 8)
        self.assertEqual(top['my_comp.single'][0], 15.0)
        self.assertEqual(top['my_comp.singleint'][3], 5)
        self.assertEqual(top['my_comp.stringarray'][2], 'xyz')

        # Test out reading a single card by name
        self.assertEqual(sb.find_card('OPTION', 'floatvar'), -3.14)

    def test_read2(self):
        # Put variables in container, using rules_dict

        namelist1 = "Testing\n" + \
                    "  \n" + \
                    "&OPTION\n" + \
                    "  This is a comment\n" + \
                    "  intvar = 777\n" + \
                    "  boolvar = .FALSE.\n" + \
                    "  floatvar = -3.14\n" + \
                    "  extravar = 555\n" + \
                    "  TEXTVAR = 'That'\n" + \
                    "  ! This is a comment too\n" + \
                    "/\n" + \
                    "&NODICT\n" + \
                    "  noval = 0\n" + \
                    "/\n" + \
                    "&DUPE\n" + \
                    "  some = 0\n" + \
                    "/\n" + \
                    "&DUPE\n" + \
                    "  some = 0\n" + \
                    "/\n" + \
                    "FREEFORM\n"

        outfile = open(self.filename, 'w')
        outfile.write(namelist1)
        outfile.close()

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)
        sb.set_filename(self.filename)

        sb.parse_file()

        rules_dict = { "OPTION" : ["varcontainer"] }
        n1, n2, n3 = sb.load_model(rules_dict)

        self.assertEqual(n1[4], 'FREEFORM')
        self.assertEqual(n2[1], 'NODICT')
        self.assertEqual(n2[2], 'DUPE')
        self.assertEqual(n2[3], 'DUPE')
        self.assertEqual(n3, ['extravar'])
        self.assertEqual(top['my_comp.varcontainer:intvar'], 777)
        self.assertEqual(top['my_comp.varcontainer:boolvar'], False)
        self.assertEqual(top['my_comp.varcontainer:floatvar'], -3.14)
        self.assertEqual(top['my_comp.varcontainer:textvar'], 'That')

    def test_read3(self):
        # Parse a single group in a deck with non-unique group names

        namelist1 = "Testing\n" + \
                    "$GROUP\n" + \
                    "  intvar = 99\n" + \
                    "$END\n" + \
                    "$GROUP\n" + \
                    "  floatvar = 3.5e-23\n" + \
                    "$END\n"

        outfile = open(self.filename, 'w')
        outfile.write(namelist1)
        outfile.close()

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)
        sb.set_filename(self.filename)

        sb.parse_file()

        sb.load_model(single_group=1)

        # Unchanged
        self.assertEqual(top['my_comp.intvar'], 333)
        # Changed
        self.assertEqual(top['my_comp.floatvar'], 3.5e-23)

    def test_read4(self):
        # Variables on same line as header

        namelist1 = "Testing\n" + \
                    "  \n" + \
                    "&OPTION boolvar = T, arrayvar = 3.5, 7.76, 1.23\n" + \
                    "/\n"

        outfile = open(self.filename, 'w')
        outfile.write(namelist1)
        outfile.close()

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)
        sb.set_filename(self.filename)

        sb.parse_file()

        sb.load_model()

        self.assertEqual(top['my_comp.boolvar'], True)
        self.assertEqual(top['my_comp.arrayvar'][0], 3.5)

        namelist1 = "Testing\n" + \
                    "  \n" + \
                    "$OPTION boolvar = T, arrayvar = 3.5, 7.76, 1.23, $END\n"

        outfile = open(self.filename, 'w')
        outfile.write(namelist1)
        outfile.close()

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)
        sb.set_filename(self.filename)

        sb.parse_file()

        sb.load_model()

        self.assertEqual(top['my_comp.boolvar'], True)
        self.assertEqual(top['my_comp.arrayvar'][0], 3.5)

    def test_2Darray_read(self):

        namelist1 = "Testing\n" + \
                    "$GROUP\n" + \
                    "  arrayvartwod(1,1) = 12, 24, 36\n" + \
                    "  arrayvartwod(1,2) = 33, 66, 99\n" + \
                    "$END\n"

        outfile = open(self.filename, 'w')
        outfile.write(namelist1)
        outfile.close()

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)
        sb.set_filename(self.filename)

        sb.parse_file()

        sb.load_model()

        # Unchanged
        self.assertEqual(top['my_comp.arrayvartwod'][0][0], 12)
        self.assertEqual(top['my_comp.arrayvartwod'][1][2], 99)

    def test_vartree_write(self):

        top = Problem()
        top.root = Group()
        top.root.add('my_comp',VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)

        sb.set_filename(self.filename)
        sb.add_group('Test')
        sb.add_container("varcontainer")

        sb.generate()

        f = open(self.filename, 'r')
        contents = f.read()

        self.assertEqual("boolvar = T" in contents, True)
        self.assertEqual("textvar = 'Hey'" in contents, True)
        self.assertEqual("floatvar = 2.14543" in contents, True)
        self.assertEqual("intvar = 7777" in contents, True)


        # now test skipping

        sb = Namelist(top.root.my_comp)
        top['my_comp.varcontainer:boolvar'] = True
        top['my_comp.varcontainer:textvar'] = "Skipme"

        sb.set_filename(self.filename)
        sb.add_group('Test')
        sb.add_container("varcontainer", skip='textvar')
        sb.generate()

        f = open(self.filename, 'r')
        contents = f.read()
        self.assertEqual("boolvar = T" in contents, True)
        self.assertEqual("textvar = 'Skipme'" in contents, False)
        self.assertEqual("intvar = 7777" in contents, True)

    def test_vartree_write2(self):
        #testing namelist_util before setup()
        top = Problem()
        top.root = Group()
        myvars = VarComponent()
        top.root.add('my_comp', myvars)


        sb = Namelist(top.root.my_comp)

        sb.set_filename(self.filename)
        sb.add_group('Test')
        sb.add_container("varcontainer")

        sb.generate()

        f = open(self.filename, 'r')
        contents = f.read()

        self.assertEqual("boolvar = T" in contents, True)
        self.assertEqual("textvar = 'Hey'" in contents, True)
        self.assertEqual("floatvar = 2.14543" in contents, True)
        self.assertEqual("intvar = 7777" in contents, True)
        self.assertEqual("varcontainer" in contents,False) 

        #ensure that containers with similar names are not confused
        # varcontainer vs. varcontainer_1
        self.assertEqual("boolvar_1" in contents, False)
        self.assertEqual("intvar_1" in contents, False)
        # now test skipping

        sb = Namelist(top.root.my_comp)
        myvars._init_params_dict['varcontainer:boolvar']['val'] = False
        myvars._init_params_dict['varcontainer:textvar']['val'] = "Skipme"
        myvars._init_params_dict['varcontainer:intvar']['val'] = 8888
        myvars._init_params_dict['varcontainer:floatvar']['val'] = 3.14

        sb.set_filename(self.filename)
        sb.add_group('Test')
        sb.add_container("varcontainer",skip='textvar')
        sb.generate()

        f = open(self.filename, 'r')
        contents = f.read()
        self.assertEqual("boolvar = F" in contents, True)
        self.assertEqual("intvar = 8888" in contents, True)
        self.assertEqual("floatvar = 3.14" in contents, True)
        self.assertEqual("textvar = 'Skipme'" in contents, False)
        self.assertEqual("varcontainer" not in contents,True) 
        

        sb = Namelist(top.root.my_comp)
        sb.set_filename(self.filename)
        sb.add_group('Test')
        sb.add_container("varcontainer_1")
        sb.generate()
        
        f = open(self.filename, 'r')
        contents = f.read()
        self.assertEqual("boolvar_1 = T" in contents, True)
        self.assertEqual("intvar_1 = 4444" in contents, True)


        top.setup(check=False)
        top.run()




    def test_1Darray_write(self):

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)

        top['my_comp.arrayvar'] = zeros(3, dtype=numpy_float32)
        top['my_comp.arrayvar'][2] = 3.7
        top['my_comp.single'] = array(['a', 'b', 'c'])
        top['my_comp.singleint'] = array([1, 2, 3])
        top['my_comp.singlebool'] = array([False, True, False])

        sb.set_filename(self.filename)
        sb.add_group('Test')
        sb.add_var("arrayvar")
        # This should be ignored because it is zero-D
        sb.add_var("arrayvarzerod")
        sb.add_var("single")
        sb.add_var("singleint")
        sb.add_var("singlebool")

        sb.generate()

        f = open(self.filename, 'r')
        contents = f.read()

        compare = "\n" + \
                  "&Test\n" + \
                  "  arrayvar = 0.0, 0.0, 3.700000047683716\n" + \
                  "  single = 'a', 'b', 'c'\n" + \
                  "  singleint = 1, 2, 3\n" + \
                  "  singlebool = F, T, F\n" + \
                  "/\n"

        self.assertEqual(contents, compare)

    def test_2Darray_write(self):

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)

        top['my_comp.arrayvar'] = zeros([3, 2], dtype=numpy_float32)
        top['my_comp.arrayvar'][0, 1] = 3.7
        top['my_comp.arrayvar'][2, 0] = 7.88

        sb.set_filename(self.filename)
        sb.add_group('Test')
        sb.add_var("arrayvar")

        sb.generate()

        f = open(self.filename, 'r')
        contents = f.read()

        compare = "\n" + \
                  "&Test\n" + \
                  "  arrayvar(1,1) = 0.0,  3.700000047683716, \n" + \
                  "arrayvar(1,2) = 0.0,  0.0, \n" + \
                  "arrayvar(1,3) = 7.880000114440918,  0.0, \n" + \
                  "/\n"

        self.assertEqual(contents, compare)

    def test_unsupported_array(self):

        top = Problem()
        top.root = Group()
        top.root.add('my_comp', VarComponent())

        top.setup(check=False)
        top.run()

        sb = Namelist(top.root.my_comp)

        top['my_comp.arrayvar'] = zeros([2, 2, 2], dtype=numpy_float32)

        sb.set_filename(self.filename)
        sb.add_group('Test')
        sb.add_var("arrayvar")

        try:
            sb.generate()
        except RuntimeError as err:
            self.assertEqual(str(err),
                             "Don't know how to handle array of" + \
                                           " 3 dimensions")
        else:
            self.fail('RuntimeError expected')

    def test_bool_token_error(self):

        try:
            token = ToBool('Junk')
            token.postParse(0, 0, ["Junk"])
        except RuntimeError as err:
            msg = "Unexpected error while trying to identify a Boolean value in the namelist."
            self.assertEqual(str(err), msg)
        else:
            self.fail('RuntimeError expected')

    def test_read_pre_setup(self):
        # Put variables in top container, so no rules_dict

        namelist1 = "Testing\n" + \
                    "  \n" + \
                    "&OPTION\n" + \
                    "  This is a comment\n" + \
                    "  INTVAR = 777, single(1) = 15.0, floatvar = -3.14\n" + \
                    "  singleint(2) = 3,4,5\n" + \
                    "  stringarray(3) = 'xyz'\n" + \
                    "  boolvar = T\n" + \
                    "  textvar = 'That'\n" + \
                    "  ! This is a comment too\n" + \
                    "  arrayvar = 3.5, 7.76, 1.23\n" + \
                    "  arrayvarsplit = 3.5, 7.76\n" + \
                    "                  5.45, 22.0\n" + \
                    "                  1.23\n" + \
                    "  arrayvarsplit2 = 1\n" + \
                    "                   2\n" + \
                    "                   3\n" + \
                    "  arraysmall = 1.75\n" + \
                    "  arrayshorthand = 3.456*8\n" + \
                    "  expvar1 = 1.5e-12\n" + \
                    "  expvar2 = -1.5D12\n" + \
                    "/\n"

        outfile = open(self.filename, 'w')
        outfile.write(namelist1)
        outfile.close()

        top = Problem()
        top.root = Group()
        my_comp = top.root.add('my_comp', VarComponent())

        sb = Namelist(my_comp)
        sb.set_filename(self.filename)

        sb.parse_file()

        sb.load_model()

        self.assertEqual(sb.title, 'Testing')
        self.assertEqual(my_comp._init_params_dict['intvar']['val'], 777)
        self.assertEqual(my_comp._init_params_dict['boolvar']['val'], True)
        self.assertEqual(my_comp._init_params_dict['floatvar']['val'], -3.14)
        self.assertEqual(my_comp._init_params_dict['expvar1']['val'], 1.5e-12)
        self.assertEqual(my_comp._init_params_dict['expvar2']['val'], -1.5e12)
        self.assertEqual(my_comp._init_params_dict['textvar']['val'], 'That')
        self.assertEqual(my_comp._init_params_dict['arrayvar']['val'][0], 3.5)
        self.assertEqual(my_comp._init_params_dict['arrayvar']['val'][1], 7.76)
        self.assertEqual(my_comp._init_params_dict['arrayvar']['val'][2], 1.23)
        self.assertEqual(my_comp._init_params_dict['arrayvarsplit']['val'][0], 3.5)
        self.assertEqual(my_comp._init_params_dict['arrayvarsplit']['val'][1], 7.76)
        self.assertEqual(my_comp._init_params_dict['arrayvarsplit']['val'][2], 5.45)
        self.assertEqual(my_comp._init_params_dict['arrayvarsplit']['val'][3], 22.0)
        self.assertEqual(my_comp._init_params_dict['arrayvarsplit']['val'][4], 1.23)
        self.assertEqual(my_comp._init_params_dict['arrayvarsplit2']['val'][0], 1)
        self.assertEqual(my_comp._init_params_dict['arrayvarsplit2']['val'][1], 2)
        self.assertEqual(my_comp._init_params_dict['arrayvarsplit2']['val'][2], 3)
        self.assertEqual(my_comp._init_params_dict['arraysmall']['val'], 1.75)
        self.assertEqual(my_comp._init_params_dict['arrayshorthand']['val'][4], 3.456)
        self.assertEqual(len(my_comp._init_params_dict['arrayshorthand']['val']), 8)
        self.assertEqual(my_comp._init_params_dict['single']['val'][0], 15.0)
        self.assertEqual(my_comp._init_params_dict['singleint']['val'][3], 5)
        self.assertEqual(my_comp._init_params_dict['stringarray']['val'][2], 'xyz')

if __name__ == '__main__':
    import nose
    sys.argv.append('--cover-package=openmdao.util')
    sys.argv.append('--cover-erase')
    nose.runmodule()
