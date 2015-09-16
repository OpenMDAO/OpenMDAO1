import unittest
from openmdao.components.indep_var_comp import IndepVarComp


class TestUnitComp(unittest.TestCase):

    def test_bad_init1(self):
        try:
            p = IndepVarComp('P')
        except Exception as err:
            self.assertEqual(str(err),
                             "IndepVarComp init: a value must be provided as the second arg.")

    def test_bad_init2(self):
        try:
            p = IndepVarComp(('P',))
        except Exception as err:
            self.assertEqual(str(err),
                             "IndepVarComp init: arg ('P',) must be a tuple of the form (name, value) or (name, value, keyword_dict).")

    def test_bad_init3(self):
        try:
            p = IndepVarComp(('P',1.0,()))
        except Exception as err:
            self.assertEqual(str(err),
                             "IndepVarComp init: arg ('P', 1.0, ()) must be a tuple of the form (name, value) or (name, value, keyword_dict).")

    def test_bad_init4(self):
        try:
            p = IndepVarComp(('P',1.0,1.0,1.0))
        except Exception as err:
            self.assertEqual(str(err),
                             "IndepVarComp init: arg ('P', 1.0, 1.0, 1.0) must be a tuple of the form (name, value) or (name, value, keyword_dict).")

    def test_bad_init5(self):
        try:
            p = IndepVarComp(1.0)
        except Exception as err:
            self.assertEqual(str(err),
                             "first argument to IndepVarComp init must be either of type `str` or an iterable of tuples of the form (name, value) or (name, value, keyword_dict).")

    def test_bad_init6(self):
        try:
            p = IndepVarComp([('x',)])
        except Exception as err:
            self.assertEqual(str(err),
                             "IndepVarComp init: arg ('x',) must be a tuple of the form (name, value) or (name, value, keyword_dict).")
