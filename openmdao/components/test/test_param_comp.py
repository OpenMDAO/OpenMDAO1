import unittest
from openmdao.components.param_comp import ParamComp


class TestUnitComp(unittest.TestCase):

    def test_bad_init1(self):
        try:
            p = ParamComp('P')
        except Exception as err:
            self.assertEqual(str(err),
                             "ParamComp init: a value must be provided as the second arg.")

    def test_bad_init2(self):
        try:
            p = ParamComp(('P',))
        except Exception as err:
            self.assertEqual(str(err),
                             "ParamComp init: arg ('P',) must be a tuple of the form (name, value) or (name, value, keyword_dict).")

    def test_bad_init3(self):
        try:
            p = ParamComp(('P',1.0,()))
        except Exception as err:
            self.assertEqual(str(err),
                             "ParamComp init: arg ('P', 1.0, ()) must be a tuple of the form (name, value) or (name, value, keyword_dict).")

    def test_bad_init4(self):
        try:
            p = ParamComp(('P',1.0,1.0,1.0))
        except Exception as err:
            self.assertEqual(str(err),
                             "ParamComp init: arg ('P', 1.0, 1.0, 1.0) must be a tuple of the form (name, value) or (name, value, keyword_dict).")

    def test_bad_init5(self):
        try:
            p = ParamComp(1.0)
        except Exception as err:
            self.assertEqual(str(err),
                             "first argument to ParamComp init must be either of type `str` or an iterable of tuples of the form (name, value) or (name, value, keyword_dict).")

    def test_bad_init6(self):
        try:
            p = ParamComp([('x',)])
        except Exception as err:
            self.assertEqual(str(err),
                             "ParamComp init: arg ('x',) must be a tuple of the form (name, value) or (name, value, keyword_dict).")
