import unittest
import warnings
from openmdao.api import IndepVarComp, ParamComp

class TestDeprecated(unittest.TestCase):
    def test_deprecated_paramcomp(self):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Trigger a warning.
            p = ParamComp('x', 1.0)

            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message),
                             'ParamComp is deprecated. Please switch to '
                             'IndepVarComp, which can be found in '
                             'openmdao.components.indep_var_comp.')

class TestErrors(unittest.TestCase):

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

    def test_reject_promotes_kwarg(self):               
        try:
            p = IndepVarComp('x', 100.0, promotes=['*'])
        except Exception as err:            
            self.assertEqual(str(err),
                             "IndepVarComp init: promotes is not supported in IndepVarComp.")
        else:
            self.fail("Error expected")

if __name__ == "__main__":
    unittest.main()