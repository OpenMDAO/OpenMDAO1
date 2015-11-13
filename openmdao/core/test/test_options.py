""" Test for the OptionsDictionary """

import unittest
from six import PY2
from openmdao.api import OptionsDictionary


class TestOptions(unittest.TestCase):
    def test_options_dictionary(self):
        self.options = OptionsDictionary()

        # Make sure we can't address keys we haven't added

        with self.assertRaises(KeyError) as cm:
            self.options['junk']

        self.assertEqual('"Option \'{}\' has not been added"'.format('junk'), str(cm.exception))

        # Type checking - don't set a float with an int

        self.options.add_option('atol', 1e-6)
        self.assertEqual(self.options['atol'], 1.0e-6)

        with self.assertRaises(ValueError) as cm:
            self.options['atol'] = 1

        if PY2:
            self.assertEqual("'atol' should be a '<type 'float'>'", str(cm.exception))
        else:
            self.assertEqual("'atol' should be a '<class 'float'>'", str(cm.exception))

        # Check enum out of range

        self.options.add_option('iprint', 0, values = [0, 1, 2, 3])
        for value in [0,1,2,3]:
            self.options['iprint'] = value

        with self.assertRaises(ValueError) as cm:
            self.options['iprint'] = 4

        self.assertEqual("'iprint' must be one of the following values: '[0, 1, 2, 3]'", str(cm.exception))

        # Type checking for boolean

        self.options.add_option('conmin_diff', True)
        self.options['conmin_diff'] = True
        self.options['conmin_diff'] = False

        with self.assertRaises(ValueError) as cm:
            self.options['conmin_diff'] = "YES!"

        if PY2:
            self.assertEqual("'conmin_diff' should be a '<type 'bool'>'", str(cm.exception))
        else:
            self.assertEqual("'conmin_diff' should be a '<class 'bool'>'", str(cm.exception))

        # Test Max and Min

        self.options.add_option('maxiter', 10, lower=0, higher=10)
        for value in range(0, 11):
            self.options['maxiter'] = value

        with self.assertRaises(ValueError) as cm:
            self.options['maxiter'] = 15

        self.assertEqual("maximum allowed value for 'maxiter' is '10'", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.options['maxiter'] = -1

        self.assertEqual("minimum allowed value for 'maxiter' is '0'", str(cm.exception))

        # Make sure we can't do this
        with self.assertRaises(ValueError) as cm:
            self.options.maxiter = -1

        self.assertEqual("Use dict-like access for option 'maxiter'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
