import unittest
from unittest import SkipTest

from openmdao.core.options import OptionsDictionary

class TestOptions(unittest.TestCase):
    def setUp(self):
        self.options = OptionsDictionary()

    def test_options_dictionary(self):
        with self.assertRaises(ValueError) as cm:
            self.options['junk']

        self.assertEqual("'junk' is not a valid option", str(cm.exception))

        self.options.add_option('atol', 1e-6, doc = 'Absolute tolerance for convergence')

        self.assertEqual(self.options['atol'], 1.0e-6)

        self.options.add_option('iprint', 0, values = [0, 1, 2, 3])
        map(self.options.__setitem__, ['iprint'] * 4, range(4))

        raise SkipTest('the following is not implemented yet')

        with self.assertRaises(ValueError) as cm:
            self.options['iprint'] = "Hello"

        self.assertEqual("'iprint' should be in '[0, 1, 2, 3]'", str(cm.exception))

        self.options.add_option('conmin_diff', True)
        map(self.options.__setitem__, ['conmin_diff']*2, [True, False])

        with self.assertRaises(ValueError) as cm:
            self.options['conmin_diff'] = "YES!"

        self.assertEqual("'conmin_diff' should be a boolean", str(cm.exception))

        self.options.add_option('maxiter', 10, low=0, high=10)
        map(self.options.__setitem__, ['maxiter'] * 11, xrange(0, 10))

        with self.assertRaises(ValueError) as cm:
            self.options['maxiter'] = 15

        self.assertEqual("max value for 'maxiter' is 10", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
