
import unittest
import numpy as np

from openmdao.util.array_util import SubArray


class TestSubArray(unittest.TestCase):
    def test_contains(self):
        idxarray = np.arange(0, 9, dtype=int)
        sub = SubArray(idxarray)
        self.assertTrue(0 in sub)
        self.assertTrue(8 in sub)
        self.assertFalse(9 in sub)
        self.assertTrue(isinstance(sub._idx, slice))

        idxarray = np.arange(0, 9, 2, dtype=int)
        sub = SubArray(idxarray)
        for i in [0,2,4,6,8]:
            self.assertTrue(i in sub)
        for i in [1,3,5,7,9]:
            self.assertFalse(i in sub)
        self.assertTrue(isinstance(sub._idx, slice))




if __name__ == '__main__':
    unittest.main()
