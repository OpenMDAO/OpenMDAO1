import numpy as np

from math import ceil
from scipy.spatial import cKDTree

class NNBase(object):
    """
    Base class for common functionality between nearest neighbor interpolants.
    """

    def __init__(self, training_points, training_values, num_leaves=2):
        """
        Initialize the nearest neighbor interpolant by scaling input to the
        unit hypercube.

        Args
        ----
        training_points : ndarray
            ndarray of shape (num_points x independent dims) containing
            training inpit locations.

        training_values : ndarray
            ndarray of shape (num_points x dependent dims) containing
            training output values.

        """
        # training_points and training_values are the known points and their
        # respective values which will be interpolated against.
        # Grab the mins and ranges of each dimension
        self.tpm = np.amin(training_points, axis=0)
        self.tpr = (np.amax(training_points, axis=0) - self.tpm)
        self.tvm = np.amin(training_values, axis=0)
        self.tvr = (np.amax(training_values, axis=0) - self.tvm)

        # This prevents against collinear data (range = 0)
        self.tpr[self.tpr == 0] = 1
        self.tvr[self.tvr == 0] = 1

        # Normalize all points
        self.tp = (training_points - self.tpm) / self.tpr
        self.tv = (training_values - self.tvm) / self.tvr

        # Record number of dimensions and points
        self.indep_dims = training_points.shape[1]
        self.dep_dims = training_values.shape[1]
        self.ntpts = training_points.shape[0]

        # Make training data into a Tree
        leavesz = ceil(self.ntpts / float(num_leaves))
        self.KData = cKDTree(self.tp, leafsize=leavesz)
