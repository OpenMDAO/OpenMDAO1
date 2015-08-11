# Based off of the N-Dimensional Interpolation library by Stephen Marone.
# https://github.com/SMarone/NDInterp

import numpy as np

from collections import OrderedDict
from math import ceil
from openmdao.surrogate_models.surrogate_model import SurrogateModel
from scipy.spatial import cKDTree
from six.moves import range


class _NNBase(object):
    """
    Base class for common functionality between nearest neighbor interpolants.
    """
    def __init__(self, TrnPts, TrnVals, NumLeaves=2):
        super(_NNBase, self).__init__()
        # TrainPts and TrainVals are the known points and their
        # respective values which will be interpolated against.
        # Grab the mins and normals of each dimension
        self.tpm = np.amin(TrnPts, axis=0)
        self.tpr = (np.amax(TrnPts, axis=0) - self.tpm)
        self.tvm = np.amin(TrnVals, axis=0)
        self.tvr = (np.amax(TrnVals, axis=0) - self.tvm)

        # This prevents against colinear data (range = 0)
        self.tpr[self.tpr == 0] = 1
        self.tvr[self.tvr == 0] = 1

        # Normalize all points
        self.tp = (TrnPts - self.tpm) / self.tpr
        self.tv = (TrnVals - self.tvm) / self.tvr

        # Set important variables
        self.indep_dims = TrnPts.shape[1]
        self.dep_dims = TrnVals.shape[1]
        self.ntpts = TrnPts.shape[0]

        # Make training data into a Tree
        leavesz = ceil(self.ntpts / float(NumLeaves))
        self.KData = cKDTree(self.tp, leafsize=leavesz)


class LinearInterpolator(_NNBase):
    def main(self, nppts, nloc):
        # Extra Inputs for Finding the normal are found below
        # Number of row vectors needed always dimensions - 1
        indep_dims = self.indep_dims
        dep_dims = self.dep_dims

        # Preallocate storage
        pc = np.empty((nppts, dep_dims), dtype='float')
        normal = np.empty((nppts, indep_dims + 1, dep_dims), dtype='float')
        nvect = np.empty((nppts, indep_dims, indep_dims + 1), dtype='float')
        trnd = np.concatenate((self.tp[nloc, :],
                               self.tv[nloc, 0].reshape(nppts, indep_dims + 1, 1)),
                              axis=2)
        nvect[:, :, :-1] = trnd[:, 1:, :-1] - trnd[:, :-1, :-1]

        for i in range(dep_dims):
            # Planar vectors need both dep and ind dimensions
            trnd[:, :, -1] = self.tv[nloc, i]

            # Go through each neighbor
            # Creates array[neighbor, dimension] from NN results
            nvect[:, :, -1] = trnd[:, 1:, -1] - trnd[:, :-1, -1]

            # Normal vector is in the null space of nvect.
            # Since nvect is of size indep x (indep + 1),
            # the normal vector will be the last entry in
            # V in the U, Sigma, V = svd(nvect).
            normal[:, :, i] = np.linalg.svd(nvect)[2][:, -1, :]

            # Use the point of the closest neighbor to
            # solve for pc - the constant of the n-dimensional plane.
            pc[:, i] = np.einsum('ij,ij->i', trnd[:, 0, :], normal[:, :, i])
        return normal, pc

    def main2D(self, PrdPts, nppts, nloc):
        # Need to find a tangent instead of a normal, y=mx+b
        d = self.tp[nloc[:, 1], 0] - self.tp[nloc[:, 0], 0]
        m = (self.tv[nloc[:, 1], 0] - self.tv[nloc[:, 0], 0]) / d
        b = self.tv[nloc[:, 0], 0] - (m * self.tp[nloc[:, 0], 0])
        m.shape = (nppts, 1)
        b.shape = (nppts, 1)
        return m, b

    def __call__(self, PredPoints):
        # This method uses linear interpolation by defining a plane with
        # a set number of nearest neighbors to the predicted

        if len(PredPoints.shape) == 1:
            # Reshape vector to n x 1 array
            PredPoints.shape = (1, PredPoints.shape[0])

        normPredPts = (PredPoints - self.tpm) / self.tpr
        nppts = normPredPts.shape[0]
        # Linear interp only uses as many neighbors as it has dimensions
        dims = self.indep_dims + 1
        # Find them neigbors.  Linear only uses #neighs=#dims
        # KData query takes (data, #ofneighbors) to determine closest
        # training points to predicted data
        ndist, nloc = self.KData.query(normPredPts.real, dims)

        # Need to ensure there are enough dimensions to find the normal with
        if self.indep_dims > 1:
            normal, pc = self.main(nppts, nloc)
            # raise ValueError(normal, prdtemp, pc)
            # Set all prdz from values on plane
            prdz = np.einsum('ij,ijk->ik', normPredPts, normal[:, :self.indep_dims, :]) - pc
            # Check to see if there are any colinear points and replace them
            n0 = np.where(normal[:, -1, :] == 0)
            prdz[n0, :] = self.tv[nloc[0, n0], :]
            # Finish computation for the good normals
            n = np.where(normal[:, -1, :] != 0)
            prdz[n] /= -normal[:, -1, :][n]

        else:
            m, b = self.main2D(normPredPts, nppts, nloc)
            prdz = (m * normPredPts) + b

        predz = (prdz * self.tvr) + self.tvm
        return predz

    def gradient(self, PredPoints):
        # Extra method to find the gradient at each location of a set of
        # supplied predicted points.

        if len(PredPoints.shape) == 1:
            # Reshape vector to n x 1 array
            PredPoints.shape = (1, PredPoints.shape[0])

        normPredPts = (PredPoints - self.tpm) / self.tpr
        nppts = normPredPts.shape[0]
        gradient = np.zeros((nppts, self.dep_dims, self.indep_dims), dtype="float")
        # Linear interp only uses as many neighbors as it has dimensions
        dims = self.indep_dims + 1
        # Find them neighbors
        ndist, nloc = self.KData.query(normPredPts.real, dims)
        # Need to ensure there are enough dimensions to find the normal with
        if self.indep_dims > 1:
            normal, pc = self.main(nppts, nloc)
            if np.any(normal[:, -1, :]) == 0:
                return gradient
            gradient[:] = (-normal[:, :-1, :] / normal[:, -1, :]).squeeze().T

        else:
            gradient[:, 0, :], b = self.main2D(normPredPts, nppts, nloc)

        grad = gradient * (self.tvr[:, np.newaxis] / self.tpr)
        return grad


_interpolators = OrderedDict([('linear', LinearInterpolator),
                  ('weighted', None),
                  ('cosine', None),
                  ('hermite', None),
                  ('rbf', None)])


class NearestNeighbor(SurrogateModel):
    def __init__(self, interpolant_type='rbf'):
        super(NearestNeighbor, self).__init__()

        if interpolant_type not in _interpolators.keys():
            msg = "NearestNeighbor: interpolant_type '{0}' not supported." \
                  " interpolant_type must be one of {1}.".format(
                      interpolant_type, list(_interpolators.keys())
                  )
            raise ValueError(msg)

        self.interpolant_type = interpolant_type
        self.interpolant = None

    def train(self, x, y):
        super(NearestNeighbor, self).train(x, y)
        self.interpolant = _interpolators[self.interpolant_type](x, y)

    def predict(self, x):
        super(NearestNeighbor, self).predict(x)
        return self.interpolant(x)

    def jacobian(self, x):
        jac = self.interpolant.gradient(x)
        if jac.shape[0] == 1:
            return jac[0, ...]
        return jac
