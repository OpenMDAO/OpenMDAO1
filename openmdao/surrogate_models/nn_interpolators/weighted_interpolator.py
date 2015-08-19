import numpy as np

from openmdao.surrogate_models.nn_interpolators.nn_base import NNBase


class WeightedInterpolator(NNBase):
    # Weighted Neighbor Interpolation

    @staticmethod
    def _get_weights(ndist, dist_eff):

        # Find the weighted neighbors per defined formula for distance effect
        dist = np.power(ndist, dist_eff)

        # Ignore Division by Zero Warnings
        with np.errstate(divide='ignore'):
            weights = 1.0 / dist

        # Replace any nan's to use the training data
        rows, cols = np.where(np.isinf(weights))

        weights[rows, :] = 0.
        weights[rows, cols] = 1.

        return weights

    def __call__(self, prediction_points, n=5, dist_eff=0):

        if self.ntpts < n:
            raise ValueError('WeightedInterpolant does not have sufficient '
                             'training data to use n={0}, only {1} points'
                             ' available.'.format(n, self.ntpts))

        if dist_eff == 0:
            # If default, use #dims + 1
            dist_eff = self.indep_dims + 1

        if len(prediction_points.shape) == 1:
            # Reshape vector to n x 1 array
            prediction_points.shape = (1, prediction_points.shape[0])

        normalized_pts = (prediction_points - self.tpm) / self.tpr
        nppts = normalized_pts.shape[0]
        # Find them neigbors
        # KData query takes (data, #ofneighbors) to determine closest
        # training points to predicted data
        ndist, nloc = self.KData.query(normalized_pts.real, n)

        # Setup problem

        # Reshape ndist for 1D problems.
        if len(ndist.shape) == 1:
            ndist.shape = (1, ndist.shape[0])
            nloc.shape = (1, nloc.shape[0])

        weights = self._get_weights(ndist, dist_eff)

        weight_sum = np.sum(weights, axis=1)
        vals = self.tv[nloc]
        wt = np.einsum('ijk,ij->ik', vals, weights)
        predz = ((wt / weight_sum[:, np.newaxis]) * self.tvr) + self.tvm

        return predz

    def gradient(self, prediction_points, n=5, dist_eff=0):

        if self.ntpts < n:
            raise ValueError('WeightedInterpolant does not have sufficient '
                             'training data to use n={0}, only {1} points'
                             ' available.'.format(n, self.ntpts))

        if dist_eff == 0:
            # If default, use #dims + 1
            dist_eff = self.indep_dims + 1

        if len(prediction_points.shape) == 1:
            # Reshape vector to n x 1 array
            prediction_points.shape = (1, prediction_points.shape[0])

        normalized_pts = (prediction_points - self.tpm) / self.tpr

        # KData query takes (data, #ofneighbors) to determine closest
        # training points to predicted data
        ndist, nloc = self.KData.query(normalized_pts.real, n)

        # Reshape ndist for 1D problems.
        if len(ndist.shape) == 1:
            ndist.shape = (1, ndist.shape[0])
            nloc.shape = (1, nloc.shape[0])

        dimdiff = normalized_pts - self.tp[nloc]

        weights = np.power(ndist, -dist_eff)
        dweights = -dist_eff * np.power(ndist[..., np.newaxis], -(dist_eff + 2)) * dimdiff

        weight_sum = np.sum(weights, axis=1)

        vals = self.tv[nloc]

        gradient = (weight_sum * np.einsum('ikj,ikl->ilj', dweights, vals)
                    - (np.einsum('ij,ijk->ik', weights, vals)[..., np.newaxis]
                    * np.sum(dweights, axis=1))) / np.power(weight_sum, 2)

        grad = gradient * (self.tvr[..., np.newaxis] / self.tpr)

        return grad
