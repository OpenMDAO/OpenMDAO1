""" Surrogate model based on Kriging. """
from math import log

from openmdao.surrogate_models.surrogate_model import SurrogateModel

# pylint: disable-msg=E0611,F0401
from scipy.optimize import minimize
from six.moves import range, zip
import numpy as np
import scipy.linalg as linalg

MACHINE_EPSILON = np.finfo(np.double).eps

class KrigingSurrogate(SurrogateModel):
    """Surrogate Modeling method based on the simple Kriging interpolation.
    Predictions are returned as a tuple of mean and RMSE. Based on Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams. (see also: scikit-learn).

    Args
    ----
    nugget : double or ndarray, optional
        Nugget smoothing parameter for smoothing noisy data. Represents the variance of the input values.
        If nugget is an ndarray, it must be of the same length as the number of training points.
        Default: 10. * Machine Epsilon
    """

    def __init__(self, nugget=10. * MACHINE_EPSILON):
        super(KrigingSurrogate, self).__init__()

        self.n_dims = 0       # number of independent
        self.n_samples = 0       # number of training points
        self.thetas = np.zeros(0)
        self.nugget = nugget     # nugget smoothing parameter from [Sasena, 2002]

        self.alpha = np.zeros(0)
        self.L = np.zeros(0)
        self.sigma2 = np.zeros(0)

        # Normalized Training Values
        self.X = np.zeros(0)
        self.Y = np.zeros(0)
        self.X_mean = np.zeros(0)
        self.X_std = np.zeros(0)
        self.Y_mean = np.zeros(0)
        self.Y_std = np.zeros(0)

    def train(self, x, y):
        """
        Train the surrogate model with the given set of inputs and outputs.

        Args
        ----
        x : array-like
            Training input locations

        y : array-like
            Model responses at given inputs.
        """

        super(KrigingSurrogate, self).train(x, y)

        x, y = np.atleast_2d(x, y)

        self.n_samples, self.n_dims = x.shape

        if self.n_samples <= 1:
            raise ValueError(
                'KrigingSurrogate require at least 2 training points.'
            )

        # Normalize the data
        X_mean = np.mean(x, axis=0)
        X_std = np.std(x, axis=0)
        Y_mean = np.mean(y, axis=0)
        Y_std = np.std(y, axis=0)

        X_std[X_std == 0.] = 1.
        Y_std[Y_std == 0.] = 1.

        X = (x - X_mean) / X_std
        Y = (y - Y_mean) / Y_std

        self.X = X
        self.Y = Y
        self.X_mean, self.X_std = X_mean, X_std
        self.Y_mean, self.Y_std = Y_mean, Y_std

        def _calcll(thetas):
            """ Callback function"""
            return -self._calculate_reduced_likelihood_params(np.power(10., thetas))[0]

        cons = []
        for i in range(self.n_dims):
            cons.append({'type': 'ineq', 'fun': lambda logt: logt[i] - np.log10(1e-3)})  # min
            cons.append({'type': 'ineq', 'fun': lambda logt: np.log10(3) - logt[i]})     # max

        optResult = minimize(_calcll, 1e-1*np.ones(self.n_dims), method='cobyla',
                             constraints=cons)
        self.thetas = np.power(10., optResult.x)
        _, params = self._calculate_reduced_likelihood_params()
        self.alpha = params['alpha']
        self.L = params['L']
        self.sigma2 = params['sigma2']

    def _calculate_reduced_likelihood_params(self, thetas=None):
        """
        Calculates a quantity with the same maximum location as the log-likelihood for a given theta.

        Args
        ----
        thetas : ndarray, optional
            Given input correlation coefficients. If none given, uses self.thetas from training.
        """
        if thetas is None:
            thetas = self.thetas
        R = np.zeros((self.n_samples, self.n_samples))
        X, Y = self.X, self.Y


        params = {}
        # Correlation Matrix
        for i in range(self.n_samples):
            # squared exponential weighted distance formula
            R[i, i+1:self.n_samples] = np.exp(-thetas.dot(np.square(X[i, ...] - X[i + 1:self.n_samples, ...]).T))
        R += R.T
        R[np.diag_indices_from(R)] = 1. + self.nugget

        try:
            # Cholesky Decomposition
            L = linalg.cholesky(R, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf, params

        alpha = linalg.cho_solve((L, True), Y)
        sigma2 = np.dot(Y.T, alpha).sum(axis=0) / self.n_samples
        det_factor = np.prod(np.power(np.diag(L),  2./self.n_samples))
        reduced_likelihood = -np.sum(sigma2) * det_factor
        params['alpha'] = alpha
        params['L'] = L
        params['sigma2'] = sigma2 * np.square(self.Y_std)
        return reduced_likelihood, params

    def predict(self, x, eval_rmse=True):
        """
        Calculates a predicted value of the response based on the current
        trained model for the supplied list of inputs.

        Args
        ----
        x : array-like
            Point at which the surrogate is evaluated.
        eval_rmse : bool
            Flag indicating whether the Root Mean Squared Error (RMSE) should be computed.
        """

        super(KrigingSurrogate, self).predict(x)

        X, Y = self.X, self.Y
        thetas = self.thetas
        if isinstance(x, list):
            x = np.array(x)
        x = np.atleast_2d(x)
        n_eval = x.shape[0]
        n_outputs = self.Y.shape[1]

        # Normalize input
        x_n = (x - self.X_mean) / self.X_std
        y = np.zeros(n_eval)

        r = np.zeros((n_eval, self.n_samples))
        for r_i, x_i in zip(r, x_n):
            r_i[:] = np.exp(-thetas.dot(np.square((x_i - X).T)))

        # Scaled Predictor
        y_t = np.dot(r, self.alpha)

        # Predictor
        y = self.Y_mean + self.Y_std * y_t

        if eval_rmse:
            v = linalg.solve_triangular(self.L, r.T, lower=True)
            # np.einsum('ij,ij->j', v, v) = diag( <v^T, v> )
            mse = (1. - np.einsum('ij,ij->j', v, v)) * self.sigma2
            # Forcing negative RMSE to zero if negative due to machine precision
            mse[mse < 0.] = 0.

            return y, np.sqrt(mse)

        return y

    def linearize(self, x):
        """
        Calculates the jacobian of the Kriging surface at the requested point.

        Args
        ----
        x : array-like
            Point at which the surrogate Jacobian is evaluated.
        """

        thetas = self.thetas

        # Normalize Input
        x_n = (x - self.X_mean) / self.X_std

        r = np.exp(-thetas.dot(np.square((x_n - self.X).T)))

        # Z = einsum('i,ij->ij', X, Y) is equivalent to, but much faster and
        # memory efficient than, diag(X).dot(Y) for vector X and 2D array Y.
        # I.e. Z[i,j] = X[i]*Y[i,j]
        gradr = r * -2 * np.einsum('i,ij->ij', thetas, (x_n - self.X).T)
        jac = self.Y_std/self.X_std * gradr.dot(self.alpha).T
        return jac


class FloatKrigingSurrogate(KrigingSurrogate):
    """Surrogate model based on the simple Kriging interpolation. Predictions are returned as floats,
    which are the mean of the model's prediction."""

    def predict(self, x):
        dist = super(FloatKrigingSurrogate, self).predict(x, eval_rmse=False)
        return dist[0]  # mean value
