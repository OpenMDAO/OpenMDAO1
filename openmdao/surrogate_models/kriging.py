""" Surrogate model based on Kriging. """
from math import log

from openmdao.surrogate_models.surrogate_model import SurrogateModel

# pylint: disable-msg=E0611,F0401
from scipy.optimize import minimize
from six.moves import range
import numpy as np
import scipy.linalg as linalg

MACHINE_EPSILON = np.finfo(np.double).eps

class KrigingSurrogate(SurrogateModel):
    """Surrogate Modeling method based on the simple Kriging interpolation.
    Predictions are returned as a tuple of mean and RMSE. Based on the DACE Matlab toolbox (see also: scikit-learn).

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

        self.beta = None
        self.gamma = None
        self.sigma2 = None
        self.L = None
        self.F_t = None
        self.G = None

        # Normalized Training Values
        self.X = None
        self.Y = None
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.F = None


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

        self.n_dims = len(x[0])
        self.n_samples = len(x)

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

        self.F = self._regression(X)

        def _calcll(thetas):
            """ Callback function"""
            return -self._calculate_log_likelihood_params(10. ** thetas)[0]

        cons = []
        for i in range(self.n_dims):
            cons.append({'type': 'ineq', 'fun': lambda logt: logt[i] - np.log10(1e-3)})  # min
            cons.append({'type': 'ineq', 'fun': lambda logt: np.log10(3) - logt[i]})     # max

        optResult = minimize(_calcll, 1e-1*np.ones(self.n_dims), method='COBYLA',
                             constraints=cons)
        self.thetas = 10. ** optResult.x
        likelihood, params = self._calculate_log_likelihood_params()
        self.log_likelihood = - likelihood
        self.beta = params['beta']
        self.gamma = params['gamma']
        self.sigma2 = params['sigma2']
        self.L = params['L']
        self.F_t = params['F_t']
        self.G = params['G']

    # Regression Matrix (assuming constant, this part can be extended to include other regressions)
    def _regression(self, x):
        return np.ones((x.shape[0], 1))

    def _calculate_log_likelihood_params(self, thetas=None):
        """
        Calculates the log-likelihood (up to a constant) for a given
        thetas.

        Args
        ----
        thetas : ndarray, optional
            Given input (log10) correlation coefficients. If none given, uses self.thetas from training.
        """
        if thetas is None:
            thetas = self.thetas
        R = np.zeros((self.n_samples, self.n_samples))
        X, Y = self.X, self.Y


        params = {}

        # Regression
        F = self.F

        # Correlation Matrix
        for i in range(self.n_samples):
            # squared exponential weighted distance formula
            R[i, i+1:self.n_samples] = np.exp(-thetas.dot(np.square(X[i, ...] - X[i + 1:self.n_samples, ...]).T))
        R += R.T + np.eye(self.n_samples) * (1. + self.nugget)

        try:
            # Cholesky Decomposition
            L = linalg.cholesky(R, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf, params

        # Least Squares Solution
        F_t = linalg.solve_triangular(L, F, lower=True)
        Q, G = linalg.qr(F_t, mode='economic')

        # Scikit-learn checks the conditioning of G, F here at the expense of an SVD.

        Y_t = linalg.solve_triangular(L, Y, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Y_t))
        rho = Y_t - np.dot(F_t, beta)
        sigma2 = (rho ** 2.).sum(axis=0) / self.n_samples

        # det(R) = det(L)*det(L^T) = [triangular] prod(diag(L))*prod(diag(L^T)) = prod(diag(L))^2
        # likelihood = -sigma^2 det(R)^(1/n_samples)
        det_factor = np.prod(np.diag(L) ** (2./self.n_samples))

        log_likelihood = -np.sum(sigma2) * det_factor
        params['sigma2'] = sigma2 * self.Y_std ** 2.
        params['beta'] = beta
        params['gamma'] = linalg.solve_triangular(L.T, rho)
        params['L'] = L
        params['F_t'] = F_t
        params['G'] = G

        return log_likelihood, params

    def predict(self, x):
        """
        Calculates a predicted value of the response based on the current
        trained model for the supplied list of inputs.

        Args
        ----
        x : array-like
            Point at which the surrogate is evaluated.
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

        r = np.exp(-thetas.dot(np.square((x_n - X).T)))
        f = self._regression(x_n)

        # Scaled Predictor
        y_t = np.dot(f, self.beta) + np.dot(r, self.gamma)

        # Predictor
        y = self.Y_mean + self.Y_std * y_t

        r_t = linalg.solve_triangular(self.L, r.T, lower=True)
        u = linalg.solve_triangular(self.G.T, np.dot(self.F_t.T, r_t) - f.T, lower=True)
        mse = np.dot(self.sigma2.reshape(n_outputs, 1),
                     (1. - (r_t ** 2.).sum(axis=0) +
                     (u ** 2.).sum(axis=0))[np.newaxis, :])
        # Forcing negative RMSE to zero if negative due to machine precision
        mse[mse < 0.] = 0.
        
        mse = np.sqrt((mse ** 2.).sum(axis=0) / n_outputs)

        return y, np.sqrt(mse)

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
        jac = self.Y_std/self.X_std * gradr.dot(self.gamma).T
        return jac


class FloatKrigingSurrogate(KrigingSurrogate):
    """Surrogate model based on the simple Kriging interpolation. Predictions are returned as floats,
    which are the mean of the model's prediction."""

    def predict(self, x):
        dist = super(FloatKrigingSurrogate, self).predict(x)
        return dist[0]  # mean value
