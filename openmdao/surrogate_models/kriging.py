""" Surrogate model based on Kriging. """
from math import log

from openmdao.surrogate_models.surrogate_model import SurrogateModel

# pylint: disable-msg=E0611,F0401
from numpy import zeros, dot, ones, eye, abs, exp, log10, diagonal,\
    prod, square, column_stack, ndarray, sqrt, inf, einsum, sum, power
from numpy.linalg import slogdet, linalg
from numpy.dual import lstsq
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from six.moves import range


class KrigingSurrogate(SurrogateModel):
    """Surrogate Modeling method based on the simple Kriging interpolation.
    Predictions are returned as a tuple of mean and RMSE"""

    def __init__(self):
        super(KrigingSurrogate, self).__init__()

        self.m = 0       # number of independent
        self.n = 0       # number of training points
        self.thetas = zeros(0)
        self.nugget = 0     # nugget smoothing parameter from [Sasena, 2002]

        self.R = zeros(0)
        self.R_fact = None
        self.R_solve_ymu = zeros(0)
        self.R_solve_one = zeros(0)
        self.mu = zeros(0)
        self.log_likelihood = inf

        # Training Values
        self.X = zeros(0)
        self.Y = zeros(0)

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

        self.m = len(x[0])
        self.n = len(x)

        if self.n <= 1:
            raise ValueError(
                'KrigingSurrogate require at least 2 training points.'
            )

        self.X = x
        self.Y = y

        def _calcll(thetas):
            """ Callback function"""
            self.thetas = thetas
            self._calculate_log_likelihood()
            return -self.log_likelihood

        cons = []
        for i in range(self.m):
            cons.append({'type': 'ineq', 'fun': lambda logt: logt[i] - log10(1e-2)})  # min
            cons.append({'type': 'ineq', 'fun': lambda logt: log10(3) - logt[i]})     # max

        self.thetas = minimize(_calcll, zeros(self.m), method='COBYLA',
                               constraints=cons, tol=1e-8).x
        self._calculate_log_likelihood()

    def _calculate_log_likelihood(self):
        """
        Calculates the log-likelihood (up to a constant) for a given
        self.theta.

        """
        R = zeros((self.n, self.n))
        X, Y = self.X, self.Y
        thetas = power(10., self.thetas)

        # exponentially weighted distance formula
        for i in range(self.n):
            R[i, i+1:self.n] = exp(-thetas.dot(square(X[i, ...] - X[i+1:self.n, ...]).T))

        R *= (1.0 - self.nugget)
        R += R.T + eye(self.n)
        self.R = R

        one = ones(self.n)
        rhs = column_stack([Y, one])
        try:
            # Cholesky Decomposition
            self.R_fact = cho_factor(R)
            sol = cho_solve(self.R_fact, rhs)
            solve = lambda x: cho_solve(self.R_fact, x)
            det_factor = log(abs(prod(diagonal(self.R_fact[0])) ** 2) + 1.e-16)

        except (linalg.LinAlgError, ValueError):
            # Since Cholesky failed, try linear least squares
            self.R_fact = None  # reset this to none, so we know not to use Cholesky
            sol = lstsq(self.R, rhs)[0]
            solve = lambda x: lstsq(self.R, x)[0]
            det_factor = slogdet(self.R)[1]

        self.mu = dot(one, sol[:, :-1]) / dot(one, sol[:, -1])
        y_minus_mu = Y - self.mu
        self.R_solve_ymu = solve(y_minus_mu)
        self.R_solve_one = sol[:, -1]
        self.sig2 = dot(y_minus_mu.T, self.R_solve_ymu) / self.n

        if isinstance(self.sig2, ndarray):
            self.log_likelihood = -self.n/2. * slogdet(self.sig2)[1] \
                                  - 1./2.*det_factor
        else:
            self.log_likelihood = -self.n/2. * log(self.sig2) \
                                  - 1./2.*det_factor

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
        thetas = power(10., self.thetas)
        r = exp(-thetas.dot(square((x - X).T)))

        if self.R_fact is not None:
            # Cholesky Decomposition
            sol = cho_solve(self.R_fact, r).T
        else:
            # Linear Least Squares
            sol = lstsq(self.R, r)[0].T

        f = self.mu + dot(r, self.R_solve_ymu)
        term1 = dot(r, sol)

        # Note: sum(sol) should be 1, since Kriging is an unbiased
        # estimator. This measures the effect of numerical instabilities.
        bias = (1.0 - sum(sol)) ** 2. / sum(self.R_solve_one)

        mse = self.sig2 * (1.0 - term1 + bias)
        rmse = sqrt(abs(mse))

        return f, rmse

    def linearize(self, x):
        """
        Calculates the jacobian of the Kriging surface at the requested point.

        Args
        ----
        x : array-like
            Point at which the surrogate Jacobian is evaluated.
        """

        thetas = power(10., self.thetas)
        r = exp(-thetas.dot(square((x - self.X).T)))

        # Z = einsum('i,ij->ij', X, Y) is equivalent to, but much faster and
        # memory efficient than, diag(X).dot(Y) for vector X and 2D array Y.
        # I.e. Z[i,j] = X[i]*Y[i,j]
        gradr = r * -2 * einsum('i,ij->ij', thetas, (x - self.X).T)
        jac = gradr.dot(self.R_solve_ymu).T
        return jac


class FloatKrigingSurrogate(KrigingSurrogate):
    """Surrogate model based on the simple Kriging interpolation. Predictions are returned as floats,
    which are the mean of the model's prediction."""

    def predict(self, x):
        dist = super(FloatKrigingSurrogate, self).predict(x)
        return dist[0]  # mean value
