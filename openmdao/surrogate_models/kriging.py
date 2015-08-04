""" Surrogate model based on Kriging. """
from math import log

from openmdao.surrogate_models.surrogate_model import SurrogateModel

# pylint: disable-msg=E0611,F0401
from numpy import zeros, dot, ones, eye, abs, vstack, exp, log10,\
    power, diagonal, prod, square, hstack, ndarray, sqrt, inf, einsum
from numpy.linalg import slogdet, linalg, lstsq
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
                'KrigingSurrogates require at least 2 training points.'
            )

        # self.X = array(x)
        # self.Y = array(y)
        self.X = x
        self.Y = y

        thetas = zeros(self.m)

        def _calcll(thetas):
            ''' Callback function'''
            self.thetas = thetas
            self._calculate_log_likelihood()
            return -self.log_likelihood

        cons = []
        for i in range(self.m):
            cons.append({'type': 'ineq', 'fun': lambda log10t: log10t[i] - log10(1e-2)})  # min
            cons.append({'type': 'ineq', 'fun': lambda log10t: log10(3) - log10t[i]})     # max

        self.thetas = minimize(_calcll, thetas, method='COBYLA', constraints=cons, tol=1e-8).x
        self._calculate_log_likelihood()

    def _calculate_log_likelihood(self):
        """
        Calculates the log-likelihood (up to a constant) for a given
        self.theta.

        """
        R = zeros((self.n, self.n))
        X, Y = self.X, self.Y
        thetas = power(10., self.thetas)

        #weighted distance formula
        for i in range(self.n):
            R[i, i+1:self.n] = exp(-thetas.dot(square(X[i, ...] - X[i+1:self.n, ...]).T))

        R *= (1.0 - self.nugget)
        R += R.T + eye(self.n)
        self.R = R

        one = ones((self.n, 1))
        try:
            self.R_fact = cho_factor(R)
            rhs = hstack([Y, one])
            cho = cho_solve(self.R_fact, rhs)

            self.mu = dot(one.T, cho[:, :-1]) / dot(one.T, cho[:, -1])
            y_minus_mu = Y - self.mu
            self.R_solve_ymu = cho_solve(self.R_fact, y_minus_mu)

            self.sig2 = dot(y_minus_mu.T, self.R_solve_ymu)/self.n
            det_factor = abs(prod(diagonal(self.R_fact[0]))**2) + 1.e-16

            if isinstance(self.sig2, ndarray):
                self.log_likelihood = -self.n / 2. * slogdet(self.sig2)[1] - \
                    1. / 2. * log(det_factor)
            else:
                self.log_likelihood = -self.n/2.*log(self.sig2) - \
                    1./2.*log(det_factor)

        except (linalg.LinAlgError, ValueError):
            #------LSTSQ---------
            self.R_fact = None  # reset this to none, so we know not to use cholesky
            # self.R = self.R+diag([10e-6]*self.n)  # improve conditioning[Booker et al., 1999]
            rhs = hstack([Y, one])
            lsq = lstsq(self.R.T, rhs)[0]
            self.mu = dot(one.T, lsq[:, :-1])/dot(one.T, lsq[:, -1])
            y_minus_mu = Y - self.mu
            self.R_solve_ymu = lstsq(self.R, y_minus_mu)[0]
            self.sig2 = dot(y_minus_mu.T, self.R_solve_ymu)/self.n
            if isinstance(self.sig2, ndarray):
                self.log_likelihood = -self.n / 2. * slogdet(self.sig2)[1] - \
                                      1. / 2. * slogdet(self.R)[1]
            else:
                self.log_likelihood = -self.n / 2. * log(self.sig2) - \
                                      1. / 2. * slogdet(self.R)[1]

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
        # x = array(x)
        r = exp(-thetas.dot(square((X - x).T)))

        one = ones(self.n)
        if self.R_fact is not None:
            # ---CHOLESKY DECOMPOSTION ---

            rhs = vstack([r, one]).T
            cho = cho_solve(self.R_fact, rhs).T

            f = self.mu + dot(r, self.R_solve_ymu)
            term1 = dot(r, cho[0])
            term2 = (1.0 - dot(one, cho[0])) ** 2. / dot(one, cho[1])

        else:
            # -----LSTSQ-------
            rhs = vstack([r, one]).T
            lsq = lstsq(self.R.T, rhs)[0].T

            f = self.mu + dot(r, self.R_solve_ymu)
            term1 = dot(r, lsq[0])
            term2 = (1.0 - dot(one, lsq[0])) ** 2. / dot(one, lsq[1])

        MSE = self.sig2 * (1.0 - term1 + term2)
        RMSE = sqrt(abs(MSE))

        return (f, RMSE)

    def jacobian(self, x):
        thetas = power(10., self.thetas)
        r = exp(-thetas.dot(square((x - self.X).T)))
        gradr = r * -2 * einsum('i,ij->ij', thetas, (x - self.X).T)
        jac = gradr.dot(self.R_solve_ymu).T
        return jac


class FloatKrigingSurrogate(KrigingSurrogate):
    """Surrogate model based on the simple Kriging interpolation. Predictions are returned as floats,
    which are the mean of the model's prediction."""

    def predict(self, x):
        dist = super(FloatKrigingSurrogate, self).predict(x)
        return dist[0] # mean value
