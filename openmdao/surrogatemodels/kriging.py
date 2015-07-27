""" Surrogate model based on Kriging. """
from math import log, e, sqrt

from openmdao.surrogatemodels.surrogate_model import SurrogateModel

# pylint: disable-msg=E0611,F0401
from numpy import array, zeros, dot, ones, eye, abs, vstack, exp, sum, log10,\
    power, diagonal, prod
from numpy.linalg import slogdet, linalg, lstsq
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from six.moves import range


class KrigingSurrogate(SurrogateModel):
    """Surrogate Modeling method based on the simple Kriging interpolation.
    Predictions are returned as a NormalDistribution instance."""

    def __init__(self):
        super(KrigingSurrogate, self).__init__()

        self.m = 0       # number of independent
        self.n = 0       # number of training points
        self.thetas = None
        self.nugget = 0     # nugget smoothing parameter from [Sasena, 2002]

        self.R = None
        self.R_fact = None
        self.mu = None
        self.log_likelihood = None

    def train(self, x, y):
        """Train the surrogate model with the given set of inputs and outputs."""
        
        super(KrigingSurrogate, self).train(x, y)
        
        #TODO: Check if one training point will work... if not raise error
        """self.X = []
        self.Y = []
        for ins,out in zip(X,Y):
            if ins not in self.X:
                self.X.append(ins)
                self.Y.append(out)
            else: "duplicate training point" """

        self.X = array(x)
        self.Y = array(y)
        self.m = len(x[0])
        self.n = len(x)

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
        #if self.m == None:
        #    Give error message
        R = zeros((self.n, self.n))
        X, Y = self.X, self.Y
        thetas = power(10., self.thetas)

        #weighted distance formula
        for i in range(self.n):
            R[i, i+1:self.n] = exp(-sum(thetas*(X[i] - X[i+1:self.n])**2., 1))

        R *= (1.0 - self.nugget)
        R += R.T + eye(self.n)
        self.R = R

        one = ones(self.n)
        try:
            self.R_fact = cho_factor(R)
            rhs = vstack([Y, one]).T
            R_fact = (self.R_fact[0].T, not self.R_fact[1])
            cho = cho_solve(R_fact, rhs).T

            self.mu = dot(one, cho[0])/dot(one, cho[1])
            ymdotone = Y - dot(one, self.mu)
            self.sig2 = dot(ymdotone, cho_solve(self.R_fact,
                                                (ymdotone)))/self.n

            det_factor = abs(prod(diagonal(R_fact[0]))**2) + 1.e-16

            self.log_likelihood = -self.n/2.*log(self.sig2) - \
                1./2.*log(det_factor)

        except (linalg.LinAlgError, ValueError):
            #------LSTSQ---------
            self.R_fact = None  # reset this to none, so we know not to use cholesky
            # self.R = self.R+diag([10e-6]*self.n)  # improve conditioning[Booker et al., 1999]
            rhs = vstack([Y, one]).T
            lsq = lstsq(self.R.T, rhs)[0].T
            self.mu = dot(one, lsq[0])/dot(one, lsq[1])
            ymdotone = Y - dot(one, self.mu)
            self.sig2 = dot(ymdotone, lstsq(self.R, ymdotone)[0])/self.n
            self.log_likelihood = -self.n/2.*log(self.sig2) - \
                1./2.*(slogdet(self.R)[1])
            #print self.log_likelihood

    def predict(self, x):
        """Calculates a predicted value of the response based on the current
        trained model for the supplied list of inputs.
        """

        super(KrigingSurrogate, self).predict(x)

        X, Y = self.X, self.Y
        thetas = power(10., self.thetas)
        x = array(x)
        r = exp(-thetas.dot(power((X - x).T, 2)))

        one = ones(self.n)
        if self.R_fact is not None:
            # ---CHOLESKY DECOMPOSTION ---
            # f = self.mu+dot(r,cho_solve(self.R_fact,Y-dot(one,self.mu)))
            # term1 = dot(r,cho_solve(self.R_fact,r))
            # term2 = (1.0-dot(one,cho_solve(self.R_fact,r)))**2./dot(one,cho_solve(self.R_fact,one))

            rhs = vstack([(Y - dot(one, self.mu)), r, one]).T
            R_fact = (self.R_fact[0].T, not self.R_fact[1])
            cho = cho_solve(R_fact, rhs).T

            f = self.mu + dot(r, cho[0])
            term1 = dot(r, cho[1])
            term2 = (1.0 - dot(one, cho[1])) ** 2. / dot(one, cho[2])

        else:
            # -----LSTSQ-------
            rhs = vstack([(Y - dot(one, self.mu)), r, one]).T
            lsq = lstsq(self.R.T, rhs)[0].T

            f = self.mu + dot(r, lsq[0])
            term1 = dot(r, lsq[1])
            term2 = (1.0 - dot(one, lsq[1])) ** 2. / dot(one, lsq[2])
            """
      #-----LSTSQ-------
      rhs = vstack([(Y-dot(one, self.mu)), r, one]).T
      lsq = lstsq(self.R.T, rhs)[0].T

      f = self.mu + dot(r, lsq[0])
      term1 = dot(r, lsq[1])
      term2 = (1.0 - dot(one, lsq[1]))**2./dot(one, lsq[2])
      """
        MSE = self.sig2 * (1.0 - term1 + term2)
        RMSE = sqrt(abs(MSE))

        return (f, RMSE)


class FloatKrigingSurrogate(KrigingSurrogate):
    """Surrogate model based on the simple Kriging interpolation. Predictions are returned as floats,
    which are the mean of the NormalDistribution predicted by the model."""

    def predict(self, x):
        dist = super(FloatKrigingSurrogate, self).predict(x)
        return dist[0] # mean value
