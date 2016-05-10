"""
OpenMDAO design-of-experiments Driver implementing the Latin Hypercube and Optimized Latin Hypercube methods.
"""

from collections import OrderedDict
import os
from random import shuffle, randint, seed

from six import iteritems, itervalues
from six.moves import range, zip

import numpy as np

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from openmdao.util.array_util import evenly_distrib_idxs

trace = os.environ.get('OPENMDAO_TRACE')
from openmdao.core.mpi_wrap import debug


class LatinHypercubeDriver(PredeterminedRunsDriver):
    """Design-of-experiments Driver implementing the Latin Hypercube method.

    Args
    ----
    num_samples : int, optional
        The number of samples to run. Defaults to 1.

    seed : int or None, optional
        Random seed.  Defaults to None.

    num_par_doe : int, optional
        The number of DOE cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, num_samples=1, seed=None, num_par_doe=1, load_balance=False):
        super(LatinHypercubeDriver, self).__init__(num_par_doe=num_par_doe,
                                                   load_balance=load_balance)
        self.num_samples = num_samples
        self.seed = seed

    def _build_runlist(self):
        """Build a runlist based on the Latin Hypercube method."""
        design_vars = self.get_desvar_metadata()

        # Add up sizes
        self.num_design_vars = sum(meta['size'] for meta in itervalues(design_vars))

        if self.seed is not None:
            seed(self.seed)
            np.random.seed(self.seed)

        # Generate an LHC of the proper size
        rand_lhc = self._get_lhc()

        # Map LHC to buckets
        buckets = OrderedDict()
        j = 0

        for (name, bounds) in iteritems(design_vars):
            buckets[name] = []

            # Support for array desvars
            val = self.root.unknowns._dat[name].val
            nval = bounds['size']

            for k in range(nval):

                lowb = bounds['lower']
                upb = bounds['upper']
                if isinstance(lowb, np.ndarray):
                    lowb = lowb[k]
                if isinstance(upb, np.ndarray):
                    upb = upb[k]

                design_var_buckets = self._get_buckets(lowb, upb)
                buckets[name].append([design_var_buckets[rand_lhc[i, j]]
                                      for i in range(self.num_samples)])
                j += 1

        # Return random values in given buckets
        for i in range(self.num_samples):
            sample = []
            for key, bounds in iteritems(buckets):
                sample.append([key, np.array([np.random.uniform(bounds[k][i][0],
                                                                bounds[k][i][1])
                                              for k in range(design_vars[key]['size'])])
                               ])
            yield sample

    def _distrib_build_runlist(self):
        """
        Returns an iterator over only those cases meant to execute
        in the current rank as part of a parallel DOE. A latin hypercube,
        unlike some other DOE generators, is created in one rank and then
        the appropriate cases are scattered to the appropriate ranks.
        """
        comm = self._full_comm

        # get the par_doe_id from every rank in the full comm so we know which
        # cases to scatter where
        doe_ids = comm.allgather(self._par_doe_id)

        job_list = None
        if comm.rank == 0:
            if trace:
                debug('Parallel DOE using %d procs' % self._num_par_doe)
            run_list = [list(case) for case in self._build_runlist()] # need to run iterator

            run_sizes, run_offsets = evenly_distrib_idxs(self._num_par_doe,
                                                         len(run_list))
            jobs = [run_list[o:o+s] for o, s in zip(run_offsets, run_sizes)]

            job_list = [jobs[i] for i in doe_ids]

        if trace: debug("scattering job_list: %s" % job_list)
        run_list = comm.scatter(job_list, root=0)
        if trace: debug('Number of DOE jobs: %s (scatter DONE)' % len(run_list))

        for case in run_list:
            yield case

    def _get_lhc(self):
        """Generates a Latin Hypercube based on the number of samples and the
        number of design variables.
        """

        rand_lhc = _rand_latin_hypercube(self.num_samples, self.num_design_vars)
        return rand_lhc.astype(int)

    def _get_buckets(self, low, high):
        """Determines the distribution of samples."""
        bucket_walls = np.linspace(low, high, self.num_samples + 1)
        return list(zip(bucket_walls[0:-1], bucket_walls[1:]))


class OptimizedLatinHypercubeDriver(LatinHypercubeDriver):
    """Design-of-experiments Driver implementing the Morris-Mitchell method for
    an Optimized Latin Hypercube.
    """

    def __init__(self, num_samples=1, seed=None, population=20, generations=2,
                norm_method=1, num_par_doe=1, load_balance=False):
        super(OptimizedLatinHypercubeDriver, self).__init__(num_par_doe=num_par_doe,
                                                            load_balance=load_balance)
        self.qs = [1, 2, 5, 10, 20, 50, 100]  # List of qs to try for Phi_q optimization
        self.num_samples = num_samples
        self.seed = seed
        self.population = population
        self.generations = generations
        self.norm_method = norm_method

    def _get_lhc(self):
        """Generate an Optimized Latin Hypercube
        """

        rand_lhc = _rand_latin_hypercube(self.num_samples, self.num_design_vars)

        # Optimize our LHC before returning it
        best_lhc = _LHC_Individual(rand_lhc, q=1, p=self.norm_method)
        for q in self.qs:
            lhc_start = _LHC_Individual(rand_lhc, q, self.norm_method)
            lhc_opt = _mmlhs(lhc_start, self.population, self.generations)
            if lhc_opt.mmphi() < best_lhc.mmphi():
                best_lhc = lhc_opt

        return best_lhc._get_doe().astype(int)


class _LHC_Individual(object):
    def __init__(self, doe, q=2, p=1):
        self.q = q
        self.p = p
        self.doe = doe
        self.phi = None  # Morris-Mitchell sampling criterion

    @property
    def shape(self):
        """Size of the LatinHypercube DOE (rows,cols)."""

        return self.doe.shape

    def mmphi(self):
        """Returns the Morris-Mitchell sampling criterion for this Latin
        hypercube.
        """

        if self.phi is None:
            distdict = {}

            # Calculate the norm between each pair of points in the DOE
            arr = self.doe
            n, m = arr.shape
            for i in range(1, n):
                nrm = np.linalg.norm(arr[i] - arr[:i], ord=self.p, axis=1)
                for j in range(0, i):
                    nrmj = nrm[j]
                    if nrmj in distdict:
                        distdict[nrmj] += 1
                    else:
                        distdict[nrmj] = 1

            size = len(distdict)

            distinct_d = np.fromiter(distdict, dtype=float, count=size)

            # Mutltiplicity array with a count of how many pairs of points
            # have a given distance
            J = np.fromiter(itervalues(distdict), dtype=int, count=size)

            self.phi = sum(J * (distinct_d ** (-self.q))) ** (1.0 / self.q)

        return self.phi

    def perturb(self, mutation_count):
        """ Interchanges pairs of randomly chosen elements within randomly chosen
        columns of a DOE a number of times. The result of this operation will also
        be a Latin hypercube.
        """

        new_doe = self.doe.copy()
        n, k = self.doe.shape
        for count in range(mutation_count):
            col = randint(0, k - 1)

            # Choosing two distinct random points
            el1 = randint(0, n - 1)
            el2 = randint(0, n - 1)
            while el1 == el2:
                el2 = randint(0, n - 1)

            new_doe[el1, col] = self.doe[el2, col]
            new_doe[el2, col] = self.doe[el1, col]

        return _LHC_Individual(new_doe, self.q, self.p)

    def __iter__(self):
        return self._get_rows()

    def _get_rows(self):
        for row in self.doe:
            yield row

    def __repr__(self):
        return repr(self.doe)

    def __str__(self):
        return str(self.doe)

    def __getitem__(self, *args):
        return self.doe.__getitem__(*args)

    def _get_doe(self):
        return self.doe


def _rand_latin_hypercube(n, k):
    # Calculates a random Latin hypercube set of n points in k dimensions
    # within [0,n-1]^k hypercube.
    arr = np.zeros((n, k))
    row = list(range(0, n))
    for i in range(k):
        shuffle(row)
        arr[:, i] = row
    return arr


def _is_latin_hypercube(lh):
    """Returns True if the given array is a Latin hypercube.
    The given array is assumed to be a numpy array.
    """

    n, k = lh.shape
    for j in range(k):
        col = lh[:, j]
        colset = set(col)
        if len(colset) < len(col):
            return False  # something was duplicated
    return True


def _mmlhs(x_start, population, generations):
    """Evolutionary search for most space filling Latin-Hypercube.
    Returns a new LatinHypercube instance with an optimized set of points.
    """

    x_best = x_start
    phi_best = x_start.mmphi()
    n = x_start.shape[1]

    level_off = np.floor(0.85 * generations)
    for it in range(generations):
        if it < level_off and level_off > 1.:
            mutations = int(round(1 + (0.5 * n - 1) * (level_off - it) / (level_off - 1)))
        else:
            mutations = 1

        x_improved = x_best
        phi_improved = phi_best

        for offspring in range(population):
            x_try = x_best.perturb(mutations)
            phi_try = x_try.mmphi()

            if phi_try < phi_improved:
                x_improved = x_try
                phi_improved = phi_try

        if phi_improved < phi_best:
            phi_best = phi_improved
            x_best = x_improved

    return x_best
