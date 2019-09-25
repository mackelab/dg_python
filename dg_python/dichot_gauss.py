import numpy as np
from scipy.stats import multivariate_normal as mnorm
from numpy.linalg import LinAlgError
import warnings


def heaviside(_input, center=0):
    """Implements sgn(_input - center)."""
    spikes = np.zeros_like(_input)
    spikes[_input > center] = 1.
    return spikes


def cov_to_corr(cov):
    """Converts input covariance matrix into correlation matrix."""
    std = np.sqrt(np.diag(cov))
    std_mat = np.outer(std, std)
    return cov / (std_mat + 1e-8)


def make_symmetric(M):
    """Makes input matrix symmetric, if it is non-symmetric."""
    M_copy = M
    if np.any(M != M.T):
        print('made symmetric')
        tril_inds = np.tril_indices(len(M), -1)
        M_copy[tril_inds] = M[tril_inds[1], tril_inds[0]].flatten()
    return M_copy


class WarningDG(UserWarning):
    pass


class Higham:
    """ Converts an input symmetric matrix M into a positive semi-definite matrix A using the Higham iterative
        projection algorithm to minimize the Frobenius norm between A and M.
        Reference: NJ Higham, Computing the nearest correlation matrix - a problem from finance, IMA Journal of
        Numerical Analysis, 2002

        Inputs:
        maxiters: max. number of iterations for iterative projection algorithm. Default is 100,000.
        tol: tolerance value for Frobenius norm. Default is 1e-10.
    """

    def __init__(self, maxiters=1e5, tol=1e-10):
        self.maxiters = maxiters
        self.tol = tol

    def projection_S(self, M):
        eigval, eigvec = np.linalg.eig(M)
        eigval[eigval < 0.] = 0.
        return eigvec.dot(np.diag(eigval).dot(eigvec.T))

    def projection_U(self, M):
        U = np.diag(np.diag(M - np.eye(len(M))))
        return M - U

    def higham_correction(self, M):

        it = 0
        DS = 0.
        Yo = M
        Xo = M
        delta = np.inf
        # triu_inds = np.triu_indices(len(cov), 1)

        while (it < self.maxiters) and (delta > self.tol):
            R = Yo - DS
            Xn = self.projection_S(R)
            DS = Xn - R
            Yn = self.projection_U(Xn)

            del_x = max(np.abs(Xn - Xo).sum(1)) / max(np.abs(Xn).sum(1))
            del_y = max(np.abs(Yn - Yo).sum(1)) / max(np.abs(Yn).sum(1))
            del_xy = max(np.abs(Yn - Xn).sum(1)) / max(np.abs(Yn).sum(1))
            delta = max(del_x, del_y, del_xy)
            Xo = Xn
            Yo = Yn

            it += 1
        if it == self.maxiters:
            warnings.warn("Iteration limit reached without convergence.", WarningDG)
            print('Frobenius norm:', del_x, del_y, del_xy)

        eigvals, eigvec = np.linalg.eig(Yn)
        if min(eigvals) < 0:
            warnings.warn("Higham corrected matrix was not positive definite. Converting into pd matrix.",
                          WarningDG)
            eigvals[eigvals < 0.] = 1e-6
            Yn = eigvec.dot(np.diag(eigvals).dot(eigvec.T))
            Yn = cov_to_corr(Yn)
            Yn = 0.5 * (Yn + Yn.T)

        return Yn.real


class DichotGauss:
    """
        Creates dichotomous Gaussian model. The model takes the mean and correlation of a multivariate Gaussian as
        input and generates binary population spike trains, assuming that they are independent across timebins, but have
        fixed correlations across neurons in each timebin.

        Inputs:
            :param num_neur: number of neurons.
            :param mean: mean of multivariate Gaussian. Default is zero for all timebins and neurons.
            :param corr: fixed correlation matrix for multivariate Gaussian, assumed to be symmetric.
            Default is the identity matrix.
            :param make_pd: set to True to make input correlation matrix positive definite using Higham algorithm.
            :param kwargs: hyper-parameters for class Higham which performs the Higham correction (see help(Higham)).
    """

    def __init__(self, num_neur, mean=None, corr=None, make_pd=False, **kwargs):
        super(DichotGauss, self).__init__()
        self.num_neur = num_neur

        self.tril_inds = np.tril_indices(self.num_neur, -1)
        self.make_pd = make_pd
        self.higham = Higham(**kwargs)

        if mean is None:
            mean = np.zeros((1, self.num_neur))

        if corr is None:    # Generate default identity correlation matrix
            corr = np.eye(self.num_neur)
            self.make_pd = False

        if self.make_pd is True:    # Make input correlation matrix positive
            corr = self.do_higham_correction(make_symmetric(corr))

        self.mean = mean
        self.corr = corr

    def sample(self, mean=None, corr=None, repeats=1):
        """
        Samples binary spike trains from DG model with input mean and correlation matrix.
        Inputs:
            :param mean: mean of multivariate Gaussian of size (timebins x number of neurons).
            :param corr: correlation matrix for multivariate Gaussian of size (number of neurons x number of neurons).
            :param repeats: number of binary spike trains to generate for the given mean and covariance matrix.
        Returns:
            :return: binary spike count tensor of size timebins x repeats x neurons
        """
        if mean is None:
            mean = self.mean
        if corr is None:
            corr = self.corr

        # Check if input mean and corr are of correct size
        self._check_mean_size(mean)
        self._check_corr_size(corr)

        # Check if input correlation matrix is positive definite, and do Higham correction if required
        self.do_higham_correction(corr)

        z = mnorm(np.zeros(self.num_neur),
                  cov=self.corr
                  ).rvs(size=[repeats, len(mean)])  # Generate samples from a multivariate Gaussian
        z = z.reshape(repeats, -1, self.num_neur)
        z = z + mean
        return heaviside(z.transpose(1, 0, 2))

    def do_higham_correction(self, M):
        """
        Finds nearest positive definite matrix to the input matrix using the Higham algorithm.
        """
        is_pd = self._check_pd(M)  # Check if input matrix is already positive definite.
        if is_pd is False:
            if self.make_pd is False:  # Raise warning if input matrix is not pd, and make_pd is False.
                warnings.warn('Input covariance matrix is not positive definite. Set \'make_pd\' to True to do Higham '
                              'correction.',
                              WarningDG)
                raise NotImplementedError
            else:
                warnings.warn('Input covariance matrix is not positive definite. Doing Higham correction.', WarningDG)
                M = self.higham.higham_correction(M)
        return M

    def _check_pd(self, cov):
        """Checks if input covariance matrix is positive definite."""
        try:
            np.linalg.cholesky(cov)
            return True
        except LinAlgError:
            return False

    def _check_corr_size(self, _input):
        """Checks if input correlation matrix has the required shape."""
        if np.all(list(_input.shape) == [self.num_neur, self.num_neur]) is False:
            warnings.filterwarnings(action="error",
                                    message="Shape mismatch. Input matrix should be of size "
                                            "%d x %d" % (self.num_neur, self.num_neur),
                                    category=WarningDG)

    def _check_mean_size(self, _input):
        """Checks if input mean has the required shape."""
        if _input.shape[-1] != self.num_neur:
            warnings.warn("Shape mismatch. Last dimension of input mean should be of size %d" % self.num_neur,
                          WarningDG)
            raise NotImplementedError
