import numpy as np
from scipy.stats import norm, multivariate_normal as mnorm
from scipy.special import erfinv, erf
from IPython.display import clear_output

import warnings


class WarningDGOpt(UserWarning):
    pass


def get_bivargauss_cdf(vals, corr_coef):
    """
    Computes cdf of a bivariate Gaussian distribution with mean zero, variance 1 and input correlation.

    Inputs:
        :param vals: arguments for bivariate cdf (μi, μj).
        :param corr_coef: correlation coefficient of biavariate Gaussian (Λij).

    Returns:
        :return: Φ2([μi, μj], Λij)
    """
    cov = np.eye(2)
    cov[1, 0], cov[0, 1] = corr_coef, corr_coef
    cdf = mnorm.cdf(vals, mean=[0., 0.], cov=cov)
    return cdf


def function(data_means, gauss_means, data_covar, gauss_covar):
    """
    Computes the pairwise covariance eqn for root finding algorithm.

    Inputs:
        :param data_means: mean of binary spike train of 2 neurons (ri, rj).
        :param gauss_means: mean of bivariate Gaussian that calculated from data for the 2 neurons (μi, μj).
        :param data_covar: covariance between the spike trains of the 2 neurons (Σij).
        :param gauss_covar: covariance of the bivariate Gaussian distribution corresponding to the 2 neurons (Λij).

    Returns:
        :return: Φ2([μi, μi], Λij) - ri*rj - Σij
    """
    bivar_gauss_cdf = np.mean(get_bivargauss_cdf(vals=np.array(gauss_means).T,
                                                 corr_coef=gauss_covar))
    return bivar_gauss_cdf - np.prod(data_means) - data_covar


def find_root_bisection(*eqn_input, eqn=function, maxiters=1000, tol=1e-10):
    """
    Finds root of input equation using the bisection algorithm.

    Inputs:
        :param eqn_input: list containing inputs to \'eqn\' method.
        :param eqn: method implementing the equation for which we need the root.
        :param maxiters: max. number of iterations for bisection algorithm.
        :param tol: tolerance value for convergence of bisection algorithm.

    Returns:
        :return: root of \'eqn\'.
    """
    λ0 = -.99999
    λ1 = .99999

    f0 = eqn(*eqn_input, λ0)
    f1 = eqn(*eqn_input, λ1)

    # print('f0, f1', f0, f1)

    if np.abs(f0) < tol:
        warnings.warn("Warning: f0 is already close to 0. Returning initial value.", WarningDGOpt)
        return λ0

    if np.abs(f1) < tol:
        warnings.warn("Warning: f1 is already close to 0. Returning initial value.", WarningDGOpt)
        return λ1

    if f0 * f1 > tol:
        warnings.warn('Warning: Both initial covariance values lie on same side of zero crossing. '
                      'Setting value to 0.',
                      WarningDGOpt)
        λ = 0.
        return λ

    f = np.inf
    it = 0
    while np.abs(f) > tol and it < maxiters:
        λ = (λ0 + λ1) / 2
        f = eqn(*eqn_input, λ)

        # print('λ, f(λ)', λ, f)

        if f > 0:
            λ1 = λ
        elif f < 0:
            λ0 = λ
        it += 1
    clear_output(wait=True)
    return λ


class DGOptimise(object):
    """
        Finds the parameters of the multivariate Gaussian that best fit the given binary spike train.
        Inputs:
            :param data: binary spike count data of size timebins x repeats x neurons
    """

    def __init__(self, data):
        self.timebins, self.trials, self.num_neur = data.shape
        self.tril_inds = np.tril_indices(self.num_neur, -1)
        self.data = data

    @property
    def gauss_mean(self):
        """
        Computes mean of the multivariate Gaussian corresponding to the input binary spike train.
        """
        data = self.data

        mean = data.mean(1)
        self._check_mean(mean)  # Check if mean lies between 0 and 1

        # Need this to ensure inverse cdf calculation (norm.ppf()) does not break
        mean[mean == 0.] += 1e-4
        mean[mean == 1.] -= 1e-4

        gauss_mean = norm.ppf(mean)
        return gauss_mean

    @property
    def data_tvar_covariance(self):
        """Computes covariance between spike trains from different neurons, averaged across timebins and trials.
           Calculated for time-varying firing rate"""
        data = self.data

        data_norm = (data - data.mean(0)).reshape(self.timebins, -1)
        tot_covar = data_norm.T.dot(data_norm).reshape(self.trials, self.num_neur, self.trials, self.num_neur)
        inds = range(self.trials)
        tot_covar = tot_covar[inds, :, inds, :].mean(0) / self.timebins
        return tot_covar

    @property
    def data_tfix_covariance(self):
        """Computes covariance between spike trains from different neurons, averaged across repeats. Calculated for
           fixed firing rate."""
        data = self.data
        data_norm = (data - data.mean(1)).reshape(-1, self.num_neur)
        tot_covar = data_norm.T.dot(data_norm) / (self.timebins * self.trials)

        return tot_covar

    def get_gauss_correlation(self, set_attr=True, **kwargs):
        """
        Computes the correlation matrix of the multivariate Gaussian that best fits the input binary spike trains.
        Inputs:
            :param set_attr: set to True to make computed correlation matrix an attribute of the class.
            :param kwargs: arguments for bisection algorithm method (see help(find_root_bisection)).

        Returns:
            :return: computed correlation matrix of multivariate Gaussian distribution.
        """
        data_mean = self.data.mean(1).mean(0)
        gauss_mean = self.gauss_mean
        if self.timebins > 1:
            data_covar = self.data_tvar_covariance
        else:
            data_covar = self.data_tfix_covariance

        gauss_corr = np.eye(self.num_neur)

        # Find pairwise correlation between each unique pair of neurons
        for i, j in zip(*self.tril_inds):
            # print("Neuron pair:", i, j)
            if np.abs(data_covar[i][j]) <= 1e-10:
                print('Data covariance is zero. Setting corresponding Gaussian dist. covariance to 0.')
                gauss_corr[i][j], gauss_corr[j][i] = 0., 0.

            else:
                x = find_root_bisection([data_mean[i], data_mean[j]],
                                        [gauss_mean[..., i], gauss_mean[..., j]],
                                        data_covar[i][j],
                                        **kwargs)
                gauss_corr[i][j], gauss_corr[j][i] = x, x

        if set_attr:
            setattr(self, 'gauss_corr', np.array(gauss_corr))
        return gauss_corr

    def _check_mean(self, mean):
        """Checks if input mean values lie between 0 and 1."""
        if np.any(mean < 0) or np.any(mean > 1):
            print('Mean should have value between 0 and 1.')
            raise NotImplementedError
