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


def find_root_bisection(*eqn_input, eqn=function, maxiters=1000, tol=1e-8):
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

    print('f0, f1', f0, f1)

    if np.abs(f0) < 1e-8:
        warnings.warn("Warning: f0 is already close to 0. Returning initial value.", WarningDGOpt)
        return λ0

    if np.abs(f1) < 1e-8:
        warnings.warn("Warning: f1 is already close to 0. Returning initial value.", WarningDGOpt)
        return λ1

    if f0 * f1 > 1e-8:
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

        print('λ, f', λ, f)

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

        mean[mean == 0.] += 1e-4
        mean[mean == 1.] -= 1e-4

        gauss_mean = norm.ppf(mean)
        return gauss_mean

    @property
    def data_covariance(self):
        """Computes covariance between spike trains from different neurons"""
        data = self.data

        data_norm = (data - data.mean(0)).reshape(self.timebins, -1)
        tot_covar = data_norm.T.dot(data_norm).reshape(self.trials, self.num_neur, self.trials, self.num_neur)
        inds = range(self.trials)
        tot_covar = tot_covar[inds, :, inds, :].mean(0) / self.timebins
        return tot_covar

    @property
    def data_tot_covariance(self):
        data = self.data

        # shuffle = np.random.choice(range(self.trials), size=self.trials, replace=False)
        data_norm = (data - data.mean(1)).reshape(-1, self.num_neur)
        # data_norm_re = (data[:, shuffle, :] - data.mean(1)).reshape(-1, self.num_neur)
        tot_covar = data_norm.T.dot(data_norm) / (self.timebins * self.trials)
        # sig_covar = data_norm.T.dot(data_norm_re) / (self.timebins * self.trials)

        return tot_covar# - sig_covar

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
            data_covar = self.data_covariance
        else:
            data_covar = self.data_tot_covariance

        ρ_list = []

        # Find pairwise correlation between each unique pair of neurons
        for i, j in zip(*self.tril_inds):
            print(i, j)
            if np.abs(data_covar[i][j]) <= 1e-4:
                print('Data covariance is zero. Setting corresponding Gaussian dist. covariance to 0.')
                ρ_list.append(0.)
            else:
                x = find_root_bisection([data_mean[i], data_mean[j]],
                                        [gauss_mean[..., i], gauss_mean[..., j]],
                                        data_covar[i][j],
                                        **kwargs)
                print(x)
                ρ_list.append(x)

        # Construct symmetric matrix from computed correlation values
        gauss_corr = np.eye(self.num_neur)
        gauss_corr[self.tril_inds] = np.array(ρ_list)
        gauss_corr[(self.tril_inds[1], self.tril_inds[0])] = np.array(ρ_list)

        if set_attr:
            setattr(self, 'gauss_corr', np.array(gauss_corr))
        return gauss_corr


# class DGGeneralOptimise(object):
#
#     def __init__(self, num_neur, timebins, trials, data):
#         self.num_neur = num_neur
#         self.timebins = timebins
#         self.trials = trials
#         self.tril_inds = np.tril_indices(self.num_neur, -1)
#         self.data = data
#
#     def get_data_moments(self, data=None, set_attr=True):
#         """Assumes data has shape timebins x trials x neurons"""
#         if data is None:
#             data = self.data
#         mean = data.mean(1).mean(0)
#         var = data.var(1)
#
#         mean[mean == 0.] += 1e-6
#         mean[mean == 1.] -= 1e-6
#         var += 1e-6
#
#         if set_attr:
#             setattr(self, 'data_mean', mean)
#             setattr(self, 'data_var', var)
#         return mean
#
#     def get_data_covariance(self, data=None, set_attr=True):
#         if data is None:
#             data = self.data
#         data_re = data.reshape(-1, self.num_neur)
#         tot_covar = (data_re - data_re.mean(0)).T.dot(data_re - data_re.mean(0)) / (self.trials * self.timebins)
#
#         shuffle = np.random.choice(range(self.trials), size=self.trials, replace=False)
#         data_shuff = data[:, shuffle, :].reshape(-1, self.num_neur)
#         sig_covar = (data_re - data_re.mean(0)).T.dot(data_shuff - data_shuff.mean(0)) / (self.trials * self.timebins)
#
#         ns_covar = tot_covar - sig_covar
#
#         if set_attr:
#             setattr(self, 'tot_covar', tot_covar)
#             setattr(self, 'sig_covar', sig_covar)
#             setattr(self, 'ns_covar', ns_covar)
#         return tot_covar, sig_covar, ns_covar
#
#     def get_gauss_mean(self, var, data_mean=None, set_attr=True):
#         """Assumes data has shape timebins x trials x neurons"""
#         if data_mean is None:
#             data_mean = self.data_mean
#         gauss_mean = np.sqrt(2 * (var + 1)) * erfinv(2 * data_mean - 1)
#         if set_attr:
#             setattr(self, 'gauss_mean', gauss_mean)
#         return gauss_mean
#
#     def get_bivargauss_cdf(self, vals, covar):
#         cdf = mnorm.cdf(vals, mean=[0., 0.], cov=covar)
#         return cdf
#
#     def gen_cov_mat(self, gauss_var=None, sig_corr=None, noise_corr=None):
#         if sig_corr is None and noise_corr is None:
#             return np.array([gauss_var] * 4).reshape(2, 2) + np.eye(2)
#
#         elif (sig_corr is not None) and (noise_corr is None) and (len(gauss_var) == 2):
#             sig_cov = np.array([[1., sig_corr], [sig_corr, 1.]])
#             return sig_cov * np.outer(np.sqrt(gauss_var), np.sqrt(gauss_var)) + np.eye(2)
#
#         elif (noise_corr is not None) and (sig_corr is None) and (gauss_var is None):
#             return np.array([[1., noise_corr], [noise_corr, 1.]])
#
#         elif (sig_corr is not None) and (noise_corr is not None) and (len(gauss_var) == 2):
#             sig_cov = np.array([[1., sig_corr], [sig_corr, 1.]])
#             noise_cov = np.array([[0., noise_corr], [noise_corr, 0.]])
#             return sig_cov * np.outer(np.sqrt(gauss_var), np.sqrt(gauss_var)) + np.eye(2) + noise_cov
#         else:
#             print("Cannot generate covariance matrix from given inputs.")
#             raise NotImplementedError
#
#     def pairwise_cov_eqn(self, data_means, data_covar, gauss_means, gauss_cov):
#         return - data_covar + self.get_bivargauss_cdf(gauss_means, covar=gauss_cov) - np.prod(data_means)
#
#     def eqn_var_per_neur(self, neur_ind, gauss_var):
#         data_mean = self.data_mean[neur_ind]
#         gauss_mean = self.get_gauss_mean(gauss_var, data_mean, set_attr=False)
#         data_covar = np.diag(self.sig_covar)[neur_ind]
#         cov_mat = self.gen_cov_mat(gauss_var=gauss_var)
#
#         return self.pairwise_cov_eqn(data_means=[data_mean, data_mean],
#                                      data_covar=data_covar,
#                                      gauss_means=[gauss_mean, gauss_mean],
#                                      gauss_cov=cov_mat)
#
#     def eqn_sig_cov_btwn_neurs(self, neur_ind_1, neur_ind_2, gauss_sig_corr):
#         data_means = [self.data_mean[neur_ind_1], self.data_mean[neur_ind_2]]
#         gauss_means = [self.get_gauss_mean(self.gauss_var[neur_ind_1], data_means[0], set_attr=False),
#                        self.get_gauss_mean(self.gauss_var[neur_ind_2], data_means[1], set_attr=False)]
#
#         gauss_var = np.array([self.gauss_var[neur_ind_1], self.gauss_var[neur_ind_2]])
#
#         data_covar = self.sig_covar[neur_ind_1][neur_ind_2]
#         sig_cov_mat = self.gen_cov_mat(gauss_var=gauss_var, sig_corr=gauss_sig_corr)
#
#         return self.pairwise_cov_eqn(data_means=data_means, data_covar=data_covar,
#                                      gauss_means=gauss_means, gauss_cov=sig_cov_mat)
#
#     def eqn_noise_cov_btwn_neurs(self, neur_ind_1, neur_ind_2, gauss_noise_corr):
#         data_means = [self.data_mean[neur_ind_1], self.data_mean[neur_ind_2]]
#         gauss_means = [self.get_gauss_mean(self.gauss_var[neur_ind_1], data_means[0], set_attr=False),
#                        self.get_gauss_mean(self.gauss_var[neur_ind_2], data_means[1], set_attr=False)]
#
#         gauss_var = [self.gauss_var[neur_ind_1], self.gauss_var[neur_ind_2]]
#         gauss_sig_corr = self.gauss_sig_corr[neur_ind_1][neur_ind_2]
#
#         data_covar = self.tot_covar[neur_ind_1][neur_ind_2]
#         tot_cov_mat = self.gen_cov_mat(gauss_var=gauss_var, sig_corr=gauss_sig_corr,
#                                        noise_corr=gauss_noise_corr)
#         return self.pairwise_cov_eqn(data_means=data_means, data_covar=data_covar,
#                                      gauss_means=gauss_means, gauss_cov=tot_cov_mat)
#
#     def find_root_bisection(self, eqn_to_solve, eqn_inputs, lb=-.999, ub=.999, maxiters=1000):
#         λ0 = lb
#         λ1 = ub
#
#         f0 = eqn_to_solve(*eqn_inputs, λ0)
#         f1 = eqn_to_solve(*eqn_inputs, λ1)
#         print('f0, f1', f0, f1)
#
#         if np.abs(f0) < 1e-4:
#             print("Warning: f0 is already close to 0. Returning initial value.")
#             return λ0
#
#         if np.abs(f1) < 1e-4:
#             print("Warning: f1 is already close to 0. Returning initial value.")
#             return λ1
#
#         if f0 * f1 > 1e-8:
#             print('Warning: Both initial covariance values lie on same side of zero crossing. '
#                   'Setting value to 0.')
#             λ = 0.
#             return λ
#
#         f = np.inf
#         it = 0
#         print("Neuron Indices:", eqn_inputs)
#         while np.abs(f) > 1e-8 and it < maxiters:
#
#             λ = (λ0 + λ1) / 2
#             f = eqn_to_solve(*eqn_inputs, λ)
#
#             print('it, λ, f', it, λ, f)
#
#             if f > 0:
#                 λ1 = λ
#             elif f < 0:
#                 λ0 = λ
#             it += 1
#         clear_output(wait=True)
#         return λ
#
#     def comp_gauss_variance_mean(self, maxiters=1000, ub=1, set_attr=True):
#         self.get_data_moments(set_attr=set_attr)
#
#         variance = np.zeros(self.num_neur)
#         for i in range(self.num_neur):
#             variance[i] = self.find_root_bisection(self.eqn_var_per_neur, eqn_inputs=[i],
#                                                    lb=0., ub=ub,  maxiters=maxiters)
#
#         if set_attr is True:
#             setattr(self, "gauss_var", variance)
#         mean = self.get_gauss_mean(variance, data_mean=self.data_mean, set_attr=set_attr)
#         return variance, mean
#
#     def comp_gauss_sig_cov(self, maxiters=1000, set_attr=True):
#         if (hasattr(self, "gauss_var")) is False or (hasattr(self, "gauss_mean") is False):
#             self.comp_gauss_variance_mean(maxiters=maxiters, set_attr=True)
#         sig_covariance = np.eye(self.num_neur)
#         for i in range(self.num_neur):
#             for j in range(i+1, self.num_neur):
#                 sig_covariance[i][j] = self.find_root_bisection(self.eqn_sig_cov_btwn_neurs, [i, j],
#                                                                 maxiters=maxiters)
#                 sig_covariance[j][i] = sig_covariance[i][j]
#         sig_covariance = np.outer(np.sqrt(self.gauss_var), np.sqrt(self.gauss_var)) * sig_covariance \
#                          + np.eye(self.num_neur)
#         if set_attr is True:
#             setattr(self, "gauss_sig_corr", sig_covariance)
#         return sig_covariance
#
#     def comp_gauss_ns_cov(self, maxiters=1000, set_attr=True):
#         if hasattr(self, "gauss_sig_covar") is False:
#             self.comp_gauss_sig_cov(maxiters=maxiters, set_attr=True)
#
#         ns_covariance = np.eye(self.num_neur)
#
#         for i in range(self.num_neur):
#             for j in range(i+1, self.num_neur):
#                 ns_covariance[i][j] = self.find_root_bisection(self.eqn_noise_cov_btwn_neurs, [i, j],
#                                                                maxiters=maxiters)
#                 ns_covariance[j][i] = ns_covariance[i][j]
#         if set_attr is True:
#             setattr(self, "gauss_ns_covariance", ns_covariance)
#         return ns_covariance
