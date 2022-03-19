# normal libraries

# priv_libraries
import numpy as np
import pandas as pd

from corai_estimator import Estimator


# other files


# section ######################################################################
#  #############################################################################
# class


class Estim_hawkes(Estimator):
    CORE_COL = {'parameter', 'n', 'm', 'time estimation',
                'weight function', 'T_max', 'time_burn_in',
                'true value', 'value'}
    # m is the line index
    # n is the column index
    # M_m,n is the classical index notation.
    # This notation makes sense because (m,n) is contiguous in the last dimension, and we save the matrix row major.
    #

    LIST_CORE_COL = ['parameter', 'n', 'm', 'time estimation',
                     'weight function', 'T_max', 'time_burn_in',
                     'true value', 'value']

    # : make sure they correspond with the core col above.
    # We have both bc the ordering is not preserved when converting from set to list,
    # and we would like the columns to be ordered in a specific way.

    def __init__(self, df=None, *args, **kwargs):
        super().__init__(df=df, *args, **kwargs)
        self.df = self.df[self.LIST_CORE_COL]

    def data_in_append_creation(self, alpha, beta, nu, alpha_true, beta_true, nu_true,
                                dim_param, T_max, list_kernel, estimations_times, simul_burnin_time,
                                nb_simul):
        # the correct way to save data is to flatten the parameters
        # and put the parameters together.
        # Then, following such pattern, we create the rest of the parameters.

        # the way for flattening is (x,y,z) you get (z * y * x). Like matrix (n,m) -> (m*n).

        alpha_flat = alpha.flatten()  # if alpha is in format  (dim, dim, len_estim_times, nb_simul)
        beta_flat = beta.flatten()
        nu_flat = nu.flatten()
        estimation_total = np.concatenate([alpha_flat, beta_flat, nu_flat], axis=0)

        alpha_true_flat = alpha_true.flatten()  # if alpha is in format  (dim, dim, len_estim_times, nb_simul)
        beta_true_flat = beta_true.flatten()
        nu_true_flat = nu_true.flatten()

        len_estim_times = len(estimations_times)
        times_rounded = np.around(estimations_times, 8)

        # : the estimations_times is rounded because sometimes the registered number is not exactly correct.

        def time_structure(arr):
            # np.repeat allow to go left to right. no.tile from right to left
            matrix_time_estimation = np.tile(np.repeat(arr, nb_simul), dim_param * dim_param)
            # matrix_time_estimation = np.tile(np.repeat(times_rounded, dim_param * dim_param), nb_simul)
            vector_time_estimation = np.tile(np.repeat(arr, nb_simul), dim_param)
            # vector_time_estimation = np.tile(np.repeat(times_rounded, dim_param), nb_simul)
            estimations_times_total = np.concatenate([matrix_time_estimation,
                                                      matrix_time_estimation,
                                                      vector_time_estimation])
            return estimations_times_total

        estimations_times_total = time_structure(times_rounded)
        kernels_total = time_structure([kern.name for kern in list_kernel])

        true_value = np.concatenate([np.array(alpha_true_flat),
                                     np.array(beta_true_flat),
                                     np.array(nu_true_flat)],
                                    axis=0)

        # generate name_param_total
        name_param_total = np.concatenate([np.repeat("alpha", len(alpha_flat)),
                                           np.repeat("beta", len(beta_flat)),
                                           np.repeat("nu", len(nu_flat))], axis=0)

        # generate mm
        mm_alpha = np.repeat(np.arange(dim_param), nb_simul * len_estim_times * dim_param)
        mm_nu = np.repeat(0, dim_param * len_estim_times * nb_simul)
        # mm_alpha_index_pattern = np.tile(np.arange(dim_param).reshape((1, dim_param)).transpose(), (1, dim_param)).flatten()
        # mm_alpha_index_pattern = np.tile(mm_alpha_index_pattern, len_estim_times * nb_simul)
        # mm_nu_index_pattern = np.repeat(0, dim_param * len_estim_times * nb_simul)
        mm = np.concatenate([mm_alpha, mm_alpha, mm_nu], axis=0)

        # generate nn
        nn_alpha = np.tile(np.repeat(np.arange(dim_param), nb_simul * len_estim_times), dim_param)
        nn_nu = np.repeat(np.arange(dim_param), nb_simul * len_estim_times)
        # nn_alpha_index_pattern = np.tile(np.arange(dim_param), (dim_param, 1)).flatten()
        # nn_alpha_index_pattern = np.tile(nn_alpha_index_pattern, len_estim_times * nb_simul)
        # nn_nu_index_pattern = np.tile(np.arange(dim_param), len_estim_times * nb_simul)
        nn = np.concatenate([nn_alpha, nn_alpha, nn_nu], axis=0)

        size = dim_param + dim_param * dim_param * 2
        nb_lines_constant_param = size * len_estim_times * nb_simul
        estimator_dict_form = {
            "time estimation": estimations_times_total,
            "parameter": name_param_total,
            "m": mm,
            "n": nn,
            "weight function": kernels_total,
            "value": estimation_total,
            'T_max': [T_max] * nb_lines_constant_param,
            'time_burn_in': [simul_burnin_time] * nb_lines_constant_param,
            'true value': true_value
        }
        return estimator_dict_form

    def append_from_lists(self, alpha, beta, nu, alpha_true, beta_true, nu_true,
                          dim, T_max, list_kernel,
                          times_estimation, time_burn_in, nb_simul):
        """
        Semantics:
            Append information from the estimation to the estimator.
        Args:

        Returns:
            self
        """
        estimator_dict_form = self.data_in_append_creation(alpha, beta, nu, alpha_true, beta_true, nu_true,
                                                           dim, T_max, list_kernel, times_estimation,
                                                           time_burn_in, nb_simul)

        estimator_df_form = pd.DataFrame(estimator_dict_form)
        super().append(estimator_df_form)

        self.df = self.df[self.LIST_CORE_COL]
        return self

    def mean(self, separator):
        """

        Args:
            separator: is a list, of the estimators to gather together.

        Returns: the output format is list of lists with on each line [ans_N, ans_A, ans_B],
        and on every single additional dimension, the separator.

        """
        separators = ['parameter', 'm', 'n']
        M = self.df["m"].max() + 1
        ans_dict = {}

        # first separation with respect to some separator: time e.g.
        dictionary = self.df.groupby(separator)
        global_dict, keys = dictionary, dictionary.groups.keys()

        # separation wrt the essential parameters
        for key in keys:
            data = global_dict.get_group(key)
            dict_of_means = data.groupby(separators)['value'].mean()

            ans_N, ans_A, ans_B = [], [], []
            for i in range(M):
                ans_N.append(dict_of_means[('nu', 0, i)])
                for j in range(M):
                    if not j:  # if j == 0
                        ans_A.append([])
                        ans_B.append([])
                    # we append to this new small list the j's.
                    ans_A[i].append(dict_of_means[('alpha', i, j)])
                    ans_B[i].append(dict_of_means[('beta', i, j)])
            # i get triple list like usually.
            ans_dict[key] = [ans_N, ans_A, ans_B]
        return ans_dict

    def add_SPE_APE_col(self):
        # todo test true_value is not nan
        # squared percentage error
        # absolute percentage error

        def error_compute_spe(row):
            # if row["parameter"] == "beta":
            #     return 0.  # no error associated to beta.
            # if row["parameter"] == "alpha":
            #     return 0.  # no error associated to alpha.
            # if row["parameter"] == "nu":
            #     return 0.  # no error associated to nu.

            # type of error:

            # L2 rescale
            true = row["true value"]
            err = (true - row["value"]) / true
            return err * err * 100

        def error_compute_ape(row):
            if row["parameter"] == "beta":
                return 0.  # no error associated to beta.
            if row["parameter"] == "alpha":
                return 0.  # no error associated to alpha.
            # if row["parameter"] == "nu":
            #     return 0.  # no error associated to nu.

            # type of error:

            # L1 rescaled
            true = row["true value"]
            return abs(true - row["value"]) / true * 100

        self.df["SPE"] = self.df.apply(error_compute_spe, axis=1)

        self.df["APE"] = self.df.apply(error_compute_ape, axis=1)
