# normal libraries
import warnings

import numpy as np
from tqdm import tqdm

# priv_libraries
from corai_error import Error_convergence
from corai_util.tools.src.function_iterable import is_iterable
from corai_util.tools.src.function_writer import list_of_dicts_to_json, json2python
# other files
from root_dir import linker_path_to_data_file
from run_simulations.setup_parameters import setup_parameters
from src.estim_hawkes.estim_hawkes import Estim_hawkes
from src.hawkes.kernel import Kernel, fct_plain
from src.utilities.fct_type_process import function_parameters_hawkes
from src.utilities.mle_hawkes import adaptor_mle

kernel_plain = Kernel(fct_kernel=fct_plain, name="flat")


def simulation_creation(tt, hp, nb_simul, id_hp=None):
    to_save = [0] * nb_simul
    for i in range(nb_simul):
        _, time_real = hp.simulation_Hawkes_exact_with_burn_in(tt=tt, plotFlag=False, silent=True)
        to_save[i] = time_real
    # to_save looks like: [   list_per_dim_of_jumps_time  ]
    data_simulated = {"time series": to_save,
                      "information about the process": id_hp}
    return data_simulated


def save_simulation_file(data_simulated, path_save, compress=False):
    # path without json. List of str.
    path = linker_path_to_data_file(path_save)
    list_of_dicts_to_json(data_simulated, file_name=path + ".json", compress=compress)
    return


def load_simulation_file(path_save, compress=False):
    # path without json. List of str.
    path = linker_path_to_data_file(path_save)
    complete_data = json2python(path + ".json", compress)
    return complete_data


def estim_from_simulations(data_simulated, times_estimation, list_kernel, silent=False, first_guess=None):
    """

    Args:
        data_simulated:
        times_estimation:
        list_kernel (kernel or iter<kernel>):
        silent:
        first_guess:

    Returns: np.array of dimensions: dim,dim,nb_time_estim,nb_simul

    """
    # read file and for each data, do estimation.
    # If fails, put NaN as estimation and add to some counter the number of fail.

    # times_estimation is an iterable of the times when to perform an estimation.
    len_estim_times = len(times_estimation)

    # data_simulated["time series"] is the data.
    timeseries = data_simulated["time series"]  # should be 3d level nested list.
    # 1st: the list of simulation, 2nd: the dimension of realisations, 3d: the jumps.
    nb_simul = len(timeseries)

    dim_param = data_simulated["information about the process"]['dim']
    T_max = data_simulated["information about the process"]["T max"]

    alpha_hat_total = np.zeros((dim_param, dim_param, len_estim_times, nb_simul))
    beta_hat_total = np.zeros((dim_param, dim_param, len_estim_times, nb_simul))
    nu_hat_total = np.zeros((dim_param, len_estim_times, nb_simul))

    flag_simul_tqdm = (len(timeseries) > 1)  # the cdt indicate that if only one simul, use tqdm for lower loop.
    # for each time estim estim, then repeat per simul.
    for j_simul in tqdm(range(len(timeseries)), disable=silent | (not flag_simul_tqdm)):
        time_real = timeseries[j_simul]
        time_real = [np.array(time_real[i], dtype=np.float32) for i in range(len(time_real))]
        # convert data into a low memory footprint data. The induced error is negligible for our computations.

        for i_time in tqdm(range(len_estim_times), disable=(silent) and (flag_simul_tqdm)):
            time_estimation = times_estimation[i_time]
            kern = list_kernel[i_time]
            w = kern(T_t=time_real, eval_point=time_estimation, T_max=T_max)

            try:  # try encapsulating the error of convergence

                # branching condition if we give a first guess.
                if first_guess is not None:  # first guess is a vector of the true values.
                    # we do the slicing now since we will need the whole vector for doing the estimation at every time
                    nu, alpha, beta = first_guess
                    nu_sliced, alpha_sliced, beta_sliced = (nu[:, i_time, 0],
                                                            alpha[:, :, i_time, 0], beta[:, :, i_time, 0])
                    # we fix the value 0 because the last dimension is the one of parallel simulations,
                    # where the true value is identical for all of them.
                    current_first_guess = (nu_sliced, alpha_sliced, beta_sliced)

                    if not silent:
                        print(
                            "First guess for estimation at time {}: \n{}".format(time_estimation, current_first_guess))
                else:
                    current_first_guess = first_guess  # current_first_guess = None

                # estimation
                alpha_hat, beta_hat, nu_hat = adaptor_mle(time_real, T_max, w=w, silent=silent,
                                                          first_guess=current_first_guess)

            except Error_convergence as err:
                warnings.warn(err.message)
                alpha_hat, beta_hat, nu_hat = (np.full((dim_param, dim_param), np.nan),
                                               np.full((dim_param, dim_param), np.nan),
                                               np.full(dim_param, np.nan))  # NaN because value not found.

            # save the data
            alpha_hat_total[:, :, i_time, j_simul] = alpha_hat
            beta_hat_total[:, :, i_time, j_simul] = beta_hat
            nu_hat_total[:, i_time, j_simul] = nu_hat

    return alpha_hat_total, beta_hat_total, nu_hat_total


def true_values_from_function(true_functions, data_simulated, times_estimation, time_burn_in):
    # return numpy array.
    nu_true, alpha_true, beta_true = true_functions
    len_estim_times = len(times_estimation)
    timeseries = data_simulated["time series"]  # should be 3d level nested list.
    # 1st: the list of simulation, 2nd: the dimension of realisations, 3d: the jumps.
    nb_simul = len(timeseries)
    dim_param = data_simulated["information about the process"]['dim']
    T_max = data_simulated["information about the process"]["T max"]

    # generate true value
    alpha_hat_total_true = np.zeros((dim_param, dim_param, len_estim_times, nb_simul))
    beta_hat_total_true = np.zeros((dim_param, dim_param, len_estim_times, nb_simul))
    nu_hat_total_true = np.zeros((dim_param, len_estim_times, nb_simul))
    for l in range(nb_simul):
        for i in range(dim_param):
            nu_hat_total_true[i, :, l] = nu_true[i](times_estimation, T_max, time_burn_in)
            for j in range(dim_param):
                alpha_hat_total_true[i, j, :, l] = alpha_true[i][j](times_estimation, T_max, time_burn_in)
                beta_hat_total_true[i, j, :, l] = beta_true[i][j](times_estimation, T_max, time_burn_in)

    return alpha_hat_total_true, beta_hat_total_true, nu_hat_total_true


def estimation2estimatorhp(alpha, beta, nu, estimator_hp, data_simulated, times_estimation, list_kernel,
                           alpha_true=None, beta_true=None, nu_true=None):
    if alpha_true is None:
        alpha_true = [lambda x, T_max, time_burn_in: np.nan]
    if beta_true is None:
        beta_true = [lambda x, T_max, time_burn_in: np.nan]
    if nu_true is None:
        nu_true = [lambda x, T_max, time_burn_in: np.nan]

    timeseries = data_simulated["time series"]  # should be 3d level nested list.
    # 1st: the list of simulation, 2nd: the dimension of realisations, 3d: the jumps.
    nb_simul = len(timeseries)
    dim_param = data_simulated["information about the process"]['dim']
    time_burn_in = data_simulated["information about the process"]['time burn-in']
    T_max = data_simulated["information about the process"]['T max']

    # for the data I have, convert it into a df, then append it to the estimator hp.
    estimator_hp.append_from_lists(alpha, beta, nu, alpha_true, beta_true, nu_true, dim_param, T_max, list_kernel,
                                   times_estimation, time_burn_in, nb_simul)


def complete_pipe_estim_hawkes(data_simulated, times_estimation, kernel_choice=kernel_plain, silent=False):
    # loads the function directly from the data_simulated, where the profile of the HP is stored.
    # kernel_choice: iter<kernel> or kernel.

    # give back the total estimation over times_estimation, for all kernel, for all data_simulated.

    if is_iterable(kernel_choice):
        assert len(kernel_choice) == len(times_estimation), \
            "The number of kernel must match the number of estimation points."
        kernels = kernel_choice
    else:  # kernel is not iterable, hence it is a unique kernel.
        kernels = [kernel_choice] * len(times_estimation)

    time_burn_in = data_simulated["information about the process"]["time burn-in"]
    SEED = data_simulated["information about the process"]["seed"]
    STYL = data_simulated["information about the process"]["styl"]
    DIM = data_simulated["information about the process"]["dim"]
    UNDERLYING_FUNCTION_NUMBER = data_simulated["information about the process"]["type evol"]
    parameters, t0, time_batch = setup_parameters(SEED, DIM, STYL)
    fct_parameters, true_breakpoints = function_parameters_hawkes(UNDERLYING_FUNCTION_NUMBER, parameters)

    # true values:
    (alpha_true, beta_true, nu_true) = true_values_from_function(fct_parameters, data_simulated,
                                                                 times_estimation, time_burn_in)
    (alpha_true, beta_true, nu_true) = (np.round(alpha_true, 4),
                                        np.round(beta_true, 4), np.round(nu_true, 4))

    # get the estimations:
    alpha, beta, nu = estim_from_simulations(data_simulated, times_estimation, list_kernel=kernels, silent=silent,
                                             first_guess=(nu_true, alpha_true, beta_true))
    alpha, beta, nu = np.round(alpha, 4), np.round(beta, 4), np.round(nu, 4)

    estim_hp = Estim_hawkes()  # saving it
    estimation2estimatorhp(alpha, beta, nu, estim_hp, data_simulated, times_estimation, kernels,
                           alpha_true, beta_true, nu_true)
    return estim_hp
