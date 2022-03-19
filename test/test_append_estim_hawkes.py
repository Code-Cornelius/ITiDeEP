"""
test that estim hawkes is correct.
"""

import numpy as np

from run_simulations.setup_parameters import setup_parameters
from src.estim_hawkes.estim_hawkes import Estim_hawkes
from src.hawkes.hawkes_process import Hawkes_process
from src.utilities.fct_type_process import function_parameters_hawkes
from src.utilities.pipeline_estimation import simulation_creation, estim_from_simulations, kernel_plain, \
    true_values_from_function

nb_simul = 2
seed = 42
dim_param = 1
styl = 2
switch = 0  # or 21, or 3
nb_of_points_estimation = 3
kernel = kernel_plain

np.random.seed(seed)
T_max = 20
nb_points_tt = int(1E5)
id_hp = {'seed': seed, 'styl': styl, 'dim': dim_param, 'type evol': switch, 'time burn-in': Hawkes_process.TIME_BURN_IN,
         'T max': T_max, 'nb points tt': nb_points_tt}
tt = np.linspace(0, T_max, nb_points_tt)

parameters, t0, time_batch = setup_parameters(seed, dim_param, styl)
# time_batch corresponds to how much time required for 50 jumps
parameter_functions, true_breakpoints = function_parameters_hawkes(switch, parameters)
hawksy = Hawkes_process(parameter_functions)

data_simulated = simulation_creation(tt, hawksy, nb_simul, id_hp=id_hp)

times_estimation = np.linspace(0, T_max, nb_of_points_estimation)
nu, alpha, beta = estim_from_simulations(data_simulated, times_estimation, [kernel] * nb_of_points_estimation)
estim_hp = Estim_hawkes()
(alpha_true, beta_true, nu_true) = true_values_from_function(parameter_functions, data_simulated,
                                                             times_estimation, Hawkes_process.TIME_BURN_IN)

estimator_dict_form = estim_hp.data_in_append_creation(alpha, beta, nu, alpha_true, beta_true, nu_true,
                                                       dim_param, T_max, [kernel] * nb_of_points_estimation,
                                                       times_estimation,
                                                       Hawkes_process.TIME_BURN_IN, nb_simul)
test_time_estimation = estimator_dict_form["time estimation"]
test_parameters_name = estimator_dict_form["parameter"]
test_mm = estimator_dict_form["m"]
test_nn = estimator_dict_form["n"]

size = nb_simul * nb_of_points_estimation * (dim_param * dim_param * 2 + dim_param)
assert size == len(test_time_estimation) == len(test_parameters_name) == len(test_nn) == len(test_mm), "size is wrong."

assert list(test_time_estimation) == [0, 0,
                                      10, 10,
                                      20, 20,
                                      0, 0,
                                      10, 10,
                                      20, 20,
                                      0, 0,
                                      10, 10,
                                      20, 20], "time_estimation is wrong"

assert list(test_parameters_name) == ['alpha', 'alpha', 'alpha',
                                      'alpha', 'alpha', 'alpha',
                                      'beta', 'beta', 'beta',
                                      'beta', 'beta', 'beta',
                                      'nu', 'nu', 'nu',
                                      'nu', 'nu', 'nu'], "parameter names is wrong"
assert list(test_mm) == [0] * 2 * 3 * 3, "mm is wrong"

assert list(test_nn) == [0] * 2 * 3 * 3, "nn is wrong"

######################
######################
######################
######################
# second case scenario, where dim is 2.
nb_simul = 2
seed = 42
dim_param = 2
styl = 2
switch = 0  # or 21, or 3
nb_of_points_estimation = 3
kernel = kernel_plain

np.random.seed(seed)
T_max = 20
nb_points_tt = int(1E5)
id_hp = {'seed': seed, 'styl': styl, 'dim': dim_param, 'type evol': switch, 'time burn-in': Hawkes_process.TIME_BURN_IN,
         'T max': T_max, 'nb points tt': nb_points_tt}
tt = np.linspace(0, T_max, nb_points_tt)

parameters, t0, time_batch = setup_parameters(seed, dim_param, styl)
# time_batch corresponds to how much time required for 50 jumps
parameter_functions, true_breakpoints = function_parameters_hawkes(switch, parameters)
hawksy = Hawkes_process(parameter_functions)

data_simulated = simulation_creation(tt, hawksy, nb_simul, id_hp=id_hp)

times_estimation = np.linspace(0, T_max, nb_of_points_estimation)
alpha, beta, nu = estim_from_simulations(data_simulated, times_estimation, [kernel] * nb_of_points_estimation)
estim_hp = Estim_hawkes()
(alpha_true, beta_true, nu_true) = true_values_from_function(parameter_functions, data_simulated,
                                                             times_estimation, Hawkes_process.TIME_BURN_IN)

estimator_dict_form = estim_hp.data_in_append_creation(alpha, beta, nu, alpha_true, beta_true, nu_true,
                                                       dim_param, T_max, [kernel] * nb_of_points_estimation,
                                                       times_estimation,
                                                       Hawkes_process.TIME_BURN_IN, nb_simul)

test_time_estimation = estimator_dict_form["time estimation"]
test_parameters_name = estimator_dict_form["parameter"]
test_mm = estimator_dict_form["m"]
test_nn = estimator_dict_form["n"]

size = nb_simul * nb_of_points_estimation * (dim_param * dim_param * 2 + dim_param)
assert size == len(test_time_estimation) == len(test_parameters_name) == len(test_nn) == len(test_mm), "size s wrong."

assert list(test_time_estimation) == [0, 0, 10, 10, 20, 20,
                                      0, 0, 10, 10, 20, 20,
                                      0, 0, 10, 10, 20, 20,
                                      0, 0, 10, 10, 20, 20,

                                      0, 0, 10, 10, 20, 20,
                                      0, 0, 10, 10, 20, 20,
                                      0, 0, 10, 10, 20, 20,
                                      0, 0, 10, 10, 20, 20,

                                      0, 0, 10, 10, 20, 20,
                                      0, 0, 10, 10, 20, 20], "time_estimation is wrong"

assert list(test_parameters_name) == ['alpha', 'alpha', 'alpha', 'alpha',
                                      'alpha', 'alpha', 'alpha', 'alpha',
                                      'alpha', 'alpha', 'alpha', 'alpha',
                                      'alpha', 'alpha', 'alpha', 'alpha',
                                      'alpha', 'alpha', 'alpha', 'alpha',
                                      'alpha', 'alpha', 'alpha', 'alpha',
                                      'beta', 'beta', 'beta', 'beta',
                                      'beta', 'beta', 'beta', 'beta',
                                      'beta', 'beta', 'beta', 'beta',
                                      'beta', 'beta', 'beta', 'beta',
                                      'beta', 'beta', 'beta', 'beta',
                                      'beta', 'beta', 'beta', 'beta',
                                      'nu', 'nu', 'nu', 'nu', 'nu', 'nu',
                                      'nu', 'nu', 'nu', 'nu', 'nu', 'nu'], "parameter names is wrong"

assert list(test_mm) == [0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,

                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,

                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0], "mm is wrong"

assert list(test_nn) == [0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1,
                         0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1,

                         0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1,
                         0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1,

                         0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1], "nn is wrong"
