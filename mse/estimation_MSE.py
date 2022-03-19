"""

"""
import sys

import numpy as np

from data_input.json.parameter_loader import fetch_param_json_loader_simulation, fetch_param_json_loader_itideep, \
    t_max_parameters
from root_dir import linker_path_to_result_file
from src.estim_hawkes.estim_hawkes import Estim_hawkes
from src.utilities.fct_itideep import creator_kernels_adaptive
from src.utilities.pipeline_estimation import load_simulation_file, complete_pipe_estim_hawkes

######## data from command line
NB_LINE = int(sys.argv[1]) - 1  # bc we run  nb_simul + 1. Should go from 1 to ...
# nb_line = 0 # debug case scenario
NB_T_MAX = int(sys.argv[2]) - 1  # from 1 to 10.
NB_TH_OF_CURRENT_ESTIMATION = int(sys.argv[3])  # any int > 0. Represents the refinement of the ITiDeEP.
# 1 is the first naive estimation.
#########


# pass the str config to the fetch function.
STR_CONFIG = "MSE"
(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, _, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, _, parameters, t0, time_batch,
 fct_parameters, true_breakpoints, _, _, _) = fetch_param_json_loader_simulation(False, STR_CONFIG)
(L, R, h, l, CONSIDERED_PARAM, ALL_KERNELS_DRAWN,
 TYPE_ANALYSIS, NUMBER_OF_BREAKPOINTS, MODEL,
 MIN_SIZE, WIDTH) = fetch_param_json_loader_itideep(flagprint=True, str_config=STR_CONFIG)

######################### we redefine all parameters that are createdby fetch function but depends on T_max.
T_MAX = np.linspace(6000, 33000, 10)[NB_T_MAX]
(KERNEL, half_width_kernel, id_hp, times_estimation) = t_max_parameters(DIM, KERNEL_DIVIDER, NB_DIFF_TIME_ESTIM,
                                                                        NB_POINTS_TT, SEED,
                                                                        STYL, T_MAX, UNDERLYING_FUNCTION_NUMBER)
#########################

assert NB_LINE < NB_SIMUL, "The line number given does not match with how many data were simulated!"

# load data
data_simulated = load_simulation_file(["dataset_hawkes", "MSE", STR_CONFIG + str(NB_T_MAX)], compress=False)
data_simulated["time series"] = [data_simulated["time series"][NB_LINE]]  # slice it for the particular task nb

if NB_TH_OF_CURRENT_ESTIMATION > 1:  # case where we use the previous estimation and apply ITiDeEP
    # fetch data and fetch first estimation.
    path_result_estim1 = linker_path_to_result_file(
        ["MSE", f"{STR_CONFIG}_res_" + str(NB_TH_OF_CURRENT_ESTIMATION - 1), f"data_together_{NB_T_MAX}",
         f"results_together.csv"])
    previous_estimation_hp = Estim_hawkes.from_csv(path_result_estim1)
    #### iterative ITiDeEP: second estimation
    _, list_of_kernels_for_itideep = creator_kernels_adaptive(previous_estimation_hp, times_estimation,
                                                              CONSIDERED_PARAM,
                                                              [half_width_kernel] * len(times_estimation), L, R, h,
                                                              l, tol=0.05, silent=True)
else:  # no previous estimation so we use the basic naive kernel.
    list_of_kernels_for_itideep = KERNEL  # a single kernel and is converted into a list inside the pipe below

estim_hp = complete_pipe_estim_hawkes(data_simulated, times_estimation, list_of_kernels_for_itideep, silent=True)

# storing on the disk
path_result_directory = linker_path_to_result_file(
    ["MSE", f"{STR_CONFIG}_res_" + str(NB_TH_OF_CURRENT_ESTIMATION), f"data_{NB_T_MAX}",
     f"result_{NB_LINE + 1}.csv"])
estim_hp.to_csv(path_result_directory)
