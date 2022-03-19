"""

"""

import sys

from data_input.json.parameter_loader import fetch_param_json_loader_simulation, fetch_param_json_loader_itideep
from root_dir import linker_path_to_result_file
from src.utilities.pipeline_estimation import load_simulation_file, complete_pipe_estim_hawkes

NB_LINE = int(sys.argv[1]) - 1  # bc we run  nb_simul + 1. Should go from 1 to ...
# nb_line = 0 # debug case scenario
STR_CONFIG = str(sys.argv[2])

# pass the str config to the fetch function.
(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
 fct_parameters, true_breakpoints,
 KERNEL, times_estimation, half_width_kernel) = fetch_param_json_loader_simulation(False, STR_CONFIG)

assert NB_LINE < NB_SIMUL, "The line number given does not match with how many data were simulated!"

# load data
data_simulated = load_simulation_file(["dataset_hawkes", STR_CONFIG], compress=False)
# slice it for the particular task nb
data_simulated["time series"] = [data_simulated["time series"][NB_LINE]]

estim_hp = complete_pipe_estim_hawkes(data_simulated, times_estimation, KERNEL, silent=True)

# storing on the disk.
path_result_directory = linker_path_to_result_file(
 ["euler_hawkes", f"{STR_CONFIG}_res_1", "data", f"result_{NB_LINE + 1}.csv"])
estim_hp.to_csv(path_result_directory)
