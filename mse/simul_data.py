"""
Creates dataset at data_input/dataset_hawkes/NAME.json
"""

import numpy as np
from tqdm import tqdm

from data_input.json.parameter_loader import fetch_param_json_loader_simulation
from src.hawkes.hawkes_process import Hawkes_process
from src.utilities.pipeline_estimation import save_simulation_file, simulation_creation

STR_CONFIG = "MSE"

(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, _, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
 parameter_functions, true_breakpoints,
 KERNEL, times_estimation, half_width_kernel) = fetch_param_json_loader_simulation(True, STR_CONFIG)

hawksy = Hawkes_process(parameter_functions)

######################### we redefine all parameters that are createdby fetch function but depends on T_max.
LIST_T_MAX = np.linspace(6000, 33000, 10)

for i, T_MAX in enumerate(tqdm(LIST_T_MAX)):
    id_hp['T max'] = T_MAX
    # simulation data
    tt = np.linspace(0, T_MAX, NB_POINTS_TT)
    data_simulated = simulation_creation(tt, hawksy, NB_SIMUL, id_hp=id_hp)

    # if we want to get the lengths:
    # lengths = [len(data_simulated["time series"][i][0]) for i in range(len(data_simulated["time series"]))]
    # print("STATISTICS: Average nb of jumps: ", np.mean(lengths))

    save_simulation_file(data_simulated, ["dataset_hawkes", "MSE", STR_CONFIG + str(i)], compress=False)
