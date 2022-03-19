"""
Creates dataset at data_input/dataset_hawkes/NAME.json
"""

import numpy as np

from data_input.json.parameter_loader import fetch_param_json_loader_simulation
from src.hawkes.hawkes_process import Hawkes_process
from src.utilities.pipeline_estimation import save_simulation_file, simulation_creation

(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
 parameter_functions, true_breakpoints, KERNEL, times_estimation,
 half_width_kernel) = fetch_param_json_loader_simulation(True)

hawksy = Hawkes_process(parameter_functions)

# simulation data
tt = np.linspace(0, T_MAX, NB_POINTS_TT)
data_simulated = simulation_creation(tt, hawksy, NB_SIMUL, id_hp=id_hp)
save_simulation_file(data_simulated, ["dataset_hawkes", STR_CONFIG], compress=False)

# print average of jumps over line
_, time_real = hawksy.simulation_Hawkes_exact_with_burn_in(tt=tt, plotFlag=False, silent=True)
print("STATISTICS: Average nb of jumps per unit of time: ", len(time_real[0]) / T_MAX)
