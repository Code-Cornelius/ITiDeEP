from data_input.json.parameter_loader import fetch_param_json_loader_simulation, fetch_param_json_loader_itideep
from root_dir import linker_path_to_result_file
from src.estim_hawkes.estim_hawkes import Estim_hawkes
from src.utilities.fct_itideep import creator_kernels_adaptive

from src.utilities.pipeline_estimation import load_simulation_file, complete_pipe_estim_hawkes

# PARAMETERS
(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
 fct_parameters, true_breakpoints, KERNEL, times_estimation,
 half_width_kernel) = fetch_param_json_loader_simulation(True)
(L, R, h, l, CONSIDERED_PARAM, ALL_KERNELS_DRAWN,
 TYPE_ANALYSIS, NUMBER_OF_BREAKPOINTS, MODEL,
 MIN_SIZE, WIDTH) = fetch_param_json_loader_itideep(flagprint=True)

# fetch data and fetch first estimation.
data_simulated = load_simulation_file(["dataset_hawkes", STR_CONFIG], compress=False)
path_result_estim1 = linker_path_to_result_file(["result_hawkes_hitdep1", STR_CONFIG, "results_together.csv"])

first_estimation_hp = Estim_hawkes.from_csv(path_result_estim1)

#### iterative HITDEP: second estimation
_, list_of_kernels_for_hitdep = creator_kernels_adaptive(first_estimation_hp, times_estimation, CONSIDERED_PARAM,
                                                         [half_width_kernel] * len(times_estimation), L, R, h,
                                                         l, tol=0.1, silent=True)
print(list_of_kernels_for_hitdep)

second_estimation_hp = complete_pipe_estim_hawkes(data_simulated, times_estimation,
                                                  list_of_kernels_for_hitdep, silent=False)

# storing on the disk.
path_result = linker_path_to_result_file(["result_hawkes_hitdep2", STR_CONFIG, "results_together.csv"])
second_estimation_hp.to_csv(path_result)
