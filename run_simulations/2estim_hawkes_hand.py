from data_input.json.parameter_loader import fetch_param_json_loader_simulation
from root_dir import linker_path_to_result_file
from src.utilities.pipeline_estimation import load_simulation_file, complete_pipe_estim_hawkes

(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX,
 KERNEL_DIVIDER, NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp,
 parameters, t0, time_batch, fct_parameters, true_breakpoints, KERNEL,
 times_estimation, half_width_kernel) = fetch_param_json_loader_simulation(True)

# load data
data_simulated = load_simulation_file(["dataset_hawkes", STR_CONFIG], compress=False)

# estimation
estim_hp = complete_pipe_estim_hawkes(data_simulated, times_estimation, KERNEL, silent=False)

# storing on the disk.
path_result = linker_path_to_result_file(["result_hawkes_test", STR_CONFIG, "results_together.csv"])
estim_hp.to_csv(path_result)
