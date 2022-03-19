"""

"""

import sys

# current project
from data_input.json.parameter_loader import fetch_param_json_loader_simulation
# priv_lib
from corai_util.tools.src.function_file import remove_files_from_dir, is_empty_file
from root_dir import linker_path_to_result_file
from src.estim_hawkes.estim_hawkes import Estim_hawkes
from src.utilities.pipeline_estimation import load_simulation_file, complete_pipe_estim_hawkes

(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
 fct_parameters, true_breakpoints,
 KERNEL, times_estimation, half_width_kernel) = fetch_param_json_loader_simulation(False)

SIMULATE = True  # flag: simulate or create from the estimators one file.
if SIMULATE:
    nb_line = int(sys.argv[1]) - 1  # bc we run  nb_simul + 1. Should go from 1 to ...
    # nb_line = 0 # debug case scenario
    assert nb_line < NB_SIMUL, "The line number given does not match with how many data were simulated!"

    # load data
    data_simulated = load_simulation_file(["dataset_hawkes", STR_CONFIG], compress=False)
    # slice it for the particular task nb
    data_simulated["time series"] = [data_simulated["time series"][nb_line]]

    estim_hp = complete_pipe_estim_hawkes(data_simulated, times_estimation, KERNEL, silent=True)

    # storing on the disk.
    path_result_directory = linker_path_to_result_file(["multithread_hawkes", f"{STR_CONFIG}_temp", f"result_{nb_line + 1}.csv"])
    estim_hp.to_csv(path_result_directory)


else:
    # create one big out of all of them
    path_result_directory = linker_path_to_result_file(["multithread_hawkes", f"{STR_CONFIG}_temp"])
    assert not is_empty_file(
        path_result_directory), f"file must contain some data. Directory {path_result_directory} is empty."
    list_estim_hp = Estim_hawkes.folder_csv2list_estim(path_result_directory)
    estim_hp = Estim_hawkes.merge(list_estim_hp)
    path_super_result = linker_path_to_result_file(["multithread_hawkes", STR_CONFIG, f"results_together.csv"])
    estim_hp.to_csv(path_super_result)

    # delete the old estimators.
    remove_files_from_dir(path_result_directory, "result_", 'csv')
