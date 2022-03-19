from data_input.json.parameter_loader import fetch_param_json_loader_simulation
from corai_plot import APlot
from root_dir import linker_path_to_result_file
from src.estim_hawkes.estim_hawkes import Estim_hawkes
from src.estim_hawkes.relplot_hawkes import Relplot_hawkes
from src.utilities.general import count_the_nans

(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
 fct_parameters, true_breakpoints, KERNEL, times_estimation, half_width_kernel) = fetch_param_json_loader_simulation(
    True)

NAME_FOLDER = 'test'

path_result = linker_path_to_result_file(["result_hawkes_" + NAME_FOLDER, STR_CONFIG, "results_together.csv"])
path_plot = linker_path_to_result_file(["result_hawkes_" + NAME_FOLDER, STR_CONFIG, ""])

# path_result = linker_path_to_result_file(["multithread_hawkes", STR_CONFIG, "results_together.csv"])
# path_plot = linker_path_to_result_file(["multithread_hawkes", STR_CONFIG, ""])

estim_hp = Estim_hawkes.from_csv(path_result)

list_of_kernels = [KERNEL] * NB_DIFF_TIME_ESTIM
kernels_to_plot = list_of_kernels, times_estimation

count_the_nans(estim_hp, NB_DIFF_TIME_ESTIM, NB_SIMUL)

relplot_hp = Relplot_hawkes(estimator_hawkes=estim_hp, fct_parameters=fct_parameters,
                            number_of_estimations=NB_SIMUL, T_max=T_MAX)
relplot_hp.lineplot("value", column_name_true_values="true value",
                    envelope_flag=True, path_save_plot=path_plot,
                    kernels_to_plot=kernels_to_plot, draw_all_kern=False)

APlot.show_plot()
