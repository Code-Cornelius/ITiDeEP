import numpy as np

from data_input.json.parameter_loader import fetch_param_json_loader_simulation, fetch_param_json_loader_itideep, \
    t_max_parameters
from corai_plot import APlot
from root_dir import linker_path_to_result_file
from src.estim_hawkes.estim_hawkes import Estim_hawkes
from src.estim_hawkes.relplot_hawkes import Relplot_hawkes
from src.utilities.fct_itideep import creator_kernels_adaptive
from src.utilities.general import count_the_nans

STR_CONFIG = "MSE"
########## CHOSE A NUMBER OF THE TMAX YOU WISH TO COMPARE
TMAX_NB = 9
#########################################################


(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, _, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, _, parameters, t0, time_batch,
 fct_parameters, true_breakpoints, _, _, _) = fetch_param_json_loader_simulation(False, STR_CONFIG)
(L, R, h, l, CONSIDERED_PARAM, ALL_KERNELS_DRAWN,
 TYPE_ANALYSIS, NUMBER_OF_BREAKPOINTS, MODEL,
 MIN_SIZE, WIDTH) = fetch_param_json_loader_itideep(flagprint=True, str_config=STR_CONFIG)
T_MAX = np.linspace(6000, 33000, 10)[TMAX_NB]
(KERNEL, half_width_kernel, id_hp, times_estimation) = t_max_parameters(DIM, KERNEL_DIVIDER, NB_DIFF_TIME_ESTIM,
                                                                        NB_POINTS_TT, SEED,
                                                                        STYL, T_MAX, UNDERLYING_FUNCTION_NUMBER)

path_result_res1 = linker_path_to_result_file(
    ["MSE", f"{STR_CONFIG}_res_1", f"data_together_{TMAX_NB}", "results_together.csv"])
estim_hp = Estim_hawkes.from_csv(path_result_res1)
list_of_kernels = [KERNEL] * NB_DIFF_TIME_ESTIM
kernels_to_plot = list_of_kernels, times_estimation
count_the_nans(estim_hp, NB_DIFF_TIME_ESTIM, NB_SIMUL)
_, list_of_kernels_for_itideep = creator_kernels_adaptive(estim_hp, times_estimation, CONSIDERED_PARAM,
                                                          [half_width_kernel] * len(times_estimation), L, R, h,
                                                          l, tol=0.05, silent=True)

path_result_res2 = linker_path_to_result_file(
    ["MSE", f"{STR_CONFIG}_res_2", f"data_together_{TMAX_NB}", "results_together.csv"])
estim_hp2 = Estim_hawkes.from_csv(path_result_res2)
count_the_nans(estim_hp2, NB_DIFF_TIME_ESTIM, NB_SIMUL)
relplot_hp = Relplot_hawkes(estimator_hawkes=estim_hp2, fct_parameters=fct_parameters,
                            number_of_estimations=NB_SIMUL, T_max=T_MAX)

estim_total = estim_hp.append(estim_hp2.df)
kernels_to_plot_hitdep = list_of_kernels_for_itideep, times_estimation
relplot_hp = Relplot_hawkes(estimator_hawkes=estim_total, fct_parameters=fct_parameters,
                            number_of_estimations=NB_SIMUL, T_max=T_MAX)
relplot_hp.lineplot("value", column_name_true_values="true value", hue="weight function",
                    palette="Dark2", envelope_flag=False, path_save_plot=None,
                    kernels_to_plot=kernels_to_plot_hitdep, draw_all_kern=False)

APlot.show_plot()
