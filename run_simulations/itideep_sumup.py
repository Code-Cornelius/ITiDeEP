from data_input.json.parameter_loader import fetch_param_json_loader_simulation, fetch_param_json_loader_itideep
from corai_plot import APlot
from root_dir import linker_path_to_result_file
from src.estim_hawkes.estim_hawkes import Estim_hawkes
from src.estim_hawkes.relplot_hawkes import Relplot_hawkes
from src.utilities.fct_itideep import creator_kernels_adaptive
from src.utilities.general import count_the_nans

(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
 fct_parameters, true_breakpoints, KERNEL, times_estimation,
 half_width_kernel) = fetch_param_json_loader_simulation(True)
(L, R, h, l, CONSIDERED_PARAM, ALL_KERNELS_DRAWN,
 TYPE_ANALYSIS, NUMBER_OF_BREAKPOINTS, MODEL,
 MIN_SIZE, WIDTH) = fetch_param_json_loader_itideep(flagprint=True)

path_result_res1 = linker_path_to_result_file(["result_hawkes_hitdep1", STR_CONFIG, "results_together.csv"])
path_plot_res1 = linker_path_to_result_file(["result_hawkes_hitdep1", STR_CONFIG, ""])

print("\n~~~~~~~~~~~~~~~~~~~~~~~~First Step of ITDEP~~~~~~~~~~~~~~~~~~~~~~~~")
estim_hp = Estim_hawkes.from_csv(path_result_res1)
list_of_kernels = [KERNEL] * NB_DIFF_TIME_ESTIM
kernels_to_plot = list_of_kernels, times_estimation
count_the_nans(estim_hp, NB_DIFF_TIME_ESTIM, NB_SIMUL)
relplot_hp = Relplot_hawkes(estimator_hawkes=estim_hp, fct_parameters=fct_parameters,
                            number_of_estimations=NB_SIMUL, T_max=T_MAX)
# relplot_hp.lineplot("value", column_name_true_values="true value",
#                     envelope_flag=True, path_save_plot=path_plot_res1,
#                     kernels_to_plot=kernels_to_plot, draw_all_kern=False)

# kernels for hitdep:
print("\n~~~~~~~~~~~~~~~~~~~~~~~~Second Step of ITDEP~~~~~~~~~~~~~~~~~~~~~~~~")
_, list_of_kernels_for_hitdep = creator_kernels_adaptive(estim_hp, times_estimation, CONSIDERED_PARAM,
                                                         [half_width_kernel] * len(times_estimation), L, R, h,
                                                         l, tol=0.1, silent=True)

path_result_res2 = linker_path_to_result_file(["result_hawkes_hitdep2", STR_CONFIG, "results_together.csv"])
path_plot_res2 = linker_path_to_result_file(["result_hawkes_hitdep2", STR_CONFIG, ""])

estim_hp2 = Estim_hawkes.from_csv(path_result_res2)
kernels_to_plot_itideep = list_of_kernels_for_hitdep, times_estimation
count_the_nans(estim_hp2, NB_DIFF_TIME_ESTIM, NB_SIMUL)
relplot_hp = Relplot_hawkes(estimator_hawkes=estim_hp2, fct_parameters=fct_parameters,
                            number_of_estimations=NB_SIMUL, T_max=T_MAX)
# relplot_hp.lineplot("value", column_name_true_values="true value",
#                     envelope_flag=True, path_save_plot=path_plot_res2,
#                     kernels_to_plot=kernels_to_plot_itideep, draw_all_kern=False)

# kernels for ITiDeEP:
print("\n~~~~~~~~~~~~~~~~~~~~~~~~Image together of ITiDeEP~~~~~~~~~~~~~~~~~~~~~~~~")
estim_total = estim_hp.append(estim_hp2.df)

path_plot_res3 = linker_path_to_result_file(["result_hawkes_hitdep3", STR_CONFIG, "plot", ""])
kernels_to_plot_itideep = list_of_kernels_for_hitdep, times_estimation
relplot_hp = Relplot_hawkes(estimator_hawkes=estim_total, fct_parameters=fct_parameters,
                            number_of_estimations=NB_SIMUL, T_max=T_MAX)
relplot_hp.lineplot("value", column_name_true_values="true value", hue="weight function",
                    palette="Dark2", envelope_flag=False, path_save_plot=path_plot_res3,
                    kernels_to_plot=kernels_to_plot_itideep, draw_all_kern=True)

APlot.show_plot()
