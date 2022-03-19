##### normal libraries
import unittest

##### other files
from data_input.json.json_loader import STR_CONFIG
from data_input.json.parameter_loader import fetch_param_json_loader_simulation, fetch_param_json_loader_itideep
from corai_plot import APlot
##### my libraries
from corai_util.tools.src.function_iterable import is_iterable
from root_dir import linker_path_to_result_file
from src.estim_hawkes.relplot_hawkes import Relplot_hawkes
from src.utilities.fct_itideep import creator_kernels_adaptive
from src.utilities.pipeline_estimation import load_simulation_file, complete_pipe_estim_hawkes


# L = L_PARAM
# R = R_PARAM
# h = h_PARAM
# if l_PARAM == "automatic with respect to the total size":
#     l = width_kernel / T_max / 2
# elif isinstance(l_PARAM, float):
#     l = l_PARAM
# else:
#     raise Error_not_allowed_input("give a proper value to l.")


class Test_Estimation_Hawkes(unittest.TestCase):
    # section ######################################################################
    #  #############################################################################
    # setup

    higher_percent_bound = 0.95
    lower_percent_bound = 0.05

    def setUp(self):
        # run the run_simulations/1create_dataset_hawkes.
        list_accept_config = ["test_config", "test_config_multidim", "quick_config"]
        print(STR_CONFIG)
        if STR_CONFIG not in list_accept_config:
            raise ValueError("The config for tests is test_config or quick_config.")

    def tearDown(self):
        APlot.show_plot()

    # section ######################################################################
    #  #############################################################################
    # tests

    def test_estimation_over_time(self):
        # be careful, it retrieves the config from STR_CONFIG.
        (STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
         NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
         fct_parameters, true_breakpoints, KERNEL, times_estimation,
         half_width_kernel) = fetch_param_json_loader_simulation(True)
        data_simulated = load_simulation_file(["dataset_hawkes", STR_CONFIG], compress=False)
        estim_hp = complete_pipe_estim_hawkes(data_simulated, times_estimation, KERNEL, silent=False)
        if is_iterable(KERNEL):
            assert len(KERNEL) == len(times_estimation), \
                "The number of kernel must match the number of estimation points."
        else:  # kernel is not iterable, hence it is a unique kernel.
            list_of_kernels = [KERNEL] * len(times_estimation)
        kernels_to_plot = list_of_kernels, times_estimation
        print(estim_hp)
        relplot_hp = Relplot_hawkes(estimator_hawkes=estim_hp, fct_parameters=fct_parameters,
                                    number_of_estimations=NB_SIMUL, T_max=T_MAX)
        relplot_hp.lineplot("value", column_name_true_values="true value",
                            envelope_flag=True, path_save_plot=None,
                            kernels_to_plot=kernels_to_plot, draw_all_kern=True)

    def test_estimation_over_time_iterative(self):
        # PARAMETERS
        (STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
         NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
         fct_parameters, true_breakpoints, KERNEL, times_estimation,
         half_width_kernel) = fetch_param_json_loader_simulation(True)
        (L, R, h, l, CONSIDERED_PARAM, ALL_KERNELS_DRAWN,
         TYPE_ANALYSIS, NUMBER_OF_BREAKPOINTS, MODEL,
         MIN_SIZE, WIDTH) = fetch_param_json_loader_itideep(flagprint=True)

        data_simulated = load_simulation_file(["dataset_hawkes", STR_CONFIG], compress=False)
        first_estimation_hp = complete_pipe_estim_hawkes(data_simulated, times_estimation, KERNEL, silent=False)

        # storing on the disk.
        path_result = linker_path_to_result_file(["results1_together.csv"])
        # first_estimation_hp = Estim_hawkes.from_csv(path_result)
        first_estimation_hp.to_csv(path_result)
        print(first_estimation_hp)

        # Kernel for plotting:
        if is_iterable(KERNEL):
            assert len(KERNEL) == len(times_estimation), \
                "The number of kernel must match the number of estimation points."
            list_of_kernels_first_estim = KERNEL
        else:  # kernel is not iterable, hence it is a unique kernel.
            list_of_kernels_first_estim = [KERNEL] * len(times_estimation)

        kernels_to_plot = list_of_kernels_first_estim, times_estimation
        relplot_hp = Relplot_hawkes(estimator_hawkes=first_estimation_hp, fct_parameters=fct_parameters,
                                    number_of_estimations=NB_SIMUL, T_max=T_MAX)
        relplot_hp.lineplot("value", column_name_true_values="true value", envelope_flag=True, path_save_plot=None,
                            kernels_to_plot=kernels_to_plot, draw_all_kern=ALL_KERNELS_DRAWN)

        ####iterative ITiDeEP
        first_estimator = first_estimation_hp  # get the mean

        _, list_of_kernels_for_hitdep = creator_kernels_adaptive(first_estimator, times_estimation, CONSIDERED_PARAM,
                                                                 [half_width_kernel] * len(times_estimation), L, R, h,
                                                                 l, tol=0.1, silent=False)

        second_estimation_hp = complete_pipe_estim_hawkes(data_simulated, times_estimation, list_of_kernels_for_hitdep,
                                                          silent=False)
        print(second_estimation_hp)

        kernels_to_plot = list_of_kernels_for_hitdep, times_estimation
        relplot_hp = Relplot_hawkes(estimator_hawkes=second_estimation_hp, fct_parameters=fct_parameters,
                                    number_of_estimations=NB_SIMUL, T_max=T_MAX)
        relplot_hp.lineplot("value", column_name_true_values="true value", envelope_flag=True, path_save_plot=None,
                            kernels_to_plot=kernels_to_plot, draw_all_kern=ALL_KERNELS_DRAWN)
        # storing on the disk.
        path_result = linker_path_to_result_file(["results2_together.csv"])
        second_estimation_hp.to_csv(path_result)
