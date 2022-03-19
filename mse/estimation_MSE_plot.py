import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from corai_util.tools.src.function_file import is_empty_file

from data_input.json.parameter_loader import fetch_param_json_loader_simulation, fetch_param_json_loader_itideep
from root_dir import linker_path_to_result_file
from src.estim_hawkes.estim_hawkes import Estim_hawkes

sns.set()

STR_CONFIG = "MSE"
(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, _, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
 fct_parameters, true_breakpoints, _, _, _) = fetch_param_json_loader_simulation(False, STR_CONFIG)
(L, R, h, l, CONSIDERED_PARAM, ALL_KERNELS_DRAWN,
 TYPE_ANALYSIS, NUMBER_OF_BREAKPOINTS, MODEL,
 MIN_SIZE, WIDTH) = fetch_param_json_loader_itideep(flagprint=True, str_config=STR_CONFIG)
# should match the data given in the script.sh
NB_T_MAX = 10  # from 1 to 10.
NB_TH_OF_CURRENT_ESTIMATION = 2  # any int > 0. Represents the refinement of the ITiDeEP.
# 1 is the first naive estimation.
# The number given is the number of lines on the plot / nb of repetition of the estimation process undergone.
# Only possible to plot all the lines (1, 2...) together and not a subset of it not including the lower part.
#########
LIST_T_MAX = np.linspace(6000, 33000, NB_T_MAX)
#######################################################


# TODO  explain gather result in readme + explain MSE pipeline.
# We use this file to gather the estimation together (gather function) and then plot the curve of the MSE.

matrix_err_tmax_APE = np.zeros((NB_TH_OF_CURRENT_ESTIMATION, len(LIST_T_MAX)))
matrix_err_tmax_SPE = np.zeros((NB_TH_OF_CURRENT_ESTIMATION, len(LIST_T_MAX)))

iter_refinement = NB_TH_OF_CURRENT_ESTIMATION
while iter_refinement > 0:  # we collect the data from
    # NB_TH_OF_CURRENT_ESTIMATION to 1 by reducing by 1 at every iteration.

    for i_tmax in range(len(LIST_T_MAX)):
        ######################
        # gather results of previous estimation for a given T max
        ######################
        path_result_directory = linker_path_to_result_file(["MSE",
                                                            f"{STR_CONFIG}_res_{iter_refinement}",
                                                            f"data_{i_tmax}", ""])
        assert not is_empty_file(path_result_directory), \
            f"file must contain some data. Directory {path_result_directory} is empty."

        list_estim_hp = Estim_hawkes.folder_csv2list_estim(path_result_directory)
        estim_hp = Estim_hawkes.merge(list_estim_hp)  # new estim gathered result
        path_super_result = linker_path_to_result_file(
            ["MSE",
             f"{STR_CONFIG}_res_{iter_refinement}",
             f"data_together_{i_tmax}",
             f"results_together.csv"])
        estim_hp.to_csv(path_super_result)  # saved gather result
        ######################
        # compute error:
        ######################
        path_result_res = linker_path_to_result_file(
            ["MSE", f"{STR_CONFIG}_res_{iter_refinement}", f"data_together_{i_tmax}", "results_together.csv"])
        print("Reading: ", path_result_res)

        estim_hp = Estim_hawkes.from_csv(path_result_res)

        estim_hp.add_SPE_APE_col()  # computed the SRE per parameter

        groupby_param, keys = estim_hp.groupby(['parameter', 'm', 'n'])
        total_SPE_APE = (groupby_param.get_group(('alpha', 0, 0))[["time estimation", 'SPE', 'APE']]
                         .sort_values(by="time estimation").reset_index(drop=True))  # a copy is made
        # : we create a container where the error is aggregated.
        total_SPE_APE['SPE'] = 0  # we empty the values inside the column
        total_SPE_APE['APE'] = 0  # we empty the values inside the column
        for key in keys:
            ordered_SPE_APE = (groupby_param.get_group(key)[["time estimation", 'SPE', 'APE']]
                               .sort_values(by="time estimation").reset_index(drop=True))
            # sort to be sure we add the correct values together, drop index for prettiness.
            total_SPE_APE['SPE'] += ordered_SPE_APE['SPE']
            total_SPE_APE['APE'] += ordered_SPE_APE['APE']

        # MISRE = total_SRE.mean()["RSE"] # this is wrong. We need to compute it by hand.
        # It does not account for non converging estimations.
        total_SPE_APE_grouped = total_SPE_APE.groupby("time estimation")  # we groupby so we compute the integral
        MISPE = 0
        MIAPE = 0

        # compute the mean squared error and compute the mean absolute error
        for time in total_SPE_APE_grouped.groups:
            average_per_time = total_SPE_APE_grouped.get_group(time).mean()
            MISPE += average_per_time['SPE'] / len(total_SPE_APE_grouped.groups)
            MIAPE += average_per_time['APE'] / len(total_SPE_APE_grouped.groups)

        matrix_err_tmax_SPE[iter_refinement - 1, i_tmax] = MISPE  # store result
        matrix_err_tmax_APE[iter_refinement - 1, i_tmax] = MIAPE  # store result

    iter_refinement -= 1

dict_result = {"MISPE": matrix_err_tmax_SPE.flatten(),
               "MIAPE": matrix_err_tmax_APE.flatten(),
               "nb application ITiDeEP": np.repeat(range(NB_TH_OF_CURRENT_ESTIMATION), NB_T_MAX),
               "T max": np.tile(LIST_T_MAX, NB_TH_OF_CURRENT_ESTIMATION)}
data_err = pd.DataFrame(dict_result)

fig, ax = plt.subplots(2, 1)
sns.lineplot(x="T max", y="MISPE",
             hue="nb application ITiDeEP", marker='o',
             legend='full', ci=None, err_style="band",
             palette='Dark2', ax=ax[0],
             data=data_err)

sns.lineplot(x="T max", y="MIAPE",
             hue="nb application ITiDeEP", marker='o',
             legend='full', ci=None, err_style="band",
             palette='Dark2', ax=ax[1],
             data=data_err)

path_save_plot = linker_path_to_result_file(["MSE", f"MSE_result_{NB_TH_OF_CURRENT_ESTIMATION}" + '.png'])
fig.savefig(path_save_plot, dpi=500)
plt.show()
