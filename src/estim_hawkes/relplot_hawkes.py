# normal libraries

import numpy as np

# priv_libraries
from corai_estimator import Relplot_estimator
from src.estim_hawkes.plot_estim_hawkes import Plot_estim_hawkes


# other files


class Relplot_hawkes(Plot_estim_hawkes, Relplot_estimator):
    EVOLUTION_COLUMN = 'time estimation'
    ESTIMATION_COLUMN_NAME = 'value'
    TRUE_ESTIMATION_COLUMN_NAME = 'true value'

    def __init__(self, estimator_hawkes, fct_parameters, number_of_estimations, T_max, **kwargs):
        super().__init__(estimator_hawkes, fct_parameters, number_of_estimations, T_max, **kwargs)

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_dict_fig(self, separators, key):
        # TODO LOOK AT THE TITLE
        title = self.generate_title(parameters=separators,
                                    parameters_value=key,
                                    before_text="true value drawn.",
                                    extra_text="Estimation over 5-95% of the interval, batches of {} simulations, time: {} until {}",
                                    extra_arguments=[self.number_of_estimations,
                                                     0.05 * self.T_max,
                                                     0.95 * self.T_max])

        fig_dict = {'title': "Time Dependant Estimation of Hawkes Process, " + title,
                    'xlabel': 'Time',
                    'ylabel': "Estimation"}
        return fig_dict

    def lineplot(self, column_name_draw, column_name_true_values=None, envelope_flag=True, separators_plot=None,
                 palette='PuOr',
                 hue=None, style=None, markers=None, sizes=None,
                 dict_plot_for_main_line={}, path_save_plot=None, list_aplots=None,
                 kernels_to_plot=None, draw_all_kern=False,
                 *args, **kwargs):
        # wip add kernel height on the right as ylabel

        NB_OF_KERNELS_DRAWN = 18

        times_estimation = self.get_values_evolution_column(self.estimator.df)
        # if there is only one kernel plot, then the label is not written. Then, we write it manually:
        if self.estimator.df.nunique()["weight function"] == 1:
            dict_plot_for_main_line = {'label': kernels_to_plot[0][0].name}
        else:
            dict_plot_for_main_line = {}
        current_plots, keys = super().lineplot(column_name_draw, column_name_true_values, envelope_flag,
                                               separators_plot, palette, hue, style, markers, sizes,
                                               dict_plot_for_main_line=dict_plot_for_main_line,
                                               path_save_plot=None, list_aplots=list_aplots, *args, **kwargs)
        # TODO 09/07/2021 nie_k:  take care of the label. If one kernel given, give the name.
        if kernels_to_plot is not None:
            list_kernels, list_position_centers = kernels_to_plot
            for plot, key in zip(current_plots, keys):
                for counter_kern, (kernel, center_time) in enumerate(zip(list_kernels, list_position_centers)):
                    condition = draw_all_kern \
                                or not (len(times_estimation) // NB_OF_KERNELS_DRAWN) \
                                or (not counter_kern % (len(times_estimation) // NB_OF_KERNELS_DRAWN))
                    if condition:
                        # first : whether I want all kernels to be drawn
                        # the second condition is checking whether len(times_estimation) > NB_OF_KERNELS_DRAWN. Otherwise, there is a modulo by 0, which returns an error.
                        # third condition is true for all NB_OF_KERNELS_DRAWN selected kernels.
                        tt = [np.linspace(0, self.T_max, 1000)]
                        yy = kernel(tt, center_time, self.T_max)
                        plot.uni_plot_ax_bis(nb_ax=0, xx=tt[0], yy=yy[0],
                                             dict_plot_param={"color": "m", "markersize": 0, "linewidth": 0.4,
                                                              "linestyle": "--"})
                        # plot line on the x center of the kernel
                        lim_ = plot._axs[0].get_ylim()
                        plot.plot_vertical_line(center_time, np.linspace(0, lim_[-1] * 0.92, 5), nb_ax=0,
                                                dict_plot_param={"color": "k", "markersize": 0, "linewidth": 0.2,
                                                                 "linestyle": "--"})
                super()._saveplot(plot, path_save_plot, 'relplot_', key)
        return current_plots