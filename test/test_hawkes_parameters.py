##### normal libraries
import unittest

from numpy import linalg as LA
##### my libraries
from corai_plot import APlot, AColorsetContinuous

##### other files
from run_simulations.setup_parameters import setup_parameters
from src.utilities.fct_type_process import *


class Test_hawkes_parameters(unittest.TestCase):
    """
    The tests are manual. They are there so one can check that the parameters are correct.
    """

    def setUp(self):
        pass

    def tearDown(self):
        APlot.show_plot()

    def test_plot_type_process(self):
        """ Plots the evolution of each function of the parameters. """
        length_time_interval = 0.95
        time_burn_in = 0.05
        xx = np.linspace(0, length_time_interval + time_burn_in, 1000)

        yy_constant = constant_parameter(xx, 0.5)
        yy_linear = linear_growth(xx, 1, 0.2, length_time_interval, time_burn_in)
        yy_jump = one_jump(xx, 0.7, 0.2, 0.3, length_time_interval, time_burn_in)
        yy_mount = moutain_jump(xx, 0.6, 1., 0.2, 0.1, length_time_interval, time_burn_in)
        yy_cos = periodic_stop(xx, length_time_interval, 0.4, 0.3, time_burn_in)
        yys = [yy_constant, yy_linear, yy_jump, yy_mount, yy_cos]

        titles = ["Constant", "Linear Growth", "One Jump", "Moutain Jump", "Periodic and Stop"]
        for yy, title in zip(yys, titles):
            aplot = APlot()
            aplot.uni_plot(0, xx, yy, dict_ax={"title": title, "xlabel": "time", "ylabel": "value parameter"})
            aplot.plot_vertical_line(time_burn_in, np.linspace(0, 1, 2), 0,
                                     {'color': 'black', 'linestyle': '--', 'markersize': '0'})

    def test_function_parameters_hawkes(self):
        """ Plots the parameters for the different switches together for each parameter nu alpha beta.
        Useful to see if there is some issue between the processes, like beta above alpha."""
        length_time_interval = 0.95
        time_burn_in = 0.05

        for d in [2]:  # or 1, 2, 5
            colormap = AColorsetContinuous('hsv', d * d)

            parameters, _, _ = setup_parameters(42, d, styl=1)
            for switch in [22, 3, 42]:  # [0, 1, 21, 22, 3, 42]:  # if dim 1, 41; 2 put 42; 5 no 4.

                print(f"d: {d}, switch: {switch}.")
                the_update_functions, true_breakpoints = function_parameters_hawkes(switch, parameters)

                xx = np.linspace(0, length_time_interval + time_burn_in, 1000)

                aplot = APlot()
                for c, i in zip(colormap, range(len(the_update_functions[0]))):
                    yy = the_update_functions[0][i](xx, length_time_interval, time_burn_in)
                    aplot.uni_plot(0, xx, yy,
                                   dict_plot_param={"label": f"{i}", 'color': c},
                                   dict_ax={"title": f'dimension: {d} and switch: {switch} for parameter $\mu$',
                                            "xlabel": "time", "ylabel": "value parameter"})
                aplot.plot_vertical_line(time_burn_in, np.linspace(0, 0.5, 2), 0,
                                         {'color': 'black', 'linestyle': '--', 'markersize': '0',
                                          'label': 'left side is burn-in'})
                aplot.show_legend()

                aplot = APlot()
                for i in range(len(the_update_functions[1])):
                    for j in range(len(the_update_functions[1][0])):
                        c = colormap[i * len(the_update_functions[1][0]) + j]
                        yy = the_update_functions[1][i][j](xx, length_time_interval, time_burn_in)
                        aplot.uni_plot(0, xx, yy,
                                       dict_plot_param={"label": f"{i},{j}", 'color': c},
                                       dict_ax={"title": f'dimension: {d} and switch: {switch} for parameter $\\alpha$',
                                                "xlabel": "time", "ylabel": "value parameter"})
                aplot.plot_vertical_line(time_burn_in, np.linspace(0, 3, 2), 0,
                                         {'color': 'black', 'linestyle': '--', 'markersize': '0',
                                          'label': 'left side is burn-in'})

                aplot.show_legend()

                aplot = APlot()
                for i in range(len(the_update_functions[2])):
                    for j in range(len(the_update_functions[2][0])):
                        c = colormap[i * len(the_update_functions[1][0]) + j]
                        yy = the_update_functions[2][i][j](xx, length_time_interval, time_burn_in)
                        aplot.uni_plot(0, xx, yy,
                                       dict_plot_param={"label": f"{i},{j}", 'color': c},
                                       dict_ax={"title": f'dimension: {d} and switch: {switch} for parameter $\\beta$',
                                                "xlabel": "time", "ylabel": "value parameter"})
                aplot.plot_vertical_line(time_burn_in, np.linspace(0, 3, 2), 0,
                                         {'color': 'black', 'linestyle': '--', 'markersize': '0',
                                          'label': 'left side is burn-in'})

                aplot.show_legend()

            APlot.show_plot()  # between each dimensions

    def test_parameters_eigenvalues(self):
        """ Checks eigenvalues for critical stage of hawkes."""
        length_time_interval = 0.95
        time_burn_in = 0.05
        nb_points = 200

        for d in [2]:  # or 1, 2, 5
            parameters, _, _ = setup_parameters(42, d, styl=1)
            for switch in [3]:  # if dim 1, 41; 2 put 42; 5 no 4.

                the_update_functions, _ = function_parameters_hawkes(switch, parameters)
                xx = np.linspace(0, length_time_interval + time_burn_in, nb_points)

                alpha = np.zeros((d, d, nb_points))
                beta = np.zeros((d, d, nb_points))

                for i in range(len(the_update_functions[1])):
                    for j in range(len(the_update_functions[1][0])):
                        alpha[i, j, :] = the_update_functions[1][i][j](xx, length_time_interval, time_burn_in)
                        beta[i, j, :] = the_update_functions[2][i][j](xx, length_time_interval, time_burn_in)
                beta[beta < 0.01] = 1E8
                # replacing small values of beta by infinity,
                # as we want the division by such value to be as close to zero as possible

                eigenval_vect = np.zeros(nb_points)
                for k in range(nb_points):
                    params_of_matrix = alpha[:, :, k] / beta[:, :, k]
                    eigenval, _ = LA.eig(params_of_matrix)
                    eigenval_vect[k] = np.max(np.abs(eigenval))

                aplot = APlot()
                aplot.uni_plot(0, xx, eigenval_vect,
                               dict_ax={"title": f'dimension: {d} and switch: {switch}',
                                        "xlabel": "time", "ylabel": "Spectral Radius"})
                APlot.show_plot()
