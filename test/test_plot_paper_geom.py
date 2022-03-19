# normal libraries
import unittest

import numpy as np
# priv libaries:
from corai_plot import APlot

# other file:
from src.utilities.fct_itideep import special_sin, AKDE_scaling, jump_rescale


########### test adaptive window
class Test_plot_geom_images(unittest.TestCase):
    """
    Plotting some images.
    """

    def setUp(self):
        pass

    def tearDown(self):
        APlot.show_plot()

    def test_plot_rescale_sin(self):
        # if using this test, add these constants for the values G  R  L:
        # L_quant = np.array([L * 100])
        # R_quant = np.array([R * 100])
        # G = np.array([10])

        T_t = np.linspace(0.1, 100, 10000)  # [np.newaxis, ...]

        h = 5
        res = special_sin(value_at_each_time=T_t, L=0.02, R=0.75, h=h, l=0.5)

        min = np.array([0.02 * 100])
        max = np.array([0.75 * 100])
        aplot = APlot(how=(1, 1))
        aplot.uni_plot(nb_ax=0, xx=T_t, yy=res, dict_ax={'title': 'Rescale function for ITiDeEP',
                                                         'xlabel': 'Value', 'ylabel': 'Scaling'})
        aplot.plot_vertical_line(10, np.linspace(-0.1, h * (1.1), 4), nb_ax=0,
                                 dict_plot_param={'color': 'k', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                                  'label': 'geom. mean'})
        aplot.plot_vertical_line(min, np.linspace(-0.1, h * (1.1), 4), nb_ax=0,
                                 dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                                  'label': 'lower bound'})
        aplot.plot_vertical_line(max, np.linspace(-0.1, h * (1.1), 4), nb_ax=0,
                                 dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                                  'label': 'upper bound'})
        aplot.show_legend()

    def test_plot_old_fct(self):
        T_t = [np.linspace(0.1, 100, 10000)]
        G = 10.
        res = AKDE_scaling(T_t, G, gamma=0.5)

        aplot = APlot(how=(1, 1))
        aplot.uni_plot(nb_ax=0, xx=T_t[0], yy=res[0],
                       dict_ax={'title': 'Adaptive scaling for Adaptive Window Width',
                                'xlabel': 'Value', 'ylabel': 'Scaling'})
        aplot.plot_vertical_line(G, np.linspace(-1, 10, 1000), nb_ax=0,
                                 dict_plot_param={'color': 'k', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                                  'label': 'geom. mean'})
        aplot.show_legend()

    def test_function_jump_rescale(self):
        # if using this test, add these constants for the values G  R  L:
        # L_quant = np.array([L * 100])
        # R_quant = np.array([R * 100])
        # G = np.array([50])

        T_t = np.linspace(0.1, 100, 10000)  # [np.newaxis, ...]

        h = 6
        res = jump_rescale(value_at_each_time=T_t, L=0.2, R=0.75, h=h, l=1.)

        min = np.array([0.20 * 100])
        max = np.array([0.75 * 100])
        aplot = APlot(how=(1, 1))
        aplot.uni_plot(nb_ax=0, xx=T_t, yy=res, dict_ax={'title': 'Rescale function (jump adapted) for ITiDeEP',
                                                         'xlabel': 'Value', 'ylabel': 'Scaling'})
        aplot.plot_vertical_line(50, np.linspace(-0.1, h * (1.1), 4), nb_ax=0,
                                 dict_plot_param={'color': 'k', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                                  'label': 'geom. mean'})
        aplot.plot_vertical_line(min, np.linspace(-0.1, h * (1.1), 4), nb_ax=0,
                                 dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                                  'label': 'lower bound'})
        aplot.plot_vertical_line(max, np.linspace(-0.1, h * (1.1), 4), nb_ax=0,
                                 dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                                  'label': 'upper bound'})
        aplot.show_legend()
