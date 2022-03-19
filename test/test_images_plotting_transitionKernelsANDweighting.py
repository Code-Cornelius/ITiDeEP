# normal libraries
import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
# my libraries
from corai_plot import APlot
from corai_util.tools.src.function_recurrent import phi_numpy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# other files


class Test_images(unittest.TestCase):

    def tearDown(self):
        plt.show()

    def test_image_FKDE_AKDE_CKDE(self):
        nb_of_points = 10000
        ############################## 1
        xx = np.linspace(-15, 15, nb_of_points)
        mesh = 30. / nb_of_points
        zz = np.zeros(nb_of_points)
        my_plot = APlot(how=(1, 1))
        points = np.array([-7., -6., 1., 2., 5.])
        for f in points:
            yy = phi_numpy(xx, f, 2) / len(points)
            my_plot.uni_plot(0, xx, yy, dict_plot_param={'label': f'Kernel at {f}'})
            zz += yy

        print(np.sum(zz * mesh))

        my_plot.uni_plot(0, xx, zz, dict_plot_param={'color': 'r', 'label': 'KDE'},
                         dict_ax={'xlabel': 'Time $t$', 'ylabel': 'Probability',
                                  'title': 'KDE estimation, fixed size kernel'})
        my_plot.show_legend()

        ############################## 2
        zz = np.zeros(nb_of_points)
        my_plot = APlot(how=(1, 1))
        points = np.array([-7., -6., 1., 2., 5.])
        for f in points:
            yy = phi_numpy(xx, f, 2 * (1 + math.fabs(f) / 10)) / len(points)
            my_plot.uni_plot(0, xx, yy, dict_plot_param={'label': f'Kernel at {f}'})
            zz += yy

        my_plot.uni_plot(0, xx, zz, dict_plot_param={'color': 'r', 'label': 'KDE'},
                         dict_ax={'xlabel': 'Time $t$', 'ylabel': 'Probability',
                                  'title': 'KDE estimation, adaptive size kernel'})
        my_plot.show_legend()

        print(np.sum(zz * mesh))

        ############################## 3
        ############### left
        zz = np.zeros(nb_of_points)
        max_x = 0.125
        my_plot = APlot(how=(1, 2), sharey=True)
        my_plot.uni_plot(0, [0 for _ in xx],
                         np.linspace(-0.004, max_x, len(xx)),
                         dict_plot_param={'color': 'g', 'label': 'Estimation point',
                                          'linestyle': '--', 'linewidth': 2,
                                          'markersize': 0})
        points = np.array([-1.1, 0.5, 5.])
        for f in points:
            my_plot.uni_plot(0, [f for _ in xx],
                             np.linspace(-0.004, max_x, len(xx)),
                             dict_plot_param={'color': 'k', 'linestyle': '--', 'linewidth': 0.7,
                                              'markersize': 0,
                                              'label': None})
            yy = phi_numpy(xx, f, 2) / len(points)
            my_plot.uni_plot(0, xx, yy, dict_plot_param={'label': f'Kernel at {f}'},
                             dict_ax={'xlabel': 'Time $t$', 'ylabel': 'Probability',
                                      'title': 'Kernel represented as function of the time'})
            my_plot.plot_point(xx[nb_of_points // 2], yy[nb_of_points // 2], nb_ax=0,
                               dict_plot_param={'color': 'r', 'markersize': 8, 'marker': '*', 'label': None})
            zz += yy
        my_plot.uni_plot(0, xx, zz, dict_plot_param={'color': 'r', 'label': 'KDE'})
        print("value : ", zz[nb_of_points // 2])

        ############### right
        zz = phi_numpy(xx, 0, 2) / 3
        print(np.sum(zz * mesh))
        for f in points:
            my_plot.uni_plot(1, [f for _ in xx],
                             np.linspace(-0.004, phi_numpy(f, 0, 2) / 3, len(xx)),
                             dict_plot_param={'color': 'm', 'linestyle': '--', 'linewidth': 0.7, 'markersize': 0,
                                              'label': f'Value kernel at {f}'})
            my_plot.plot_point(f, phi_numpy(f, 0, 2) / 3, nb_ax=1,
                               dict_plot_param={'color': 'g', 'markersize': 8, 'marker': '*', 'label': None})

        my_plot.uni_plot(1, xx, zz, dict_plot_param={'color': 'r', 'label': 'Kernel for $t = 0$'})

        ### sum
        previous_f = 0
        for i, (f, c) in enumerate(zip(points, ['b', 'c', 'k'])):
            my_plot.plot_point(-15, previous_f + phi_numpy(f, 0, 2) / 3, nb_ax=1,
                               dict_plot_param={'color': c, 'markersize': 8, 'marker': '*',
                                                'label': f'cumsum leading to true result {i}'})
            my_plot.uni_plot(1, [-15 for _ in xx],
                             np.linspace(previous_f, previous_f + phi_numpy(f, 0, 2) / 3, len(xx)),
                             dict_plot_param={'color': c, 'linestyle': '--', 'linewidth': 0.7,
                                              'markersize': 0, 'label': None},
                             dict_ax={'xlabel': 'Time event $t_i$', 'ylabel': '',
                                      'title': 'Kernel represented as function of events $t_i$'})
            previous_f += phi_numpy(f, 0, 2) / 3

        my_plot.show_legend()

    def test_image_CKDE(self):
        nb_of_points = 10000
        xx = np.linspace(-15, 15, nb_of_points)

        ############################## 3
        ############### left
        max_x = 0.23
        my_plot = APlot(how=(1, 2), sharey=True)
        my_plot.uni_plot(0, [0 for _ in xx],
                         np.linspace(-0.004, max_x, len(xx)),
                         dict_plot_param={'color': 'g', 'label': 'Estimation point',
                                          'linestyle': '--', 'linewidth': 2,
                                          'markersize': 0})
        yy = (phi_numpy(xx, -4, 3) + phi_numpy(xx, 1, 1)) / 2
        my_plot.uni_plot(0, xx, yy, dict_plot_param={'color': 'r', 'label': 'Kernel for $t_i = 0$'},
                         dict_ax={'xlabel': 'Time t', 'ylabel': 'Probability',
                                  'title': 'Kernel represented as function of the time'})

        ############### right
        zz = (phi_numpy(-xx, -4, 3) + phi_numpy(-xx, 1, 1)) / 2
        my_plot.uni_plot(1, [0 for _ in xx],
                         np.linspace(-0.004, max_x, len(xx)),
                         dict_plot_param={'color': 'g', 'label': 'Estimation point',
                                          'linestyle': '--', 'linewidth': 2,
                                          'markersize': 0})
        my_plot.uni_plot(1, xx, zz, dict_plot_param={'color': 'r', 'label': 'Kernel for $t = 0$'},
                         dict_ax={'xlabel': 'Time event $t_i$', 'ylabel': '',
                                  'title': 'Kernel represented as function of events $t_i$'})

        my_plot.show_legend()
