import unittest

# priv lib
from corai_plot import APlot, AColorsetDiscrete

# other file
from src.hawkes.kernel import *


class Test_kernels(unittest.TestCase):

    def setUp(self):
        self.list_kern_fct = [fct_epa,
                              fct_biweight,
                              fct_plain,
                              fct_top_hat]
        self.list_kernels = [Kernel(fct_kernel=fct, name="", a=-25, b=25) for fct in self.list_kern_fct]
        self.list_kernels.append(Kernel(fct_kernel=fct_truncnorm, name="", a=-25, b=25, sigma=20))
        self.xx = [np.linspace(0, 100, 1000)]

    def tearDown(self):
        APlot.show_plot()

    def test_kernel_plot(self):
        res = []
        # import seaborn as sns
        # sns.set()
        for kern in self.list_kernels:
            yy = kern(self.xx, 50, 100)[0]
            APlot(datax=self.xx[0], datay=yy)
            res.append(yy)

        plot = APlot((1, 1))
        names = ['kernel epa',
                 'kernel biweight',
                 'kernel truncated normal',
                 'kernel top-hat']
        for data, c, kernel in zip(np.stack(res), AColorsetDiscrete('Dark2'), names):
            plot.uni_plot(nb_ax=0, xx=self.xx[0], yy=data,
                          dict_plot_param={'color': c, 'label': kernel, 'linewidth': 2},
                          dict_ax={'title': '', 'xlabel': '', 'ylabel': ''})
        plot.show_legend()
