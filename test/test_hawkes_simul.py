import unittest

import numpy as np
# priv lib
from corai_plot import APlot
from tqdm import tqdm

from run_simulations.setup_parameters import setup_parameters
# other file
from src.hawkes.hawkes_process import Hawkes_process
from src.utilities.fct_type_process import function_parameters_hawkes


class Test_hawkes_simul(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        APlot.show_plot()

    def test_all_hawkes_run(self):
        for d in [2]:
            for switch in tqdm([0, 1, 21, 22, 3, 42]):
                styl = 1
                parameters, t0, time_batch = setup_parameters(42, d, styl)
                parameter_functions, true_breakpoints = function_parameters_hawkes(switch, parameters)
                hawksy = Hawkes_process(parameter_functions)

                tt = np.linspace(0, 2E2, int(1E5))
                intensity, time_real = hawksy.simulation_Hawkes_exact_with_burn_in(tt, nb_of_sim=100000, plotFlag=True,
                                                                                   silent=True)
                hawksy.plot_hawkes(tt, time_real, intensity)
            APlot.show_plot()
