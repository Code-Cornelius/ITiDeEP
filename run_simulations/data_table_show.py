from corai_plot import APlot

from src.estim_hawkes.estim_hawkes import Estim_hawkes
from root_dir import linker_path_to_result_file

path_result = linker_path_to_result_file(["result_hawkes_test", "test.csv"])
path_plot = linker_path_to_result_file(["result_hawkes_test", ''])

path_result = linker_path_to_result_file(["multithread_hawkes", "all_results_combined.csv"])
estim_hp = Estim_hawkes.from_csv(path_result)
df = estim_hp.df.loc[:, ['value', 'n', 'm', 'parameter']]
df = df[df.loc[:, 'parameter'] == 'nu']
APlot.show_plot()
