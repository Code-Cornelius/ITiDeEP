import sys

from corai_util.tools.src.function_file import is_empty_file
from root_dir import linker_path_to_result_file
from src.estim_hawkes.estim_hawkes import Estim_hawkes

# STR_CONFIG = str(sys.argv[1])
# TEMP_STR = str(sys.argv[2])

# manual config
STR_CONFIG = "multidim_mountain"
TEMP_STR = 2

path_result_directory = linker_path_to_result_file(["euler_hawkes", f"{STR_CONFIG}_res_{TEMP_STR}", "data"])

assert not is_empty_file(path_result_directory), \
    f"file must contain some data. Directory {path_result_directory} is empty."

list_estim_hp = Estim_hawkes.folder_csv2list_estim(path_result_directory)
estim_hp = Estim_hawkes.merge(list_estim_hp)
path_super_result = linker_path_to_result_file(
    ["euler_hawkes", f"{STR_CONFIG}_res_{TEMP_STR}", "data_together", f"results_together.csv"])
estim_hp.to_csv(path_super_result)

# delete the old estimators.
# remove_files_from_dir(path_result_directory, "result_", 'csv')
