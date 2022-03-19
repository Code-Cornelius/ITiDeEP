import json

from root_dir import linker_path_to_data_file

# STR_CONFIG = "time_dep_config"
# STR_CONFIG = "multidim_mountain"
# STR_CONFIG = "config_jump"
STR_CONFIG = "MSE"


# STR_CONFIG = "test_config_multidim"

#### examples of configs
#  quick_config
#  test_config
#  test_config_multidim
#  multidim_mountain
#  config_jump
#  config_sin
#  MSE

def json_parameter_loader(STR_CONFIG=STR_CONFIG):
    """ from name of config to the parameters. """
    # creating paths to json
    path_to_adapt = linker_path_to_data_file(['json', 'param_adapt_estim.json'])
    path_to_change_point = linker_path_to_data_file(['json', 'param_change_point_analys.json'])
    path_to_simul = linker_path_to_data_file(['json', 'param_simul.json'])

    # loading the files
    with open(path_to_adapt) as file:
        json_param_adapt = json.load(file)
    with open(path_to_change_point) as file:
        json_param_change_point = json.load(file)
    with open(path_to_simul) as file:
        json_param_simul = json.load(file)

    # fetching the config we are interested in
    the_json_parameters_simulation = json_param_simul[STR_CONFIG]
    the_json_parameters_adaptive_simulation = json_param_adapt[STR_CONFIG]
    the_json_parameters_change_point = json_param_change_point[STR_CONFIG]

    SEED = the_json_parameters_simulation["seed"]
    UNDERLYING_FUNCTION_NUMBER = the_json_parameters_simulation["function"]
    NB_POINTS_TT = int(the_json_parameters_simulation["nb points tt"])
    T_MAX = the_json_parameters_simulation["Tmax"]
    KERNEL_DIVIDER = the_json_parameters_simulation["kernel_div"]
    NB_DIFF_TIME_ESTIM = int(the_json_parameters_simulation["nb_diff_time_estim"])
    DIM = the_json_parameters_simulation["dim"]
    STYL = the_json_parameters_simulation["styl"]
    NB_SIMUL = int(the_json_parameters_simulation["nb_simul"])

    L_PARAM = the_json_parameters_adaptive_simulation["L"]
    R_PARAM = the_json_parameters_adaptive_simulation["R"]
    h_PARAM = the_json_parameters_adaptive_simulation["h"]
    l_PARAM = the_json_parameters_adaptive_simulation["l"]
    CONSIDERED_PARAM = the_json_parameters_adaptive_simulation["considered_parameters"]
    ALL_KERNELS_DRAWN = the_json_parameters_adaptive_simulation["all_kernels_drawn"]

    TYPE_ANALYSIS = the_json_parameters_change_point["type_analysis"]
    NUMBER_OF_BREAKPOINTS = the_json_parameters_change_point["number_of_breakpoints"]
    MODEL = the_json_parameters_change_point["model"]
    MIN_SIZE = the_json_parameters_change_point["min_size"]
    WIDTH = the_json_parameters_change_point["width"]
    return (SEED, UNDERLYING_FUNCTION_NUMBER, NB_POINTS_TT, T_MAX, KERNEL_DIVIDER,
            NB_DIFF_TIME_ESTIM, DIM, STYL, NB_SIMUL, L_PARAM, R_PARAM, h_PARAM,
            l_PARAM, CONSIDERED_PARAM, ALL_KERNELS_DRAWN, TYPE_ANALYSIS,
            NUMBER_OF_BREAKPOINTS, MODEL, MIN_SIZE, WIDTH)


(SEED, UNDERLYING_FUNCTION_NUMBER, NB_POINTS_TT, T_MAX, KERNEL_DIVIDER,
 NB_DIFF_TIME_ESTIM, DIM, STYL, NB_SIMUL, L_PARAM, R_PARAM, h_PARAM,
 l_PARAM, CONSIDERED_PARAM, ALL_KERNELS_DRAWN, TYPE_ANALYSIS,
 NUMBER_OF_BREAKPOINTS, MODEL, MIN_SIZE, WIDTH) = json_parameter_loader()
