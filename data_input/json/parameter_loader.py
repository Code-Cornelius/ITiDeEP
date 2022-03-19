from data_input.json import json_loader as constant
from data_input.json.json_loader import json_parameter_loader
from run_simulations.setup_parameters import setup_parameters
from src.hawkes.hawkes_process import Hawkes_process
from src.hawkes.kernel import *
from src.utilities.fct_type_process import function_parameters_hawkes


def fetch_param_json_loader_simulation(flagprint=True, str_config=None):
    """ sets the seed and fetch all parameters """
    if str_config is not None:
        (SEED, UNDERLYING_FUNCTION_NUMBER, NB_POINTS_TT, T_MAX, KERNEL_DIVIDER
         , NB_DIFF_TIME_ESTIM, DIM, STYL, NB_SIMUL, L_PARAM, R_PARAM, h_PARAM
         , l_PARAM, CONSIDERED_PARAM, ALL_KERNELS_DRAWN, TYPE_ANALYSIS
         , NUMBER_OF_BREAKPOINTS, MODEL, MIN_SIZE, WIDTH) = json_parameter_loader(str_config)
        STR_CONFIG = str_config

    else:
        # the trick here is that when one does not give str_config, we use the constant defined in the file.
        # these constants are defined by calling json_parameter_loader without parameters.
        STR_CONFIG = constant.STR_CONFIG
        NB_SIMUL = constant.NB_SIMUL
        SEED = constant.SEED
        UNDERLYING_FUNCTION_NUMBER = constant.UNDERLYING_FUNCTION_NUMBER
        T_MAX = constant.T_MAX
        KERNEL_DIVIDER = constant.KERNEL_DIVIDER
        NB_DIFF_TIME_ESTIM = constant.NB_DIFF_TIME_ESTIM
        DIM = constant.DIM
        STYL = constant.STYL
        NB_POINTS_TT = constant.NB_POINTS_TT

    np.random.seed(SEED)

    (KERNEL, half_width_kernel, id_hp, times_estimation) = t_max_parameters(DIM, KERNEL_DIVIDER, NB_DIFF_TIME_ESTIM,
                                                                            NB_POINTS_TT, SEED,
                                                                            STYL, T_MAX, UNDERLYING_FUNCTION_NUMBER)

    if flagprint:
        print("Kernel: ", KERNEL.name)
        print("\n")

    parameters, t0, time_batch = setup_parameters(SEED, DIM, STYL)
    # time_batch corresponds to how much time required for 50 jumps

    if flagprint:
        print("Parameters of the process: ")
        print(f"nu: {parameters[0]}")
        print(f"alpha: {parameters[1]}")
        print(f"beta: {parameters[2]}\n")
        print_parameters_before_main(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
                                     NB_DIFF_TIME_ESTIM, DIM, STYL)

    fct_parameters, true_breakpoints = function_parameters_hawkes(UNDERLYING_FUNCTION_NUMBER, parameters)

    return (STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
            NB_DIFF_TIME_ESTIM, DIM, STYL, NB_POINTS_TT, id_hp, parameters, t0, time_batch,
            fct_parameters, true_breakpoints, KERNEL, times_estimation, half_width_kernel)


def t_max_parameters(DIM, KERNEL_DIVIDER, NB_DIFF_TIME_ESTIM, NB_POINTS_TT, SEED, STYL, T_MAX,
                     UNDERLYING_FUNCTION_NUMBER):
    #### T max:
    id_hp = {'seed': SEED, 'styl': STYL, 'dim': DIM, 'type evol': UNDERLYING_FUNCTION_NUMBER,
             'time burn-in': Hawkes_process.TIME_BURN_IN,
             'T max': T_MAX, 'nb points tt': NB_POINTS_TT}
    # KERNEL
    width_kernel = 1 / KERNEL_DIVIDER * T_MAX
    half_width_kernel = width_kernel / 2.
    KERNEL = Kernel(fct_kernel=fct_biweight, name=f"Bi-weight {width_kernel:.1f} width",
                    a=-half_width_kernel, b=half_width_kernel)
    # TIME ESTIMATION
    low_prcnt_tmax = 0.05 * T_MAX
    up_prcnt_tmax = 0.95 * T_MAX
    times_estimation = np.linspace(low_prcnt_tmax, up_prcnt_tmax, NB_DIFF_TIME_ESTIM)
    return KERNEL, half_width_kernel, id_hp, times_estimation


def print_parameters_before_main(STR_CONFIG, NB_SIMUL, SEED, UNDERLYING_FUNCTION_NUMBER, T_MAX, KERNEL_DIVIDER,
                                 NB_DIFF_TIME_ESTIM, DIM, STYL):
    print(f"STR CONFIG: {STR_CONFIG}")
    print(f"NB_SIMUL: {NB_SIMUL}")
    print(f"SEED: {SEED}")
    print(f"UNDERLYING_FUNCTION_NUMBER: {UNDERLYING_FUNCTION_NUMBER}")
    print(f"T_MAX: {T_MAX}")
    print(f"KERNEL_DIVIDER: {KERNEL_DIVIDER}")
    print(f"NB_DIFF_TIME_ESTIM: {NB_DIFF_TIME_ESTIM}")
    print(f"DIM: {DIM}")
    print(f"STYL: {STYL}")


def fetch_param_json_loader_itideep(flagprint=True, str_config=None):
    if str_config is not None:
        (SEED, UNDERLYING_FUNCTION_NUMBER, NB_POINTS_TT, T_MAX, KERNEL_DIVIDER,
         NB_DIFF_TIME_ESTIM, DIM, STYL, NB_SIMUL, L, R, h,
         l, CONSIDERED_PARAM, ALL_KERNELS_DRAWN, TYPE_ANALYSIS,
         NUMBER_OF_BREAKPOINTS, MODEL, MIN_SIZE, WIDTH) = json_parameter_loader(str_config)
    else:
        L = constant.L_PARAM
        R = constant.R_PARAM
        h = constant.h_PARAM
        l = constant.l_PARAM
        CONSIDERED_PARAM = constant.CONSIDERED_PARAM
        ALL_KERNELS_DRAWN = constant.ALL_KERNELS_DRAWN

        TYPE_ANALYSIS = constant.TYPE_ANALYSIS
        NUMBER_OF_BREAKPOINTS = constant.NUMBER_OF_BREAKPOINTS
        MODEL = constant.MODEL
        MIN_SIZE = constant.MIN_SIZE
        WIDTH = constant.WIDTH

    return (L, R, h, l, CONSIDERED_PARAM, ALL_KERNELS_DRAWN, TYPE_ANALYSIS,
            NUMBER_OF_BREAKPOINTS, MODEL, MIN_SIZE, WIDTH)
