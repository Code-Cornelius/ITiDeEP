import math
from functools import partial

import numpy as np
from corai_error import Error_not_allowed_input

from src.utilities.general import multi_list_generator

TAU = 2 * math.pi

"""
Information about the functions:

functions that return always a np array.
if creating new function, make sure that the behaviour for values bigger than T_max is ok.
    reason : Sometimes, required the function for values slightly bigger than T_max (for the last event).
"""

# they are all vectorised function returning np arrays.

def constant_parameter(tt, constant, *args, **kwargs):
    tt = np.array(tt, ndmin=1)  # ndmin enforces 0dim arrays to be at least 1dim.
    return constant * np.ones(tt.size)
    # we use size instead of length because len does not work on zero dim vectors.
    # this means we do not accept arrays with more strictly than 1 dimensions! then size is the total nb of elements.


def linear_growth(tt, a, b, length_time_interval, time_burn_in):
    # ax + b
    # a is already divided by t_max, so just put of how much you want to grow
    tt = np.array(tt, ndmin=1)  # ndmin enforces 0dim arrays to be at least 1dim.
    return a / (length_time_interval + time_burn_in) * tt + b


def one_jump(tt, when_jump, original_value, new_value, length_time_interval, time_burn_in):
    # when_jump should be a %
    tt = np.array(tt, ndmin=1)  # ndmin enforces 0dim arrays to be at least 1dim.
    return original_value + new_value * np.heaviside(tt - time_burn_in - length_time_interval * when_jump, 1)


def moutain_jump(tt, when_jump, a, b, base_value, length_time_interval, time_burn_in):
    # when_jump should be a %
    # ax+b until the when_jump, where it comes down to base value.
    # we use size instead of length because len does not work on zero dim vectors.
    # this means we do not accept arrays with more strictly than 1 dimensions! then size is the total nb of elements.
    tt = np.array(tt, ndmin=1)  # ndmin enforces 0dim arrays to be at least 1dim.
    indices_condt = (tt < when_jump * length_time_interval + time_burn_in)
    ans = np.zeros(tt.size)
    ans[indices_condt] = linear_growth(tt, a, b, length_time_interval, time_burn_in)[indices_condt]
    ans[~indices_condt] = base_value
    return ans


def periodic_stop(tt, length_time_interval, a, base_value, time_burn_in):
    # looks like 3.5 period and then stops at base value.
    # we use size instead of length because len does not work on zero dim vectors.
    # this means we do not accept arrays with more strictly than 1 dimensions! then size is the total nb of elements.
    speed = 1.75
    where_stops = 0.75
    where_stops = 1.  # no stop for the time being of the simulations.
    tt = np.array(tt, ndmin=1)  # ndmin enforces 0dim arrays to be at least 1dim.
    rescaled_xx = tt / (length_time_interval + time_burn_in) * TAU

    ans = np.zeros(tt.size)
    indices_condt = (rescaled_xx < TAU * where_stops)

    cos_val = np.cos(rescaled_xx[indices_condt] * speed / where_stops)

    ans[indices_condt] = base_value + a * cos_val * cos_val
    ans[~indices_condt] = base_value  # contrary of the indices
    return ans


# WIP why are bkpts in lists?
def function_parameters_hawkes(switch, parameters):
    # switch:
    # 0,1,21,22,3,41 or 42
    # parameters format: [numpy 1D arr, numpy 2D arr, numpy 2D arr]
    nan = np.NaN
    nu, ALPHA, BETA = parameters
    M = len(nu)
    parameter_functions = multi_list_generator(M)

    # CONSTANT
    if switch == 0:
        breakpoint_nu = nan
        breakpoint_alpha = nan
        breakpoint_beta = nan
        true_breakpoints = {}
        for i in range(M):
            # reason why we use two scopes:
            # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
            parameter_functions[0][i] = partial(lambda tt, length_time_interval, time_burn_in, index_1:
                                                constant_parameter(tt, constant=nu[i]), index_1=i)
            true_breakpoints[("nu", i, 0)] = [breakpoint_nu]

            for j in range(M):
                parameter_functions[1][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    constant_parameter(tt=tt, constant=ALPHA[index_1, index_2]), index_1=i, index_2=j)
                parameter_functions[2][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    constant_parameter(tt=tt, constant=BETA[index_1, index_2],
                                       length_time_interval=length_time_interval,
                                       time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                true_breakpoints[("alpha", i, j)] = [breakpoint_alpha]
                true_breakpoints[("beta", i, j)] = [breakpoint_beta]

    # linear growth
    elif switch == 1:
        breakpoint_nu = nan
        breakpoint_alpha = nan
        breakpoint_beta = nan
        true_breakpoints = {}
        for i in range(M):
            parameter_functions[0][i] = partial(lambda tt, length_time_interval, time_burn_in, index_1:
                                                linear_growth(tt, 1.3 * nu[index_1], nu[index_1] / 2,
                                                              length_time_interval, time_burn_in=time_burn_in),
                                                index_1=i)
            true_breakpoints[("nu", i, 0)] = [breakpoint_nu]
            for j in range(M):
                parameter_functions[1][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    linear_growth(tt, BETA[index_1, index_2] * 0.2 - ALPHA[index_1, index_2] * 2 / 5,
                                  ALPHA[index_1, index_2], length_time_interval,
                                  time_burn_in=time_burn_in),
                    index_1=i, index_2=j)  # it goes up to BETA 50%

                parameter_functions[2][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    constant_parameter(tt=tt, constant=BETA[index_1, index_2],
                                       length_time_interval=length_time_interval, time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                true_breakpoints[("alpha", i, j)] = [breakpoint_alpha]
                true_breakpoints[("beta", i, j)] = [breakpoint_beta]

    # one jump
    elif switch == 21:
        breakpoint_nu = 0.7
        breakpoint_alpha = 0.4
        breakpoint_beta = nan
        true_breakpoints = {}
        for i in range(M):
            parameter_functions[0][i] = partial(lambda tt, length_time_interval, time_burn_in, index_1:
                                                one_jump(tt, breakpoint_nu, nu[index_1], 1.2 * nu[index_1],
                                                         length_time_interval, time_burn_in=time_burn_in), index_1=i)
            true_breakpoints[("nu", i, 0)] = [breakpoint_nu]
            for j in range(M):
                parameter_functions[1][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    one_jump(tt, breakpoint_alpha,
                             ALPHA[index_1, index_2],
                             -0.2 * ALPHA[index_1, index_2], length_time_interval,
                             time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                parameter_functions[2][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    constant_parameter(tt=tt, constant=BETA[index_1, index_2],
                                       length_time_interval=length_time_interval,
                                       time_burn_in=time_burn_in), index_1=i, index_2=j)
                true_breakpoints[("alpha", i, j)] = [breakpoint_alpha]
                true_breakpoints[("beta", i, j)] = [breakpoint_beta]


    # one jump
    # this is the case jump at the same times.
    elif switch == 22:
        breakpoint = 0.6
        no_breakpoint = nan
        true_breakpoints = {}
        for i in range(M):
            parameter_functions[0][i] = partial(lambda tt, length_time_interval, time_burn_in, index_1:
                                                one_jump(tt, breakpoint, nu[index_1],
                                                         0.3 * nu[index_1], length_time_interval,
                                                         time_burn_in=time_burn_in), index_1=i)
            true_breakpoints[("nu", i, 0)] = [breakpoint]
            for j in range(M):
                parameter_functions[1][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    one_jump(tt, breakpoint, ALPHA[index_1, index_2],
                             0.5 * ALPHA[index_1, index_2], length_time_interval, time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                parameter_functions[2][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    constant_parameter(tt=tt, constant=BETA[index_1, index_2],
                                       length_time_interval=length_time_interval,
                                       time_burn_in=time_burn_in), index_1=i, index_2=j)
                true_breakpoints[("alpha", i, j)] = [breakpoint]
                true_breakpoints[("beta", i, j)] = [no_breakpoint]

    # mountain jump
    elif switch == 3:
        breakpoint_nu = 0.7
        breakpoint_alpha = 0.7
        breakpoint_beta = nan
        true_breakpoints = {}
        for i in range(M):
            parameter_functions[0][i] = partial(lambda tt, length_time_interval, time_burn_in, index_1:
                                                moutain_jump(tt, when_jump=breakpoint_nu,
                                                             a=nu[index_1], b=nu[index_1],
                                                             base_value=nu[index_1] / 1.3,
                                                             length_time_interval=length_time_interval,
                                                             time_burn_in=time_burn_in),
                                                index_1=i)
            true_breakpoints[("nu", i, 0)] = [breakpoint_nu]

            for j in range(M):
                parameter_functions[1][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    moutain_jump(tt, when_jump=breakpoint_alpha,
                                 a=(BETA[index_1, index_2] * 0.4 - ALPHA[index_1, index_2]),
                                 b=ALPHA[index_1, index_2],
                                 base_value=ALPHA[index_1, index_2] / 1.3,
                                 length_time_interval=length_time_interval,
                                 time_burn_in=time_burn_in), index_1=i, index_2=j)
                parameter_functions[2][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    constant_parameter(tt=tt, constant=BETA[index_1, index_2],
                                       length_time_interval=length_time_interval,
                                       time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                true_breakpoints[("alpha", i, j)] = [breakpoint_alpha]
                true_breakpoints[("beta", i, j)] = [breakpoint_beta]

    # periodic stop
    elif switch == 41:  # for dim 1
        breakpoint_nu = 0.7
        breakpoint_alpha = 0.7
        breakpoint_beta = nan
        true_breakpoints = {}
        for i in range(M):
            parameter_functions[0][i] = partial(lambda tt, length_time_interval, time_burn_in, index_1:
                                                periodic_stop(tt, length_time_interval,
                                                              nu[index_1] / 3, 0.4,
                                                              time_burn_in=time_burn_in), index_1=i)
            true_breakpoints[("nu", i, 0)] = [breakpoint_nu]
            for j in range(M):
                parameter_functions[1][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    periodic_stop(tt, length_time_interval,
                                  BETA[index_1, index_2] * 0.2 - ALPHA[index_1, index_2] * 0.4,
                                  ALPHA[index_1, index_2] * 0.8, time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                parameter_functions[2][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    constant_parameter(tt=tt, constant=BETA[index_1, index_2],
                                       length_time_interval=length_time_interval,
                                       time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                true_breakpoints[("alpha", i, j)] = [breakpoint_alpha]
                true_breakpoints[("beta", i, j)] = [breakpoint_beta]

    elif switch == 42:  # for dim2
        breakpoint_nu = 0.7
        breakpoint_alpha = 0.7
        breakpoint_beta = nan
        true_breakpoints = {}
        for i in range(M):
            parameter_functions[0][i] = partial(lambda tt, length_time_interval, time_burn_in, index_1:
                                                periodic_stop(tt, length_time_interval, nu[index_1] / 3, 0.25,
                                                              time_burn_in=time_burn_in), index_1=i)
            true_breakpoints[("nu", i, 0)] = [breakpoint_nu]
            for j in range(M):
                parameter_functions[1][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    periodic_stop(tt, length_time_interval,
                                  BETA[index_1, index_2] * 0.3 - ALPHA[index_1, index_2] * 2 / 5,
                                  ALPHA[index_1, index_2] * 0.5, time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                parameter_functions[2][i][j] = partial(
                    lambda tt, length_time_interval, time_burn_in, index_1, index_2:
                    constant_parameter(tt=tt, constant=BETA[index_1, index_2],
                                       length_time_interval=length_time_interval,
                                       time_burn_in=time_burn_in),
                    index_1=i, index_2=j)
                true_breakpoints[("alpha", i, j)] = [breakpoint_alpha]
                true_breakpoints[("beta", i, j)] = [breakpoint_beta]
    else:
        raise Error_not_allowed_input("Problem with given switch.")

    return parameter_functions, true_breakpoints
