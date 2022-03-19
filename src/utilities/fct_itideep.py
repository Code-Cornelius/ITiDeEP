# normal libraries
import math  # quick math functions

import numpy as np
# my libraries
from corai_error import Error_not_allowed_input
from corai_util.tools.src.function_iterable import is_np_arr_constant
from scipy.stats.mstats import gmean

# other files
from src.hawkes.kernel import Kernel, fct_biweight

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FUNCTION_CHOICE = "SIN"
# FUNCTION_CHOICE = "JUMP"


def special_sin(value_at_each_time, L=0.02, R=0.98, h=2.5, l=0.2 / 2, silent=True):
    """

    Args:
        value_at_each_time: vector length m with the value to be considered together.
        L:  left wing
        R:  right wing
        h:  bigger rescale
        l:  smaller rescale
        silent: verbose

    Returns: a vector of length m of the rescaled values.

    """
    # compute the geometric mean from our estimator.

    # G is a function of the product of elements, we check that no element is zero.
    # case when the geometric mean equals 0. It is an issue here because we divide by G.
    # when G = 0, we return scaling factors all equal to 0.01 such that the scaling factor widens all kernels.
    if (value_at_each_time == 0).any():
        return np.full(len(value_at_each_time), 0.01)

    G = gmean(value_at_each_time)
    L_quant = np.quantile(value_at_each_time, L)
    R_quant = np.quantile(value_at_each_time, R)


    if not np.all((L_quant < G) & (G < R_quant)):
        raise Error_not_allowed_input("cdt: L < G < R. L: {}, R: {}, G: {}.".format(L_quant, G, R_quant))

    if not silent:
        print("G : ", G)
    if not silent:
        print("Left boundary : ", L_quant)
    if not silent:
        print("Right boundary : ", R_quant)

    xx = value_at_each_time - G
    ans = 0
    scaling1 = math.pi / (G - L_quant)
    scaling2 = math.pi / (R_quant - G)

    # fixing extreme values to final value h. They correspond to math.pi.
    # I also need the scaling by +h/2 given by math.pi

    # xx2 and xx3 are the cosinus, but they are different cosinus.
    # So I fix them where I don't want them to move at 0 and then I can add the two functions.
    my_xx2 = np.where((xx * scaling1 > -math.pi) & (xx * scaling1 < 0),
                      xx * scaling1, math.pi)  # left
    my_xx3 = np.where((xx * scaling2 > 0) & (xx * scaling2 < math.pi),
                      xx * scaling2, math.pi)  # right
    ans += - (h - l) / 2 * np.cos(my_xx2)
    ans += - (h - l) / 2 * np.cos(my_xx3)

    ans += l  # avoid infinite width kernel, with a minimal value.
    return ans


def jump_rescale(value_at_each_time, L=0.02, R=0.98, h=2.5, l=0.2 / 2, silent=True):
    """ same as special_sin but reversed."""
    if (value_at_each_time == 0).any():
        return np.full(len(value_at_each_time), 0.01)

    G = gmean(value_at_each_time)
    L_quant = np.quantile(value_at_each_time, L)
    R_quant = np.quantile(value_at_each_time, R)

    if not silent:
        print("G : ", G)
    if not silent:
        print("Left boundary : ", L_quant)
    if not silent:
        print("Right boundary : ", R_quant)

    xx = value_at_each_time - G
    ans = 0
    scaling1 = 1 / (G - L_quant)
    scaling2 = 1 / (R_quant - G)

    SPEED = 0.8  # between 0 and 1.

    left_xx_cdt = (xx * scaling1 > - 1) & (xx * scaling1 < - SPEED)
    middle_xx_cdt = (xx * scaling2 < SPEED) & (xx * scaling1 > -SPEED)
    right_xx_cdt = (xx * scaling2 < 1) & (xx * scaling2 > SPEED)
    my_xx2 = np.where(left_xx_cdt,
                      (xx * scaling1 + 1) * (h - l) / (1 - SPEED), 0)  # left
    my_xx3 = np.where(right_xx_cdt,
                      -(h - l) * (xx * scaling2 - 1) / (1 - SPEED), 0)  # right
    my_xx4middle = np.where(middle_xx_cdt, (h - l), 0)
    ans += my_xx2 + my_xx3 + my_xx4middle  # (h - l) / 2 *

    ans += l  # avoid infinite width kernel, with a minimal value.

    # point_up_left =   -1. / scaling1
    # point_plateau_left =  - (R_quant - L_quant) / 4
    # point_plateau_right =  + (R_quant - L_quant) / 4
    # print(point_up_left)
    # print(point_plateau_left)
    # print(point_plateau_right)
    # print(G)
    # point_down_right =  1./ scaling2
    #
    # low_left_ind = (xx  > point_up_left)
    # low_right_ind = (xx  < point_down_right)
    # high_middle_ind = (xx > point_plateau_left) & (xx < point_plateau_right)
    # increasing_left_ind = (xx < point_plateau_left) & (xx > point_up_left)
    # increasing_right_ind = (xx < point_down_right) & (xx > point_plateau_right)
    #
    #
    # ans = np.zeros(len(value_at_each_time)) + l
    # ans[low_left_ind]        += 0
    # ans[low_right_ind]       += 0
    # ans[high_middle_ind]     += 42# h-l
    # ans[increasing_left_ind] +=  (h-l)  / (point_plateau_left - point_up_left) * xx[increasing_left_ind]
    # ans[increasing_right_ind]+=  h - (h - l) / (point_plateau_left - point_up_left) * xx[increasing_right_ind]

    return ans


def AKDE_scaling(times, G=10., gamma=0.5):
    ans = times.copy()
    for i in range(len(times)):
        xx = times[i]
        # print("before rescale", xx)
        rescaled_xx = np.power(xx / G, -gamma)
        # print("rescaled", rescaled_xx)
        ans[i] = rescaled_xx
    return ans


def rescale_min_max(arr):
    """
    From [min, max] to [0,2].
    The mean is computed by excluding nans and
    ALso, if any value is NaN in the vector, we replace it by the mean.
    Args:
        arr:

    Returns:

    """
    the_max = max(arr)
    the_min = min(arr)
    the_mean = np.nanmean(arr)
    ans = [(arr[i] - the_mean) / (the_max - the_min) + 1 for i in range(len(arr))]
    np.array(ans)[np.isnan(ans)] = the_mean  # replace where nan by mean.
    return ans


def compute_scaling_itideep(times, first_estimate, considered_param, L, R, h, l, tol=0., silent=True):
    """
    Generate the scaling parameters for the kernels given the first estimation of the parameters.
    Args:
        times: times of the estimation, corresponding to each column of first_estimate.
        first_estimate:  multi-dim matrix: (times,parameter), where parameter is nu,alpha,beta.
        considered_param: the parameter we use for rescaling, e.g.: ['nu','alpha','beta']
        L: parameter special sinus.
        R: parameter special sinus.
        h: parameter special sinus.
        l: parameter special sinus.
        tol:
        silent:

    Returns:


    """
    assert len(times) == len(first_estimate), \
        "times and first_estimate should be the same length: {}, {}".format(len(times), len(first_estimate))

    M = len(first_estimate[0][0])  # the dimension of the data.
    dim_estimation = 2 * M * M + M
    include_estimation = [False] * dim_estimation
    reshape_estimator_time_together = [[] for _ in range(dim_estimation)]  # creates a matrix(dim_estimation * 0).

    # in each reshape_estimator_time_together are stored for one parameter's estimation the whole time sequence of estimation.
    # the order is the nus, the alphas, the betas.
    # we separate the first estimate into each parameters vector of results.
    # todo replace this by a flattening procedure.
    for k in range(len(times)):  # each line are the times
        for i in range(M):  # parameters stored in vectors, row major.
            reshape_estimator_time_together[i].append(first_estimate[k][0][i])  # nu
            for j in range(M):
                reshape_estimator_time_together[M + i * M + j].append(first_estimate[k][1][i][j])  # alpha
                reshape_estimator_time_together[M + M * M + i * M + j].append(first_estimate[k][2][i][j])  # beta

    # vector that says whether a dimension is included from the previous vector.
    # by default not included.
    # computation of reshape_estimator_time_together required.
    for i in range(dim_estimation):
        # sets the parameters that we will consider for weights updating
        if i < M and 'nu' in considered_param:
            include_estimation[i] = True
        elif i < M + M * M and 'alpha' in considered_param:
            include_estimation[i] = True
        elif 'beta' in considered_param:
            include_estimation[i] = True
        # exclude included estimation that are constant.
        if include_estimation[i]:
            if is_np_arr_constant(reshape_estimator_time_together[i], tol):  # we don't keep the True
                include_estimation[i] = False
    if not silent:
        print("We include the following parameters in the norm composition:\n(nu,alpha,beta):\n", include_estimation)

    # rescaling
    reshape_estimator_filtered = []  # the values inside are rescaled in order to show how big/small they are.
    for i in range(dim_estimation):
        if include_estimation[i]:
            reshape_estimator_filtered.append(
                rescale_min_max(np.array(reshape_estimator_time_together[i]))
            )

    # norm_over_the_time is vector of norms. Each value inside it is for one time.
    # we normalise wrt the whole estimator vector.
    norm_over_the_time = np.zeros(len(times))
    for j in range(len(times)):
        norm_over_the_time[j] = np.linalg.norm(
            [reshape_estimator_filtered[i][j] for i in range(len(reshape_estimator_filtered))], 2)
    if not silent:
        print("vect  :", reshape_estimator_time_together)
        print("the norms ", norm_over_the_time)

    if FUNCTION_CHOICE == "JUMP":
        scaling_factors = jump_rescale(norm_over_the_time, L=L, R=R, h=h, l=l, silent=silent)
    else:
        scaling_factors = special_sin(norm_over_the_time, L=L, R=R, h=h, l=l, silent=silent)
    return scaling_factors


def creator_list_kernels(list_scalings, list_previous_half_width):
    # we want that both inputs list_scalings and list_previous_half_width are the same size
    # the kernel is taken as bi-weight.
    list_half_width = []
    list_of_kernels = []
    assert len(list_scalings) == len(list_previous_half_width), "Both lists need to be the same size."

    for half_width, scale in zip(list_previous_half_width, list_scalings):
        new_scaling = half_width / scale
        list_half_width.append(new_scaling)
        if FUNCTION_CHOICE == "JUMP":
            naming = f"Adaptive (jump special) Bi-weight, 1st width {2 * half_width:.1f}"
        else:
            naming = f"Adaptive Bi-weight, 1st width {2 * half_width:.1f}"

        list_of_kernels.append(Kernel(fct_biweight, a=-new_scaling, b=new_scaling, name=naming))
    return list_half_width, list_of_kernels


def creator_kernels_adaptive(first_estimator, times, considered_param, list_previous_half_width,
                             L, R, h, l, tol=0.1, silent=True):
    """

    Args:
        first_estimator: hawkes_estimator
        times:
        considered_param:
        list_previous_half_width:
        L:
        R:
        h:
        l:
        tol:
        silent:

    Returns:

    """
    # by looking at the previous estimation, we deduce the scaling
    # first_estimate_mean: should be looking like [ nu, alpha, beta ] * times
    first_estimate_mean = first_estimator.mean("time estimation")
    # convert the keys into int:
    first_estimate_mean_int_keys = {int(key): first_estimate_mean[key] for key in first_estimate_mean.keys()}
    first_estimate_mean_in_list = []
    # convert the dict into lists (helps having the keys in ascending order). Each list is one time. inside there is nu,alpha, beta
    for a_time in times:
        a_time = int(a_time)  # convert into int because that s a good enough approx
        first_estimate_mean_in_list.append(first_estimate_mean_int_keys[a_time])
    scalings_from_estimate = compute_scaling_itideep(times=times, first_estimate=first_estimate_mean_in_list,
                                                     considered_param=considered_param, tol=tol,
                                                     L=L, R=R, h=h, l=l, silent=silent)
    if not silent:
        print('the scaling : ', scalings_from_estimate)
    list_half_width, list_of_kernels = creator_list_kernels(list_scalings=scalings_from_estimate,
                                                            list_previous_half_width=list_previous_half_width)
    # list_half_width :  sequence of all the half width.
    # list_of_kernels :  sequence of all the new kernels.
    return list_half_width, list_of_kernels

# times = [0, 100, 200, 300, 400]
# first_estimate_mean = [
#     [[0.2], [[0.7]], [[2.8]]],
#     [[0.3], [[1.3]], [[3.0]]],
#     [[0.3], [[1.0]], [[3.2]]],
#     [[0.5], [[2.0]], [[3.0]]],
#     [[0.5], [[2.5]], [[4.0]]]
# ]
#
# first_estimate_reshape = [1., 4., 2., 3., 2.5]
# first_estimate_reshape = np.array(first_estimate_reshape)
# considered_param = ['nu', 'alpha', 'beta']
#
# list_of_means = []
# list_previous_half_width = [2000, 2000, 2000, 2000, 2000]
#
# L = 0.02
# R = 0.95
# h = 3
# l = 0.5
#
# print("###############-1")
# print(special_sin(first_estimate_reshape, L, R, h, l, silent=False))
# print("###############-2")
# print(compute_scaling_hitdep(times, first_estimate_mean, considered_param, L, R, h, l, silent=False))
# print("###############-3")
# creator_kernels_adaptive(first_estimate_mean, times, considered_param, list_previous_half_width,
#                          L, R, h, l, tol=0.1, silent=False)
