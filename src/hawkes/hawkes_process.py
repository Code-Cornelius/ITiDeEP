# normal libraries
import math

import matplotlib.pyplot as plt
import numpy as np
# priv libraries
from corai_error import Error_not_allowed_input, Error_type_setter
from corai_plot import APlot
from corai_util.tools import function_iterable

# other files
from src.utilities.general import multi_list_generator

np.random.seed(124)

# section ######################################################################
#  #############################################################################
# fcts


# defaut kernel, useful for default argument.
INFINITY = float("inf")


def CDF_exp(x, LAMBDA):
    return - np.log(1 - x) / LAMBDA  # I inverse 1-x bc the uniform can be equal to 0, not defined in the log.


def CDF_LEE(U, lambda_value, delta):
    if U.item() > 1 - np.exp(- lambda_value / delta):
        return INFINITY
    else:
        return -1 / delta * np.log(1 + delta / lambda_value * np.log(1 - U))


def exp_kernel(alpha, beta, t):
    return alpha * np.exp(- beta * t)


def lewis_non_homo(max_time, actual_time, max_nu, fct_value, **kwargs):
    """ Method in order to get inter-arrivals times using lewis' thinning algorithm.

    Args:
        max_time: in order to avoid infinite loop.
        actual_time: current time, in order to update the parameter nu
        max_nu:  over the interval for thinning.
        fct_value: nu fct.
        **kwargs: for nu fct.

    Returns:

    """
    arrival_time = 0
    while actual_time + arrival_time < max_time:
        U = np.random.rand(1)
        arrival_time += CDF_exp(U, max_nu)
        D = np.random.rand(1)
        if D <= fct_value(actual_time + arrival_time, **kwargs) / max_nu:
            return arrival_time
    return INFINITY


def step_fun(tt, time_real):
    # At every index where the jumps occurs and onwards, +1 to the step-function.
    y = np.zeros(len(tt))
    for i in range(len(tt)):
        jumps = function_iterable.find_smallest_rank_leq_to_K(time_real, tt[i])
        y[i] = jumps
    return y


# section ######################################################################
#  #############################################################################
# class

class Hawkes_process:
    """
    Class where given parameters, represent the process as a function omega to continuous functions.
    Can generate path.

    Attributes:

    """
    TIME_BURN_IN = 200
    NB_POINTS_BURNED = 6000
    POINTS_BURNED = np.linspace(0, TIME_BURN_IN, NB_POINTS_BURNED)

    def __init__(self, the_update_functions):
        self.M = np.shape(the_update_functions[1])[1]

        self.the_update_functions = the_update_functions.copy()
        # without the copy, if I update the the_update_functions inside HP,
        # it also updates the the_update_functions outside of the object.
        self.alpha = self.the_update_functions[1]
        self.beta = self.the_update_functions[2]
        self.nu = self.the_update_functions[0]
        self.parameters_line = np.append(np.append(self.nu, np.ravel(self.alpha)), np.ravel(self.beta))

        # I plot the functions in order to know the evolution:
        self.plot_parameters_respect_to_time_hawkes()

    def __call__(self, t, T_max):
        nu, alpha, beta = multi_list_generator(self.M)
        for i in range(self.M):
            nu[i] = self.nu[i](t, T_max, Hawkes_process.TIME_BURN_IN)
            for j in range(self.M):
                alpha[i][j] = self.alpha[i][j](t, T_max, Hawkes_process.TIME_BURN_IN)
                beta[i][j] = self.beta[i][j](t, T_max, Hawkes_process.TIME_BURN_IN)

        return f'a Hawkes process, with parameters at time {t} : {nu}, {alpha}, {beta}'

    def __repr__(self):
        """Call the hawkes process and shows the parameters evaluated at 0."""
        return self.__call__(0, 1000)

    def plot_parameters_respect_to_time_hawkes(self):
        # print the underlying parameters of the process.
        aplot = APlot(how=(1, self.M))
        tt = np.linspace(0, 1, 1000)
        my_colors = plt.cm.rainbow(np.linspace(0, 1, 2 * self.M))
        for i_dim in range(self.M):
            xx_nu = self.nu[i_dim](tt, 1, 0)
            aplot.uni_plot(nb_ax=i_dim, yy=xx_nu, xx=tt,
                           dict_plot_param={"label": f"nu, {i_dim}", "color": "blue", "markersize": 0, "linewidth": 2})
            color = iter(my_colors)
            for j_dim in range(self.M):
                c1 = next(color)
                c2 = next(color)
                xx_alpha = self.alpha[i_dim][j_dim](tt, 1, 0)
                xx_beta = self.beta[i_dim][j_dim](tt, 1, 0)
                aplot.uni_plot(nb_ax=i_dim, yy=xx_alpha, xx=tt,
                               dict_plot_param={"label": f"alpha, {i_dim},{j_dim}.", "color": c1, "markersize": 0,
                                                "linewidth": 2})
                aplot.uni_plot(nb_ax=i_dim, yy=xx_beta, xx=tt,
                               dict_plot_param={"label": f"beta, {i_dim},{j_dim}.", "color": c2, "markersize": 0,
                                                "linewidth": 2},
                               dict_ax={
                                   'title': "Evolution of the parameters, time in % of total; dimension: {}".format(
                                       i_dim), 'xlabel': '', 'ylabel': ''})

            aplot.show_legend(i_dim)

    def simulation_Hawkes_exact_with_burn_in(self, tt, nb_of_sim=100000, plotFlag=True, silent=True):
        """

        Args:
            tt: we assume an interval of the form [0,T]. No impact on the simulation, only on the intensity plot.
            nb_of_sim: nb of jumps max before exit. 100 000 is just a safe guard
            plotFlag:  plotFlag  then draw the path of the simulation.
            silent:

        Returns:

        """
        # burn in is added in such way: we simulate over T + constant of burn in.
        # Then, we only give back [constant, T + constant].
        whole_tt = np.append(Hawkes_process.POINTS_BURNED, tt + Hawkes_process.TIME_BURN_IN)
        T_max = tt[-1]

        if not silent:
            print("Start of the simulation of the Hawkes process.")
        # alpha and beta same shape. nu a column vector with the initial intensities.
        if np.shape(self.alpha) != np.shape(self.beta):
            raise Error_not_allowed_input("Why are the the_update_functions not of the good shape ?")

        # empty vector for stocking the information (the times at which something happens):
        T_t = [[] for _ in range(self.M)]

        # where I evaluate the function of intensity for plotting
        intensity = np.zeros((self.M, len(whole_tt)))  # will be an empty list if plotflag false.
        last_jump = 0
        counter = 0

        last_print = -1  # : for the printing

        # For the evaluation, we stock the last lambda.
        # prev_lambd is the old intensity,
        # we have the small_lambd, corresponding to each component of the intensity.
        # We don't need to incorporate the burned point in it. It appears in the previous lambda.
        prev_lambd = np.zeros((self.M, self.M))
        small_lambd = np.zeros((self.M, self.M, len(whole_tt)))

        condition = True

        # I need the max value of nu for thinning simulation:
        # it is an array with the max in each dimension.
        max_nu = [np.max(self.nu[i](whole_tt, T_max, Hawkes_process.TIME_BURN_IN)) for i in range(self.M)]
        while condition:
            # aa is the matrix of the a_m^i.
            aa = np.zeros((self.M, self.M + 1))
            # first loop over the m_dims.
            # second loop over where from.
            for m_dims in range(self.M):
                for i_where_from in range(self.M + 1):
                    if i_where_from == 0:  # immigration
                        aa[m_dims, i_where_from] = lewis_non_homo(T_max + Hawkes_process.TIME_BURN_IN,
                                                                  last_jump,
                                                                  max_nu[m_dims],
                                                                  self.nu[m_dims],
                                                                  length_time_interval=T_max,
                                                                  time_burn_in=Hawkes_process.TIME_BURN_IN)
                    # cases where the other processes can have an impact.
                    # If intensity not big enough, we delete the impact and reduce comput. cost
                    elif prev_lambd[i_where_from - 1, m_dims] < 10e-10:
                        aa[m_dims, i_where_from] = INFINITY

                    # cases where it is big enough:
                    else:
                        U = np.random.rand(1)
                        # todo change function beta for time dep parameters
                        aa[m_dims, i_where_from] = \
                            CDF_LEE(U, prev_lambd[i_where_from - 1, m_dims],
                                    self.beta[i_where_from - 1][m_dims]
                                    (0, T_max, Hawkes_process.TIME_BURN_IN))
            # next_a_index indicates the dimension in which the jump happens.
            if self.M > 1:
                # it is tricky : first find where the min is (index) but it is flatten.
                # So I recover coordinates with unravel index.
                # I take [0] bc I only care about where the excitation comes from.
                next_a_index = np.unravel_index(np.argmin(aa, axis=None), aa.shape)[0]
            # otherwise, excitation always from 0 dim.
            else:
                next_a_index = 0
            next_a_value = np.amin(aa)  # min value.

            previous_jump = last_jump  # previous_jump is the one before.
            last_jump += next_a_value  # last jump is the time at which the current interesting jump happened.

            # I add the time iff I haven't reached the limit already.

            if T_max is not None:
                if last_jump < T_max + Hawkes_process.TIME_BURN_IN:
                    T_t[next_a_index].append(last_jump)  # wip very bad step terms of memory!

            # previous lambda gives the lambda for simulation.
            # small lambda is the lambda in every dimension for plotting.
            for ii in range(self.M):
                for jj in range(self.M):
                    if jj == next_a_index:
                        # todo change function beta
                        prev_lambd[jj, ii] = prev_lambd[jj, ii] * math.exp(
                            - self.beta[jj][ii](last_jump, T_max, Hawkes_process.TIME_BURN_IN) * next_a_value) + \
                                             self.alpha[jj][ii](last_jump, T_max, Hawkes_process.TIME_BURN_IN)
                    else:
                        # todo change function beta
                        prev_lambd[jj, ii] = prev_lambd[jj, ii] * \
                                             math.exp(- self.beta[jj][ii](last_jump, T_max,
                                                                          Hawkes_process.TIME_BURN_IN) * next_a_value)

            if plotFlag:
                # print("previous : ", previous_jump) #debug
                # print("last : ", last_jump) #debug
                first_index_time = function_iterable.find_smallest_rank_leq_to_K(whole_tt, previous_jump)
                for i_lin in range(self.M):
                    for j_col in range(self.M):
                        for i_tim in range(first_index_time, len(whole_tt)):
                            # this is when there is the jump. It means the time is exactly smaller but the next one bigger.
                            if whole_tt[i_tim - 1] <= last_jump < whole_tt[i_tim]:
                                # I filter the lines on which I add the jump.
                                # I add the jump to the process iff the value appears on the relevant line of the alpha.
                                if i_lin == next_a_index:
                                    # todo change function alpha beta
                                    small_lambd[i_lin, j_col, i_tim] = \
                                        self.alpha[i_lin][j_col](last_jump, T_max, Hawkes_process.TIME_BURN_IN) * \
                                        np.exp(- self.beta[i_lin][j_col](last_jump, T_max, Hawkes_process.TIME_BURN_IN)
                                               * (whole_tt[i_tim] - last_jump))
                                # since we are at the jump, one doesn't have to look further. break loop.
                                break
                            # the window of times I haven't updated.
                            # I am updating all the other times.
                            # todo change function beta
                            if previous_jump < whole_tt[i_tim] < last_jump:
                                small_lambd[i_lin, j_col,
                                            i_tim] += small_lambd[i_lin, j_col, i_tim - 1] \
                                                      * np.exp(- self.beta[i_lin][j_col](last_jump, T_max,
                                                                                         Hawkes_process.TIME_BURN_IN) * (
                                                                       whole_tt[i_tim] - whole_tt[i_tim - 1]))

            if nb_of_sim is not None:
                counter += 1
                if counter % 5000 == 0:
                    if not silent:
                        print(f"Jump {counter} out of total number of jumps {nb_of_sim}.")
                if not (counter < nb_of_sim - 1):
                    condition = False

            if T_max is not None:
                if not silent:
                    if round(last_jump, -1) % 1000 == 0 and round(last_jump, -1) != last_print:
                        last_print = round(last_jump, -1)
                        print(f"Time {round(last_jump, -1)} out of total time : {T_max}.")
                # IF YOU ARE TOO BIG IN TIME:                 # I add the burn in
                if not (last_jump < T_max + Hawkes_process.TIME_BURN_IN):
                    condition = False
        if plotFlag:
            for i_lin in range(self.M):
                for counter_times, i_tim in enumerate(whole_tt):
                    intensity[i_lin, counter_times] = self.nu[i_lin](i_tim, T_max, Hawkes_process.TIME_BURN_IN)
                    for j_from in range(self.M):
                        intensity[i_lin, counter_times] += small_lambd[j_from, i_lin, counter_times]

        if not silent:
            print("inside not shifted : ", T_t)

        # intensity bis is the truncated version of intensity, truncated wrt burn-in.
        intensity_bis = np.zeros((self.M, len(whole_tt) - Hawkes_process.NB_POINTS_BURNED))
        for i in range(len(T_t)):
            # find the times big enough.
            i_time = function_iterable.find_smallest_rank_leq_to_K(np.array(T_t[i]), Hawkes_process.TIME_BURN_IN)
            # shift the times
            T_t[i] = list(np.array(T_t[i][i_time:]) - Hawkes_process.TIME_BURN_IN)
            intensity_bis[i, :] = list(np.array(intensity[i][Hawkes_process.NB_POINTS_BURNED:]))
        return intensity_bis, T_t

    # section ######################################################################
    #  #############################################################################
    # getter/setters

    @property
    def alpha(self):
        return self._ALPHA

    @alpha.setter
    def alpha(self, new_ALPHA):
        if function_iterable.is_iterable(new_ALPHA) and \
                all([callable(new_ALPHA[i][j]) for i in range(self.M) for j in range(self.M)]):
            # check if the new parameters is a list and if all of the inputs are functions
            self._ALPHA = new_ALPHA
        else:
            raise Error_type_setter(f'Argument is not an function.')

    @property
    def beta(self):
        return self._BETA

    @beta.setter
    def beta(self, new_BETA):
        if function_iterable.is_iterable(new_BETA) and \
                all([callable(new_BETA[i][j]) for i in range(self.M) for j in range(self.M)]):
            # check if the new parameters is a list and if all of the inputs are functions
            self._BETA = new_BETA
        else:
            raise Error_type_setter(f'Argument is not an function.')

    @property
    def nu(self):
        return self._NU

    @nu.setter
    def nu(self, new_NU):
        if function_iterable.is_iterable(new_NU) and all([callable(new_NU[i]) for i in range(self.M)]):
            # check if the new parameters is a list and if all of the inputs are functions
            self._NU = new_NU
        else:
            raise Error_type_setter(f'Argument is not an function.')

    # section ######################################################################
    #  #############################################################################
    # plot:

    def plot_hawkes(self, tt, time_real, intensity, path_save_file=None):

        nu, alpha, beta = multi_list_generator(self.M)
        # store the values of the parameters as constants. Bc seen as constants, times given to fct does not matter.
        for i in range(self.M):
            self.nu[i](0, 1000, 0)
            nu[i] = self.nu[i](0, 1000, 0)[0]
            for j in range(self.M):
                alpha[i][j] = self.alpha[i][j](0, 1000, 0)[0]
                beta[i][j] = self.beta[i][j](0, 1000, 0)[0]

        # I need alpha and beta in order for me to plot them.
        shape_intensity = np.shape(intensity)
        plt.figure(figsize=(10, 5))
        x = tt
        # colors :
        color = iter(plt.cm.rainbow(np.linspace(0, 1, shape_intensity[0])))
        upper_ax = plt.subplot2grid((21, 21), (0, 0), rowspan=14, colspan=16)
        lower_ax = plt.subplot2grid((21, 21), (16, 0), rowspan=8, colspan=16)
        for i_dim in range(shape_intensity[0]):
            # the main
            c = next(color)
            y = intensity[i_dim, :]
            number_on_display = i_dim + 1
            label_plot = str(" dimension " + str(number_on_display))
            upper_ax.plot(x, y, 'o-', markersize=0.2, linewidth=0.4, label=label_plot, color=c)
            upper_ax.set_ylabel("Intensity : $\lambda (t)$")
            # the underlying
            y = 4 * i_dim + step_fun(x, np.array(time_real[i_dim]))
            lower_ax.plot(x, y, 'o-', markersize=0.5, linewidth=0.5, color=c)
            lower_ax.set_xlabel("Time")
            lower_ax.set_ylabel("Point Process : $N_t$")

        upper_ax.legend(loc='best')
        upper_ax.grid(True)
        lower_ax.grid(True)
        # Matrix plot :
        plt.subplot2grid((21, 21), (1, 16), rowspan=1, colspan=5)
        plt.text(0.5, 0, "$\\alpha$", fontsize=12, color='black')
        plt.axis('off')
        ax = plt.subplot2grid((21, 21), (3, 16), rowspan=5, colspan=5)
        im = plt.imshow(alpha, cmap="coolwarm")
        for (j, i), label in np.ndenumerate(alpha):
            ax.text(i, j, label.round(decimals=2), ha='center', va='center')
        plt.colorbar(im)
        plt.axis('off')

        plt.subplot2grid((21, 21), (9, 16), rowspan=1, colspan=5)
        plt.text(0.5, 0, "$\\beta$", fontsize=12, color='black')
        plt.axis('off')
        ax = plt.subplot2grid((21, 21), (10, 16), rowspan=5, colspan=5)
        im = plt.imshow(beta, cmap="coolwarm")
        for (j, i), label in np.ndenumerate(beta):
            ax.text(i, j, label.round(decimals=2), ha='center', va='center')
        plt.colorbar(im)
        plt.axis('off')

        plt.subplot2grid((21, 21), (19, 16), rowspan=1, colspan=5)
        plt.text(0.5, 0, "$\\nu = $ " + str(np.array(nu).round(decimals=2)), fontsize=11, color='black')
        plt.axis('off')

        if path_save_file is not None:
            string = path_save_file + ".png"
            plt.savefig(string, dpi=500)

        return
