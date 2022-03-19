# normal libraries
from inspect import signature  # used in the method eval of the class

import numpy as np
import scipy.stats  # functions of statistics
# other files
from corai_error import Error_type_setter
from scipy.integrate import simps

# my libraries

np.random.seed(124)


# section ######################################################################
#  #############################################################################
# some information


# -------------------------------------------------------------------------------------------------------
# list of the possible kernels:
# fct_top_hat
# fct_plain
# fct_truncnorm
# fct_biweight
#
#

# the functions are correct, they scale and shift the way it is supposed.
# However they are written in the following way : f_t(t_i) = K( t_i - t )

# example of kernels:
# list_of_kernels =
#           [Kernel(fct_top_hat, name="wide top hat", a=-450, b=450),
#            Kernel(fct_top_hat, name="normal top hat", a=-200, b=200),
#            Kernel(fct_truncnorm, name="wide truncnorm", a=-500, b=500, sigma=350),
#            Kernel(fct_truncnorm, name="normal truncnorm", a=-350, b=350, sigma=250)]
# -------------------------------------------------------------------------------------------------------
# the functions only work for positive time. If one input negative times, it messes up the orientation.


# section ######################################################################
#  #############################################################################
# class
class Kernel:
    # kernel is a functor, used for weighting some computations.

    # the evaluation gives back a list of np.array

    # the function should hand in the list of np.arrays non scaled.

    # the parameters of the function (to be called) are gathered before:
    #   the weights do not change inside the estimation process.

    # the name is for identification in plots
    def __init__(self, fct_kernel, name=' no name ', **kwargs):
        self.fct_kernel = fct_kernel
        self.name = name
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"Function is {repr(self._fct_kernel)} and name {self.name}."

    def __call__(self, T_t, eval_point, T_max, debug=False):
        # getting the length over each dimensions for the kernel.
        shape_T_t = [len(T_t[i]) for i in range(len(T_t))]  # recall each dim has different nb of jumps
        # ans is the kernel evaluated on the jumps
        ans = self._fct_kernel(T_t=T_t, eval_point=eval_point, shape_T_t=shape_T_t,
                               **{k: self.__dict__[k] for k in self.__dict__ if
                                  k in signature(self._fct_kernel).parameters})
        # ans is a list of np arrays. It is normalized such that it is a kernel.
        # then I want to scale every vector.
        # The total integral should be T_max, so I multiply by T_max

        # If it isn't fct plain, then I have to scale.
        if self._fct_kernel.__name__ != 'fct_plain':

            # I want to rescale the results for the kernels that are not covering seen part. For that reason,
            # I compute the integral of the kernel, and scale accordingly.
            tt_integral = [np.linspace(0, T_max, int(5E5))] # in a list to respect the format list of list of T_t.
            yy = self._fct_kernel(T_t=tt_integral, eval_point=eval_point, shape_T_t=[1],
                                  **{k: self.__dict__[k] for k in self.__dict__ if
                                     k in signature(self._fct_kernel).parameters})
            integral = simps(yy[0], tt_integral[0])
            # yy[0] bc function gives back a list of arrays.

            for i in range(len(shape_T_t)):
                ans[i] = ans[i] / integral * T_max
                # *= do not work correctly since the vectors are not the same type (int/float).
                # I also divide by the sum, the vector is normalized, however,
                # possibly we're on the edge and we need to take that into account.

        if debug:
            print(f"inside kernel debug, "
                  f"that's my integral : "
                  f"{np.sum(ans[0][:-1]) * T_max / (len(ans[0]) - 1)}. "
                  f"Name : {self.fct_kernel.__name__}.")
        return ans

    # section ######################################################################
    #  #############################################################################
    # getters setters
    @property
    def fct_kernel(self):
        return self._fct_kernel

    @fct_kernel.setter
    def fct_kernel(self, new_fct_kernel):
        self._fct_kernel = new_fct_kernel

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if isinstance(new_name, str):
            self._name = new_name
        else:
            raise Error_type_setter(f'Argument is not an string.')


# section ######################################################################
#  #############################################################################
# kernels' functions


def fct_top_hat(T_t, shape_T_t, eval_point, a=-200, b=200):
    output = []
    for i in range(len(shape_T_t)):
        vector = np.array(T_t[i])
        # -1 if x < 0, 0 if x==0, 1 if x > 0.
        output.append(1 / (2 * (b - a)) *
                      (np.sign(vector - eval_point - a) +
                       np.sign(b - vector + eval_point))
                      )
    return output


def fct_plain(T_t, shape_T_t, eval_point):
    # no scaling parameter, would be full to use scaling on plain.
    return [np.full(shape_T_t[i], 1) for i in range(len(shape_T_t))] # full of 1.


def fct_truncnorm(T_t, shape_T_t, eval_point, a=-300, b=300, sigma=200):
    output = []
    for i in range(len(shape_T_t)):
        output.append(scipy.stats.truncnorm.pdf(np.array(T_t[i]), a / sigma, b / sigma,
                                                loc=eval_point, scale=sigma))
    return output


def fct_truncnorm_test(T_t, shape_T_t, eval_point, a=-300, b=300, sigma=200):
    output = []
    i = 0  # for output[i] after, but there shouldn't be any problem.
    for i in range(len(shape_T_t)):
        output.append(2 * scipy.stats.truncnorm.pdf(np.array(T_t[i]), a / sigma, b / sigma,
                                                    loc=eval_point, scale=sigma))
    output[i][T_t[i] < eval_point] = 0
    return output


def fct_biweight(T_t, shape_T_t, eval_point, a=-300, b=300):
    #  if important, I can generalize biweight with function beta.
    #  Thus creating like 4 kernels with one function ( BETA(1), BETA(2)...)
    assert a == -b, "The kernel only accepts symmetrical bounds."
    output = []
    for i in range(len(shape_T_t)):
        xx = (np.array(T_t[i]) - (a + b) / 2 - eval_point) * 2 / (b - a)
        # the correct order is eval_point - T_t,
        # bc we evaluate at eval_point but translated by T_t,
        # if kernel not symmetric a != b, then we also need to translate by the mid of them.
        xx[(xx < -1) | (xx > 1)] = 1
        output.append(15 / 16 * np.power(1 - xx * xx, 2) * 2 / (b - a))
    return output


def fct_epa(T_t, shape_T_t, eval_point, a=-300, b=300):
    assert a == -b, "The kernel only accepts symmetrical bounds."
    output = []
    for i in range(len(shape_T_t)):
        xx = (np.array(T_t[i]) - (a + b) / 2 - eval_point) * 2 / (b - a)
        # the correct order is eval_point - T_t,
        # bc we evaluate at eval_point but translated by T_t,
        # if kernel not symmetric a != b, then we also need to translate by the mid of them.
        xx[(xx < -1) | (xx > 1)] = 1
        output.append(3 / 4 * (1 - xx * xx) * 2 / (b - a))
    return output
