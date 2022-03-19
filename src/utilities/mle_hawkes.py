# normal libraries
import warnings

import numpy as np
from scipy import linalg

from corai_error import Error_convergence
from src.hawkes.kernel import Kernel, fct_plain
from src.utilities.priv_newton_raphson import extremum_root_df, newton_raphson_hawkes
from src.wmle.diff_hawkes4mle import first_derivative, second_derivative

# priv_libraries

# other files

CONST_SCIPY = True  # if true, set the dict_clearing in the hessian as well!


def adaptor_mle(T_t, T, w=None, silent=True, first_guess=None):
    """ # case specific
    # if np.min(nu_hat) < 0.2 or np.min(alpha_hat) < 0.2 or np.min(beta_hat) < 1.:"""
    # first_guess = NU,ALPHA,BETA in np.array format.

    # w shouldn't be None, however as a safety measure, just before doing the computations !
    if w is None:  # w is a vector with weights.
        w = Kernel(fct_plain, "plain", T_max=T)(T_t, 0, T_max=T)
        # eval point equals 0 because, if the weights haven't been defined earlier,
        # it means we don't care when we estimate.
    M = len(T_t)

    if first_guess is None:
        # random init
        NU = np.full(M, 0.5) + np.random.normal(1) / 10
        ALPHA = np.full((M, M), 0.7) + np.random.normal(1) / 10
        BETA = np.full((M, M), 4.5) + np.random.normal(1)
    else:
        (NU, ALPHA, BETA) = first_guess

    # f = lambda nu, alpha, beta: - likelihood(T_t, alpha, beta, nu, T, w)
    df = lambda nu, alpha, beta: first_derivative(T_t, alpha, beta, nu, T, w)
    ddf = lambda nu, alpha, beta: second_derivative(T_t, alpha, beta, nu, T, w)

    # def f_wrap(x):
    #     return f(x[:M], np.reshape(x[M:M * M + M], (M, M)), np.reshape(x[M * M + M:], (M, M)))

    def df_wrap(x):
        return first_derivative(T_t, np.reshape(x[M:M * M + M], (M, M)),
                                np.reshape(x[M * M + M:], (M, M)), x[:M], T, w)

    def ddf_wrap(x):
        return second_derivative(T_t, np.reshape(x[M:M * M + M], (M, M)),
                                 np.reshape(x[M * M + M:], (M, M)), x[:M], T, w)

    warnings.filterwarnings("ignore")  # suppress all warnings

    if CONST_SCIPY:
        alpha_hat, beta_hat, nu_hat = scipy_root_finding(ALPHA, BETA, M, NU, ddf_wrap, df_wrap, silent)
    else:
        # personal version
        nu_hat, alpha_hat, beta_hat = newton_raphson_hawkes(df, ddf, ALPHA, BETA, NU, silent=silent)

        # test, very case specific !
        if np.min(nu_hat) < 0.2 or np.min(alpha_hat) < 0.2 or np.min(beta_hat) < 1.:
            raise Error_convergence("algorithms fails to converge.")

    return alpha_hat, beta_hat, nu_hat  # this is correct order


def scipy_root_finding(ALPHA, BETA, M, NU, ddf_wrap, df_wrap, silent):
    # scipy
    nu_hat, alpha_hat, beta_hat, opt_value_hat, success = extremum_root_df(df_wrap, ddf_wrap, NU, ALPHA, BETA)
    if not success:
        # random init and estimation again
        NU = np.random.rand(M)
        rand = np.random.rand(M, M)
        ALPHA = rand * 3
        BETA = rand * 5
        nu_hat, alpha_hat, beta_hat, opt_value_hat, success = extremum_root_df(df_wrap, ddf_wrap, NU, ALPHA, BETA)
    if not silent:
        print(f"End of Root Search, success : {success}, optimal values: {nu_hat},{alpha_hat},{beta_hat}.")
    warnings.resetwarnings()  # resets the warning filter
    # routine check that there is no explosion
    throw_error_cv(M, alpha_hat, beta_hat, nu_hat, success)
    return alpha_hat, beta_hat, nu_hat


def throw_error_cv(M, alpha_hat, beta_hat, nu_hat, success):
    not_too_big_norm = (linalg.norm(nu_hat) < 2 * M and
                        linalg.norm(alpha_hat) < 4 * M * M and
                        linalg.norm(beta_hat) < 10 * M * M)
    # sometimes algorithm hand back negative values:
    all_param_positive = (nu_hat > 0).all() and (alpha_hat > 0).all() and (beta_hat > 0).all()

    values_are_acceptable = not_too_big_norm and all_param_positive
    if not success or not values_are_acceptable:
        raise Error_convergence("algorithms fails to converge.")
