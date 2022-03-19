import numpy as np
from scipy.optimize import root

# my libraries
from corai_error import Error_convergence
from corai_util.tools import function_iterable

is_invertible = function_iterable.is_numpy_matrix_invertible


def extremum_root_df(df, ddf, nu, alpha, beta):
    M = len(nu)

    x0 = np.append(np.append(nu, np.ravel(alpha)), np.ravel(beta))  # the ravel flattens the matrix
    res = root(df, x0, method='hybr', jac=ddf,
               options={'maxfev': 300})  # 300 is arbitrary. Since we are starting the opti
    # close to the solution, we should converge to the solution very quickly.
    # maxfev
    #     The maximum number of calls to the function. If zero, then 100*(N+1) is the maximum where N is the number of elements in x0.
    x, opt_value, success = res.x, res.fun, res.success
    nu = x[:M]
    alpha = np.reshape(x[M:M * M + M], (M, M))  # matrix shape
    beta = np.reshape(x[M * M + M:], (M, M))
    return nu, alpha, beta, opt_value, success

# does not work

# def newton_raphson_hawkes_pytorch(f, df, ddf, nu, alpha, beta):
#     M = len(nu)
#
#     x0 = np.append(np.append(nu, np.ravel(alpha)), np.ravel(beta))  # the ravel flattens the matrix
#     res = minimize(f, x0, method='Newton-CG',  jac=df, hess=ddf)
#     x, opt_value, success = res.x, res.fun, res.success
#     nu = x[:M]
#     alpha = np.reshape(x[M:M * M + M], (M, M))  # matrix shape
#     beta = np.reshape(x[M * M + M:], (M, M))
#     return nu, alpha, beta, opt_value, success

def newton_raphson_hawkes(df, ddf, ALPHA, BETA, MU, e=1E-2, tol=3E-4, silent=True):
    """
    CASE SPECIFIC
    DOES NOT ACCEPT NEGATIVE BETA, AND ALPHA/NU < 0.1
    Args:
        df: the derivative
        ddf: the hessian
        ALPHA:  first guess
        BETA:  first guess
        MU:  first guess
        e:  error allowed
        tol:  tolerance of movement minimal
        silent:  debug ?

    Returns:

    """

    # nb of dimensions
    M = len(MU)

    tol *= M*M # tol scaling depending on the nb of param!

    # while f is bigger than the tolerance.
    number_of_step_crash = 0
    step = 0  # first change in norm
    multi = 0.5
    b = 0.1
    # that number is for saying that if it explodes too many times, just leave that realisation out.
    nb_of_explosions = 0
    changed = False

    # this is to know if we reached a point where there is huge movement,
    # so starting from that index, we re initialize the multi coefficient.
    reset_index = 0
    derivative = df(MU, ALPHA, BETA)

    while np.linalg.norm(derivative, 2) > e and step > tol or number_of_step_crash == 0:  # I use norm 2 as criteria
        # Printing
        if not silent and number_of_step_crash % 1 == 0:
            print(f"Step Newton {number_of_step_crash} result {np.linalg.norm(derivative, 2)}, change in norm {step}.")
            print(f"GUESSES : \n ALPHA : \n {ALPHA}  \n BETA : \n {BETA} \n NU : \n {MU}")
        if number_of_step_crash > 1500:
            raise Exception("Is the function flat enough ?")
        number_of_step_crash += 1

        # stock the old guess for update
        old_x0 = np.append(np.append(MU, np.ravel(ALPHA)), np.ravel(BETA))  # the ravel flattens the matrix

        # compute the shift
        hessian = ddf(MU, ALPHA, BETA)
        # if not invertible you re do the simulations. Solve is also more effective than computing the inverse

        if not is_invertible(hessian):
            raise Error_convergence("Hessian is not invertible")
        direction = np.linalg.solve(hessian, derivative)

        # test :print(previous_Rs);;print(previous_Rs_dash_dash);print(previous_denomR)
        # new coefficient Armijo rule if the iterations are very high.
        # the conditions are first
        # 1. if we are at the beg of the iterations and convergence
        # 2. if we are not too close yet of the objective, the derivative equal to 0.
        # One scale by M bc more coeff implies bigger derivative
        # 3. nb of explosions, if there are explosions it means I need to be more gentle to find the objective
        if number_of_step_crash - reset_index < 10 and np.linalg.norm(derivative, 2) > 50 * M * M \
                and nb_of_explosions < 2:
            multi = 1 / M ** 4
        elif number_of_step_crash - reset_index < 10 and np.linalg.norm(derivative, 2) > 2 * M * M \
                and nb_of_explosions < 5:
            multi = 0.6 / M ** 4
        elif number_of_step_crash - reset_index < 100 and np.linalg.norm(derivative, 2) > 0.01 * M * M:
            multi = 0.2 / M ** 4
        elif number_of_step_crash < 750:  # and np.linalg.norm(derivative, 2) > 0.1*M:
            multi = 0.05 / M ** 4
        else:
            raise Error_convergence("algorithms fails to converge.")
            # variable_in_armijo = MU, ALPHA, BETA
            # multi, changed = armijo_rule(df, ddf, variable_in_armijo, direction, a=multi, sigma=0.5, b=b)

        # new position
        x0 = old_x0 - multi * direction

        # # if the coefficient given by armijo is too small I change it.
        # if multi < 10e-8:
        #     changed = False
        #     multi = 10e-3
        #
        # # IF armijo was applied,
        # # in order to still got some loose when moving on the derivatives, I divide by the coef
        # if changed:
        #     multi /= b

        # if the max is too big I replace the value by a random number between 0 and 1.
        # Also, I synchronize the alpha and the beta in order to avoid boundary problem.
        if np.max(np.abs(x0)) > 25:  # absolute in case goes negative infinity.
            nb_of_explosions += 1
            for i in range(len(x0)):
                if i < M:
                    x0[i] = np.random.rand(1)
                elif i < M + M * M:
                    random_value = np.random.rand(1)
                    x0[i] = random_value
                    x0[i + M * M] = 2 * M * M * random_value
                # The list is already full, break.
                else:
                    break
            # I reset the step size.
            reset_index = number_of_step_crash

        # Here I deal with negative parameters
        for i in range(len(x0)):
            if i >= M + M * M:  # betas,
                # they can't be negative otherwise overflow in many expressions involving exponentials.
                # However, close to zero may happen in multi dim.
                if x0[i] < 0:
                    x0[i] = 0.1
            elif x0[i] < 0.1: # other parameters
                x0[i] = 0.1
        # In order to avoid infinite loops, I check if there was too many blow ups.
        # If there are too many, I return flag as false.
        if nb_of_explosions > 10:
            raise Error_convergence("Algorithms fails to find a starting point")

        # normally step won't be used. It is a dummy variable "moved or not". (under Armijo)
        step = np.linalg.norm(x0 - old_x0, 2)
        # get back the guesses
        MU = x0[:M]
        ALPHA = np.reshape(x0[M:M * M + M], (M, M))  # matrix shape
        BETA = np.reshape(x0[M * M + M:], (M, M))

        # big changes, reset the multi index. Makes explosion faster. The Steps shouldn't be that big.
        if step > 5:
            reset_index = number_of_step_crash

        # reduces some computations to put it here
        derivative = df(MU, ALPHA, BETA)

    # True because it was successful.
    return MU, ALPHA, BETA


def armijo_rule(f, df, x0, direction, a, sigma, b):
    """

    Args:
        f:
        df:
        x0:
        direction:
        a:
        sigma:
        b:

    Returns:
    return 3 things, first the coefficient by which to multiply the steepest descent.
    also which direction has to change.
    finally whether the coefficient has been changed.

    """

    if abs(b) >= 1:
        raise Exception("b has to be smaller than 1.")

    MU, ALPHA, BETA = x0
    M = len(ALPHA)

    changed = False
    dir1 = np.reshape(direction[M:M * M + M], (M, M))  # matrix shape
    dir2 = np.reshape(direction[M * M + M:], (M, M))
    vector_limit_sup = np.matmul(df(MU, ALPHA, BETA), direction)  # that s the limit given in Armijo rule.
    condition = f(MU + a * direction[:M],
                  ALPHA + a * dir1,
                  BETA + a * dir2) - f(MU, ALPHA, BETA) <= sigma * a * vector_limit_sup

    # I put .all, I only update if every dimension helps improving.
    # a > 10e-1O in order to not have a too small step.

    while not all(condition) and a > 10e-10:
        a *= b
        changed = True
        condition = f(MU + a * direction[:M],
                      ALPHA + a * dir1,
                      BETA + a * dir2) - f(MU, ALPHA, BETA) <= sigma * a * vector_limit_sup

    print("we are in ARMIJO condition because too many steps, the ok directions and step :" + str(
        condition) + " and " + str(a))
    print("derivatives value : ", f(MU, ALPHA, BETA))
    return a, changed
