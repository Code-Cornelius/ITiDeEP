import numpy as np

# SETUP FILE FOR SIMULATIONS AND ESTIMATION
# set seed, set parameters and global parameters
from corai_error import Error_not_allowed_input


def setup_parameters(seed, dim, styl):
    # dim 1,2,5
    # styl: 1,2,3,4
    #           1,2
    #             1
    np.random.seed(seed)

    # time_batch corresponds to how much time required for 50 jumps
    if dim == 1:
        if styl == 1:
            alpha = [[0.6]]
            beta = [[4]]
            nu = [0.4]
            t0, time_batch = 0, 150
        elif styl == 2:
            alpha = [[2.]]
            beta = [[2.4]]
            nu = [0.2]
            t0, time_batch = 0, 45
        elif styl == 3:
            alpha = [[1.75]]
            beta = [[2]]
            nu = [0.5]
            t0, time_batch = 0, 15
        elif styl == 4:
            alpha = [[1]]
            beta = [[4]]
            nu = [0.2]
            t0, time_batch = 0, 45
        else:
            raise Error_not_allowed_input("Problem with given dimension")


    elif dim == 2:
        if styl == 1:
            alpha = [[1.5, 0.5],
                     [0.5, 1.5]]
            beta = [[5, 3],
                    [3, 5]]
            nu = [0.25, 0.25]
            t0, time_batch = 0, 70
        elif styl == 2:  # case financial with model almost critical
            # TO USE WITH SWITCH 0, 21, 3!
            alpha = [[1.5, 0.5],
                     [1., 1.5]]
            beta = [[3, 2],
                    [2, 3]]
            nu = [0.1, 0.1]
            t0, time_batch = 0, 70
        else:
            raise Error_not_allowed_input("Problem with given dimension")


    elif dim == 5:
        alpha = [[1, 0.8, 0.5, 0.5, 0.5],
                 [0.8, 1, 0.5, 0.5, 0.5],
                 [0, 0, 0.5, 0, 0],
                 [0, 0, 0., 0.5, 0.5],
                 [0, 0, 0., 0.5, 0.5]]
        beta = [[20, 12, 12, 8, 10],
                [12, 20, 12, 8, 10],
                [0, 0, 15, 0, 0],
                [0, 0, 0, 8, 10],
                [0, 0, 0, 8, 10]]
        nu = [0.2, 0.2, 0.2, 0.2, 0.2]
        t0, time_batch = 0, 5

    else:
        raise Error_not_allowed_input("Problem with given dimension")

    alpha, beta, nu = np.array(alpha, dtype=float), np.array(beta, dtype=float), np.array(nu, dtype=float)
    # dtype precised in order to cast ints into floats.
    parameters = [nu.copy(), alpha.copy(), beta.copy()]

    return parameters, t0, time_batch
