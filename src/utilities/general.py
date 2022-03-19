def multi_list_generator(size):
    """
    create the exact good list of size M*M*3 for containing parameters of hawkes processes.

    Args:
        size: size of matrices. Should be an int.

    Returns:
        a list containing three lists of sizes: [size, size*size, size*size].

    Examples:
        the function returns [   [0], [[0]], [[0]]   ] for size = 1.
        the function returns [   [0,0], [[0,0],[0,0]], [[0,0],[0,0]]   ] for size = 2.


    """
    assert size >= 1, "Size not big enough."

    ans = [] # list with alpha beta nu
    interior_list = [0] * size
    # nu ALPHA BETA
    for j in range(3):
        medium_list = []  # list parameters
        for i in range(size):
            medium_list.append(interior_list.copy())  # not recreated every i so need copy.
            if j == 0 and i == 0:  # case for nu, reduces space used.
                medium_list = medium_list[0]
                break
        ans.append(medium_list)  # no copy because recreated every j
    return ans


def count_the_nans(estim_hp, NB_DIFF_TIME_ESTIM, NB_SIMUL):
    df = estim_hp.df.loc[:, ['value', 'n', 'm', 'parameter']]  # retrieve main factors
    df = df[df.loc[:, 'parameter'] == 'nu']
    df = df[df.loc[:, 'n'] == 0]
    df = df[df.loc[:, 'm'] == 0]  # wip do a single slicing

    nb_nan = df["value"].isna().sum()
    print(
        f"There are {nb_nan / NB_DIFF_TIME_ESTIM} NaNs out of the {NB_SIMUL} estimations. \nThe average is taken over the different time estimation.")
    return