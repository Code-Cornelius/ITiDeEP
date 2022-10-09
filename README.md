# Hawkes ItiDeEP

Repository for the paper Hawkes Iterative Time-Dependent Estimation of Parameters. 
The paper is for now a preprint but might eventually get published. 

For citations in biblatex:

@article{KotlarekItideep, 
        author = "Kotlarek, Niels D. C.", 
        title = "Hawkes Iterative Time Dependent Estimation of Parameters",
        year = "2022", 
        month = 4,
	publisher = "ETH Zurich Working Paper Preprint"}

### 1. Content of Repository:

* **Simulation of multivariate Hawkes Process with exponential kernel and time-dependent parameters alpha and mu**. Beta
  could be updated to have staircase parameters (constant between jumps). Since wlog, we can assume that all Hawkes
  Processes are on an interval [0, T_max], we only allow such simulation. Burn-in is done during negative times.

* **Estimation of parameters**. One sets before estimation at what time of [0, T_max] he wishes to estimate the
  parameters. Then, for each of these times, several estimations are realised.

* **Implementation of ITiDEeP algorithm.**

### 2. What does what and where?

* The directory `data_input` is the input for the simulations. It is where all parameters, CSV files images, etc... are
  stored.
* The results are saved in `data_result`.

* In the directory `run_simulations`, one finds a simple script for running some simulations manually and locally. They
  are convenient to estimate the whole estimation batch.
* On the other hand, in `main` lie terminal computations where one time series is estimated at a time. This allows
  multi-threaded computations (simulations to be run on Euler).

* `src` is the main file of the code and classes.

* `test` file checks that everything is correct (parameters yield a stable simulation...).

* `priv_lib` are personal libraries added to the project for the latter to be self-consistent.

### 3. Configurations

The parameters for simulations are set in   `data_input/json/param_*.json`. Each configuration has a unique id, among
which:

1. `quick_config`
2. `test_config`
3. `test_config_multidim`

Uni-variate Dataset:

4. `flat_config`
5. `config_jump`
6. `config_sin`

Multidimensional Dataset:

7. `multidim_mountain`

We also wrote an `MSE` configuration, which is dedicated to the MSE convergence part of the paper.

The parameters for the functions and evolution of underlying matrices are in
`run_simulations/setup_parameters.py`, and in `src/utilities/fct_type_process.py`, `function_parameters_hawkes.py`

### 4. Local Simulations, the `run_simulation` file:

##### a) How to set parameters for local simulations?

To select the parameters of the simulation, one should modify the parameters in `data_input/JSON/param_*.json`. The
parameters from jsons are read with `data_input/json/json_loader.py`, where one can modify `STR_CONFIG`. It is the
parameter saved at the entry `STR_CONFIG` that are used. Then, inside `data_input/json/parameter_loader.py`, the
parameters are input into functions called by the scripts.

##### b) Pipeline of `run_simulations`: simple estimation

We refer to the numbering of the file.

1. `1dataset_gen_hand.py.py`. create the dataset for estimation, stored in `data_input/dataset_hawkes`,
2. `2estim_hawkes_hand.py`. estimate with the data in `data_input/dataset_hawkes` the parameters and store them
   in `data_result/name_config`. The difference between them lies in that one is automatically set regarding the
   parameters.
3. `3estim_plotting_hand.py` create the pictures: estimation mean and confidence interval.

##### c) Pipeline of `run_simulations`: batch estimations; how to speed up

It is possible to estimate in parallel for multiple time-series by running a batch of estimations at the same time on
the same core. This is what batch estimation is doing.

This is possible with `multithread_script`. It goes over all the time series (per batch as said before). After
performing the estimations, one needs to run `2estim_hawkes_terminal` once again to aggregate all the data in one file.

One should make sure the `TOTAL_NB` (from the `.sh` file) and `BATCH` parameters are correctly fixed, in order to not
miss some data, or not try to access lines that do not exist. In short, `TOTAL_NB` needs to correspond to the number of
data simulated (set in the `.json`) and
`TOTAL_NB` needs to be divisible by `BATCH`.

`TOTAL_NB` should match `NB_SIMUL` and `BATCH` depending on one's working station. The rule of thumb being fit the whole
batch's data into the RAM.

##### d) Pipeline of `run_simulations`: ITiDeEP

If one wants to use ITiDeEP, one should use `itideep_step1.py`, then the refinement with `itideep_step2.py`,
finally `itideep_sumup.py` for plotting.

### 4. Tests:

1. `test_hawkes_parameter.py` gives an overview of the parameters, what is their evolution wrt time, the highest
   eigenvalue of the transition matrix...
2. `test_hawkes_simul.py` gives a simple simulation.
3. `test_append_estim_hawkes.py` test the entangled appending behaviour for appending information to a `estim_hawkes`.
4. `test_kernel.py` to plot the different kernels studied.
5. `test_normal_estimANDiter_estim.py` testing estimation, either normal (simple WMLE) or iterative.

Indirect tests:

* `test_images_plotting_transitionKernelsANDweighting.py` file used for plotting images used in the paper regarding
  kernels and their properties.
* `test_plot_paper_geom.py` file used for plotting the behaviour of the geometric mean and rescaling function.

### 5. Command Line Computations, for fast on Euler computation `main`:

One can test that the code is functional in a terminal by running:

    python3 main/estimation1_itideep.py 1 "quick_config"  

from the root. `1` refers to which time series of the data to look at (from 1 to the NB_SIMUL, the length of the data
simulated), followed by the name of the configuration for the data.

The script `script_launching_simul` has been written to run the simulations on Euler. The pipeline written inside is:

Do `estimation1_itideep.py`, `gather_results.py`, `estimation2_itideep.py`, `gather_results.py`. Then plotting is
performed locally. We make the script run for 24H for the 2-D, but 8H is sufficient for 1-D.

Always make sure the dataset has been correctly generated to the desired parameters. We use the servers from ETH (Euler)
for performing multiple estimations.

If one estimation fails or is not completed, the file of the estimator will not be generated. Hence, the only check one
has to do after estimation to verify the integrity of data is that the desired number of estimations has been performed.
That way, we know that all the estimators have been written down.

Make sure that the parameters `number_configs`, `name_config` match the desired configuration.

### 6. Where is the data of the estimations?

We store the data in `data_result`.

Inside the directory, in `euler_hawkes` we store results from the scripts of `main`.

In `result_hawkes_test` we store the results of `run_simulations` by hand (`2estim_hawkes_hand.py`), and
inside `multithread_hawkes` the one done by batches from `2estim_terminal_hawkes.py`.

Then, we save inside the directory `{STR_CONFIG}_res` the result of the estimations, where we use the directories `data`
, `data_together`, `plot` for storing the resources in separate folders.

### 7. How to visualise the data?

2 plotting scripts, one test one for Euler. Both can be used for plotting. These are:

    main/plotting_results_estimation1&2.py

or

    run_simulations/3estim_plotting_hand.py

### 8. MSE:

Create the data with `simul_data.py`.

With `script_MSE`, one can run `estimation_MSE.py`. The logic is to fix the parameters:

`NB_TH_OF_CURRENT_ESTIMATION` (in plot), `refinement_number` (in the script)

to the current refinement of ITiDeEP.

and one runs the script `estimation_MSE.py` then `estimation_MSE_plot.py`

If one wants to run the comparison of the estimation, please use: `plot_two_mean_estimation.py`.

IF one wants to remove or add what is included in the computation of the error, please look into the
method `Estim_hawkes.add_SPE_APE_col`.

### Some Remarks:

##### Starting Point of WMLE

We fixed the starting point of the optimiser to be close to the true value of the estimation. This is set inside:
`src/utilities/pipeline_estimation.py` in the
function `complete_pipe_estim_hawkes(data_simulated, times_estimation, kernel_choice=kernel_plain, silent=False)` where
we use the known true values. The first guess parameter could be changed for something else. It is used
inside `estim_from_simulations` as:

                if first_guess is not None:  # first guess is a vector of the true values.
                    # we do the slicing now since we will need the whole vector for doing the estimation at every time
                    nu, alpha, beta = first_guess
                    nu_sliced, alpha_sliced, beta_sliced = (nu[:, i_time, 0],
                                                            alpha[:, :, i_time, 0], beta[:, :, i_time, 0])
                    # we fix the value 0 because the last dimension is the one of parallel simulations,
                    # where the true value is identical for all of them.
                    current_first_guess = (nu_sliced, alpha_sliced, beta_sliced)

However, if one passes `None` to the parameter `first_guess`, then in `adaptor_mle`:
one uses the WMLE algorithm with a random input that is coded as:

    if first_guess is None:
        # random init
        NU = np.full(M, 0.5) + np.random.normal(1) / 10
        ALPHA = np.full((M, M), 0.7) + np.random.normal(1) / 10
        BETA = np.full((M, M), 4.5) + np.random.normal(1)
    else:
        (NU, ALPHA, BETA) = first_guess

##### Algorithm choice for WMLE

We set a constant `CONST_SCIPY` that adapt the behaviour of the function
`adaptor_mle` and then inside either  `extremum_root_df` or `newton_raphson_hawkes`are used
(in other words use the scipy algorithm or the personal implementation). The latter is more resilient to the first guess
choice, but the former is faster. Since we decided to start with a very close first guess, we used the algorithm from
scipy.

Also, we converted the data into low memory footprint data. The induced error is negligible for our computations.

###### Changing the rescale function

We have not implemented an automatic way to choose the rescale function. To change the function in the simulation, go
to `src/utilities/fct_itideep.py`
and at the end of the function `compute_scaling_itideep`, the rescale function can be chosen. There is a convenient
parameter to do it all at once at the beginning of the file called
`FUNCTION_CHOICE`.
