# Import the required packages
from common.common import *
import numpy as np


def first_order_trial(net, W_in, T_vars, n_batch, task: str, **kwargs):
    """ Function that determines a first-order conditioning trial.

    Parameters
        net = trained network to evaluate
        W_in = initial weights to the trial
        T_vars: Tuple
            T_vars[0] = T_int = length of trial (in seconds)
            T_vars[1] = T_stim = length of time each stimulus is presented
            T_vars[2] = dt = time step of simulations
        n_batch = number of trials in mini-batch
        task = indicates which type of task to run (CS+, CS- or ctrl)

    Returns
        rts_trial = recurrent neuron activities for the trial
        Wts_trial = KC->MBON weights at the end of the trial
        wts_trial = plasticity variable for KC->MBON weights at end of trial
        vt_trial = MBON readout (valence) for the trial
        vt_opt_trial = target MBON valence for the trial
        err_trial = average error in valence for the entire trial (scalar)
        trial_odors = list of odors used in trial
        stim_list = list of stimulus time vectors
    """

    # Set the time variables
    T_int, T_stim, dt = T_vars
    time_int = torch.arange(0, T_int + dt / 10, dt)
    t_len = time_int.shape[0]
    time_zeros = torch.zeros(n_batch, t_len)

    # Generate odors and context (odor = KC = CS, context = ext = US)
    if task == 'CS+':
        r_kc_csp, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)
        trial_odors = [r_kc_csp]
        int_list = [int_cond_cs, int_test_cs]
        r_in_list = [(r_kc_csp, r_ext), (r_kc_csp, r_ext)]
    elif task == 'CS-':
        r_kc_csm, r_ext0 = net.gen_r_kc_ext(n_batch, **kwargs)  # CS- odor
        r_ext = torch.zeros_like(r_ext0)
        trial_odors = [r_kc_csm]
        int_list = [int_cs_alone, int_cs_alone]
        r_in_list = [(r_kc_csm, r_ext), (r_kc_csm, r_ext)]
    elif task == 'ctrl':
        r_kc_csp, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)
        r_kc_csm, _ = net.gen_r_kc_ext(n_batch, **kwargs)
        r_ext_csm = torch.zeros_like(r_ext)
        trial_odors = [r_kc_csp, r_kc_csm]
        int_list = [int_cond_cs, int_cs_alone]
        r_in_list = [(r_kc_csp, r_ext), (r_kc_csm, r_ext_csm)]
    else:
        raise Exception('This is not a valid first-order conditioning task.')
    # Set number of intervals
    n_int = len(int_list)

    # Lists to store activities, weights, readouts and target valences
    # In this example, everything is saved
    rts = []
    Wts = []
    wts = []
    vts = []
    vt_opts = []
    time_CSp = torch.zeros(n_batch, t_len * n_int)
    time_CSm = torch.zeros_like(time_CSp)
    time_US = torch.zeros_like(time_CSp)

    for i in range(n_int):
        # Calculate the CS stimulus presentation times
        st_times, st_len = gen_int_times(n_batch, dt, T_stim, **kwargs)
        # Calculate the interval inputs for a CS+ conditioning interval
        int_fnc = int_list[i]
        r_in = r_in_list[i]
        f_in = int_fnc(t_len, st_times, st_len, r_in, n_batch)
        r_kct, r_extt, stim_ls, vt_opt = f_in

        # Run the forward pass
        net_out = net(r_kct, r_extt, time_int, n_batch, W_in, **kwargs)
        rt_int, (Wt_int, wt_int), vt_int = net_out
        # Pass the KC->MBON weights to the next interval
        W_in = (Wt_int[-1], wt_int[-1])

        # Append the interval outputs to lists
        rts += rt_int
        Wts += Wt_int[-1]
        wts += wt_int[-1]
        vts += vt_int
        vt_opts.append(vt_opt)

        # Store the CS+ and CS- odor time series
        if task == 'CS+':
            time_CSp[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
            time_CSm[:, i * t_len:(i + 1) * t_len] = time_zeros
        elif task == 'CS-':
            time_CSp[:, i * t_len:(i + 1) * t_len] = time_zeros
            time_CSm[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        elif task == 'ctrl':
            if i == 0:
                time_CSp[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
                time_CSm[:, i * t_len:(i + 1) * t_len] = time_zeros
            if i == 1:
                time_CSp[:, i * t_len:(i + 1) * t_len] = time_zeros
                time_CSm[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        # Store US time series
        time_US[:, i * t_len:(i + 1) * t_len] = stim_ls[1]

    # Save stimuli time series
    time_all_CS = [time_CSp, time_CSm]
    stim_list = [time_all_CS, time_US]

    # Calculate the trial error
    vt_trial = torch.stack(vts, dim=-1).detach()
    vt_opt_trial = torch.cat(vt_opts, dim=-1).detach()
    err_trial = cond_err(vt_trial, vt_opt_trial).item()

    # Save the recurrent neuron activites
    rts_trial = torch.stack(rts, dim=-1).detach()
    # Save the KC->MBON weights from the end of each interval
    Wts_trial = torch.stack(Wts, dim=-1).detach()
    wts_trial = torch.stack(wts, dim=-1).detach()

    return rts_trial, Wts_trial, wts_trial, vt_trial, vt_opt_trial, err_trial, \
        trial_odors, stim_list


def second_order_trial(net, W_in, T_vars, n_batch, task: str, **kwargs):
    """ Function that determines a first-order conditioning trial.

    Parameters
        net = trained network to evaluate
        W_in = initial weights to the trial
        T_vars: Tuple
            T_vars[0] = T_int = length of trial (in seconds)
            T_vars[1] = T_stim = length of time each stimulus is presented
            T_vars[2] = dt = time step of simulations
        n_batch = number of trials in mini-batch
        task = indicates which type of task to run (CS+, extinct, 2nd)

    Returns
        rts_trial = recurrent neuron activities for the trial
        Wts_trial = KC->MBON weights at the end of the trial
        wts_trial = plasticity variable for KC->MBON weights at end of trial
        vt_trial = MBON readout (valence) for the trial
        vt_opt_trial = target MBON valence for the trial
        err_trial = average error in valence for the entire trial (scalar)
        trial_odors = list of odors used in trial
        stim_list = list of stimulus time vectors
    """

    # Set the time variables
    T_int, T_stim, dt = T_vars
    time_int = torch.arange(0, T_int + dt / 10, dt)
    t_len = time_int.shape[0]
    time_zeros = torch.zeros(n_batch, t_len)

    # Generate odors and context (odor = KC = CS, context = ext = US)
    if task == 'CS+':
        r_kc_csp, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)
        trial_odors = [r_kc_csp]
        int_list = [int_cond_cs, int_test_cs]
        r_in_list = [(r_kc_csp, r_ext), (r_kc_csp, r_ext)]
    elif task == 'extinct':
        r_kc_csp, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)  # CS- odor
        trial_odors = [r_kc_csp]
        int_list = [int_cond_cs, int_test_cs, int_test_cs]
        r_in_list = [(r_kc_csp, r_ext), (r_kc_csp, r_ext), (r_kc_csp, r_ext)]
    elif task == '2nd':
        r_kc_cs1, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)
        r_kc_cs2, _ = net.gen_r_kc_ext(n_batch, **kwargs)
        trial_odors = [r_kc_cs1, r_kc_cs2]
        int_list = [int_cond_cs, int_cond_cs2, int_test_cs]
        r_in_list = [(r_kc_cs1, r_ext),
                     ([r_kc_cs1, r_kc_cs2], r_ext),
                     (r_kc_cs2, r_ext)]
    else:
        raise Exception('This is not a valid second-order conditioning task.')
    # Set number of intervals
    n_int = len(int_list)

    # Lists to store activities, weights, readouts and target valences
    # In this example, everything is saved
    rts = []
    Wts = []
    wts = []
    vts = []
    vt_opts = []
    time_CS1 = torch.zeros(n_batch, t_len * n_int)
    time_CS2 = torch.zeros_like(time_CS1)
    time_US = torch.zeros_like(time_CS1)

    for i in range(n_int):
        # Calculate the CS stimulus presentation times
        st_times, st_len = gen_int_times(n_batch, dt, T_stim, **kwargs)
        # Calculate the interval inputs for a CS+ conditioning interval
        int_fnc = int_list[i]
        r_in = r_in_list[i]
        if task == 'extinct' and i == 2:
            f_in = int_fnc(t_len, st_times, st_len, r_in, n_batch, f_tar=0.5)
        else:
            f_in = int_fnc(t_len, st_times, st_len, r_in, n_batch)
        r_kct, r_extt, stim_ls, vt_opt = f_in

        # Run the forward pass
        net_out = net(r_kct, r_extt, time_int, n_batch, W_in, **kwargs)
        rt_int, (Wt_int, wt_int), vt_int = net_out
        # Pass the KC->MBON weights to the next interval
        W_in = (Wt_int[-1], wt_int[-1])

        # Append the interval outputs to lists
        rts += rt_int
        Wts += Wt_int[-1]
        wts += wt_int[-1]
        vts += vt_int
        vt_opts.append(vt_opt)

        # Store the CS+ and CS- odor time series
        if task in ['CS+', 'extinct']:
            time_CS1[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
            time_CS2[:, i * t_len:(i + 1) * t_len] = time_zeros
        elif task == '2nd':
            if i == 0:
                time_CS1[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
                time_CS2[:, i * t_len:(i + 1) * t_len] = time_zeros
            if i == 1:
                time_CS1[:, i * t_len:(i + 1) * t_len] = stim_ls[0][0]
                time_CS2[:, i * t_len:(i + 1) * t_len] = stim_ls[0][1]
            if i == 2:
                time_CS1[:, i * t_len:(i + 1) * t_len] = time_zeros
                time_CS2[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        # Store US time series
        time_US[:, i * t_len:(i + 1) * t_len] = stim_ls[1]

    # Save stimuli time series
    time_all_CS = [time_CS1, time_CS2]
    stim_list = [time_all_CS, time_US]

    # Calculate the trial error
    vt_trial = torch.stack(vts, dim=-1).detach()
    vt_opt_trial = torch.cat(vt_opts, dim=-1).detach()
    err_trial = cond_err(vt_trial, vt_opt_trial).item()

    # Save the recurrent neuron activites
    rts_trial = torch.stack(rts, dim=-1).detach()
    # Save the KC->MBON weights from the end of each interval
    Wts_trial = torch.stack(Wts, dim=-1).detach()
    wts_trial = torch.stack(wts, dim=-1).detach()

    return rts_trial, Wts_trial, wts_trial, vt_trial, vt_opt_trial, err_trial, \
        trial_odors, stim_list


def no_plasticity_trial(net, W_in, T_vars, n_batch, task: str, **kwargs):
    """ Function that determines a no plasticity trial.

    Parameters
        net = trained network to evaluate
        W_in = initial weights to the trial
        T_vars: Tuple
            T_vars[0] = T_int = length of trial (in seconds)
            T_vars[1] = T_stim = length of time each stimulus is presented
            T_vars[2] = dt = time step of simulations
        n_batch = number of trials in mini-batch
        task = indicates which type of task to run (CS+, CS- or ctrl)

    Returns
        rts = recurrent neuron activities for the trial
        Wts = KC->MBON weights at the end of the trial
        wts = plasticity variable for KC->MBON weights at end of trial
        vts = MBON readout (valence) for the trial
        vt_opts = target MBON valence for the trial
        err_trial = average error in valence for the entire trial (scalar)
        trial_odors = list of odors used in trial
        stim_list = list of stimulus time vectors
    """

    # Set the time variables
    T_int, T_stim, dt = T_vars
    time_int = torch.arange(0, T_int + dt / 10, dt)
    t_len = time_int.shape[0]
    T_range = [(5, 15), (20, 30)]

    # Generate odors and context (odor = KC = CS, context = ext = US)
    r_in = net.gen_r_kc_ext(n_batch, **kwargs)
    trial_odors = [r_in[0]]
    if task == 'CS+':
        CSM = False
    elif task == 'CS-':
        CSM = True
    else:
        raise Exception('This is not a valid first-order conditioning task.')

    # Calculate the CS stimulus presentation times
    st_times = []
    for i in range(len(T_range)):
        st_times_int, st_len = gen_int_times(n_batch, dt, T_stim, T_range[i],
                                             **kwargs)
        st_times.append(st_times_int)
    # Calculate the interval inputs for a CS+ conditioning interval
    f_in = tr_no_plasticity(t_len, st_times, st_len, r_in, n_batch, csm=CSM)
    r_kct, r_extt, stim_ls, vt_opt_trial = f_in

    # Run the forward pass
    net_out = net(r_kct, r_extt, time_int, n_batch, W_in, **kwargs)
    rts, (Wts, wts), vts = net_out

    # Save the recurrent neuron activities
    rts_trial = torch.stack(rts, dim=-1).detach()
    # Save the KC->MBON weights from the end of each interval
    Wts_trial = torch.stack(Wts, dim=-1).detach()
    wts_trial = torch.stack(wts, dim=-1).detach()

    # Save stimuli time series
    stim_list = [stim_ls[0], stim_ls[1]]

    # Calculate the trial error
    vt_trial = torch.stack(vts, dim=-1).detach()
    err_trial = cond_err(vt_trial, vt_opt_trial).item()

    return rts_trial, Wts_trial, wts_trial, vt_trial, vt_opt_trial, err_trial, \
        trial_odors, stim_list


def continual_trial(net, W_in, T_vars, n_batch, nsp=True, n_odor=4, **kwargs):
    """ Function that determines a no plasticity trial.

    Parameters
        net = trained network to evaluate
        W_in = initial weights to the trial
        T_vars: Tuple
            T_vars[0] = T_int = length of trial (in seconds)
            T_vars[1] = T_stim = length of time each stimulus is presented
            T_vars[2] = dt = time step of simulations
        n_batch = number of trials in mini-batch
        nsp = indicates whether non-specific potentiation is included
        n_odor = number of odors in a trial
        kwargs
            stim_mean = average number of stimulus presentations per trial

    Returns
        rts = recurrent neuron activities for the trial
        Wts = KC->MBON weights at the end of the trial
        wts = plasticity variable for KC->MBON weights at end of trial
        vts = MBON readout (valence) for the trial
        vt_opts = target MBON valence for the trial
        err_trial = average error in valence for the entire trial (scalar)
        trial_odors = list of odors used in trial
        stim_list = list of stimulus time vectors
    """

    # Set the time variables
    T_int, T_stim, dt = T_vars
    time_int = torch.arange(0, T_int + dt / 10, dt)
    t_len = time_int.shape[0]

    # Generate odors and context (odor = KC = CS, context = ext = US)
    r_kcs = []
    r_exts = []
    for i in range(n_odor):
        r_kc, _ = net.gen_r_kc_ext(n_batch, **kwargs)
        r_kcs.append(r_kc)
        # The first US is appetitive, second is aversive, rest are neutral
        if i == 0:
            r_exts.append(torch.tensor([1, 0]).repeat(n_batch, 1))
        elif i == 1:
            r_exts.append(torch.tensor([0, 1]).repeat(n_batch, 1))
        else:
            r_exts.append(torch.zeros(n_batch, 2))
    trial_odors = [r_kcs]
    r_in = (r_kcs, r_exts)

    # Calculate the CS stimulus presentation times
    st_times, st_len = gen_cont_times(n_batch, dt, T_stim, T_int, **kwargs)
    # Calculate the interval inputs for a CS+ conditioning interval
    f_in = tr_continual(t_len, st_times, st_len, r_in, n_batch)
    r_kct, r_extt, stim_ls, vt_opt_trial = f_in

    # Run the forward pass
    net_out = net(r_kct, r_extt, time_int, n_batch, W_in, nsp=nsp, **kwargs)
    rts, (Wts, wts), vts = net_out

    # Save the recurrent neuron activites
    rts_trial = torch.stack(rts, dim=-1).detach()
    # Save the KC->MBON weights from the end of each interval
    Wts_trial = torch.stack(Wts, dim=-1).detach()
    wts_trial = torch.stack(wts, dim=-1).detach()

    # Save stimuli time series
    stim_list = [stim_ls[0], stim_ls[1]]

    # Calculate the trial error
    vt_trial = torch.stack(vts, dim=-1).detach()
    err_trial = cond_err(vt_trial, vt_opt_trial).item()

    return rts_trial, Wts_trial, wts_trial, vt_trial, vt_opt_trial, err_trial, \
        trial_odors, stim_list
