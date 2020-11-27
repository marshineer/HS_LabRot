# Import the required packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# TODO: Try to eliminate as many passed parameters as possible (eliminate n_natch?)
def cond_loss(vt, vt_opt, r_DAN, lam=0.1):
    """ Calculates the loss for conditioning tasks.

    Composed of an MSE cost based on the difference between output and
    target valence, and a regularization cost that penalizes excess
    dopaminergic activity. Reference Eqs. (3) and (9) in Jiang 2020.

    Parameters
        vt = time dependent valence output of network
        vt_opt = target valence (must be a torch tensor)
        r_DAN = time series of dopaminergic neuron activities
        lam = regularization constant

    Returns
        loss_tot = scalar loss used in backprop
    """

    # Set the baseline DAN activity
    DAN_baseline = 0.1

    # Calculate the MSE loss of the valence
    v_sum = torch.mean((vt - vt_opt) ** 2, dim=1)
    v_loss = torch.mean(v_sum)

    # Calculate regularization term
    r_sum = torch.sum(F.relu(r_DAN - DAN_baseline) ** 2, dim=1)
    r_loss = torch.mean(r_sum, dim=1) * lam

    # Calculate the summed loss (size = n_batch)
    loss = v_loss + r_loss

    # Average the loss over all batches
    loss_tot = torch.mean(loss)

    return loss_tot


def first_order_cond_csp(*, t_len, st_times, st_len, r_in, n_batch, p_ctrl=0.,
                         p_csm=0., **kwargs):
    """ Runs a first-order conditioning interval (CS+ and US).

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
        p_ctrl (kwarg) = fraction of control trials during training
        p_csm (kwarg) = fraction of control trials that are CS-
    """

    # Set odors and context signals for this interval
    r_kc, r_ext = r_in
    n_kc = r_kc.shape[1]
    n_ext = r_ext.shape[1]

    # TODO: Incorporate control trials into interval functions
    #  How can I pass control trial info to the next interval?
    #  eg. If the conditioning interval is a control, this must be know for
    #  the test interval.
    #  Do I need to do this, or can I just make separate 'train' functions?
    # # Determine whether CS or US are randomly omitted
    # omit_inds = torch.rand(n_batch) < p_ctrl
    # # If omitted, determine which one is omitted
    # x_omit_CS = torch.rand(n_batch)
    # omit_CS_inds = torch.logical_and(omit_inds, x_omit_CS > p_csm)
    # omit_US_inds = torch.logical_and(omit_inds, x_omit_CS < p_csm)

    # Initialize stimulus time matrices
    time_CS = torch.zeros(n_batch, t_len)
    time_US = torch.zeros_like(time_CS)
    vt_opt = torch.zeros_like(time_CS)

    # Set the stimulus step inputs
    for b in range(n_batch):
        # Convert stimulus time into range of indices
        stim_inds = st_times[b] + torch.arange(st_len)
        # Set the CS input times
        time_CS[b, stim_inds] = 1
        # Set the US input times
        time_US[b, (stim_inds + st_len)] = 1

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    stim_ls = [time_CS, time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def first_order_csm(*, t_len, st_times, st_len, r_in, n_batch, **kwargs):
    """ Runs a first-order conditioning interval (CS- alone).

    This function can be used for both the conditioning and test intervals
    of CS- trials. Only the CS stimulus is presented.

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
    """

    # TODO: Check for control trials
    # Set odors and context signals for each trial
    r_kc, r_ext = r_in
    n_kc = r_kc.shape[1]
    n_ext = r_ext.shape[1]

    # Initialize stimulus time matrices
    time_CS = torch.zeros(n_batch, t_len)
    time_US = torch.zeros_like(time_CS)
    vt_opt = torch.zeros_like(time_CS)

    # Set the stimulus step inputs
    for b in range(n_batch):
        # Convert stimulus time into range of indices
        stim_inds = st_times[b] + torch.arange(st_len)
        # Set the CS input times
        time_CS[b, stim_inds] = 1

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    stim_ls = [time_CS, time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def second_order_cond(*, t_len, st_times, st_len, r_in, n_batch, **kwargs):
    """ Runs a first-order conditioning interval (CS alone).

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
    """

    # Set odors and context signals for each trial
    r_kc1, r_ext = r_in
    n_kc = r_kc1.shape[1]
    n_ext = r_ext.shape[1]
    # Initialize a second odor
    r_kc2 = torch.zeros_like(r_kc1)
    n_ones = r_kc1.shape[1]

    # Initialize stimulus time matrices
    time_CS1 = torch.zeros(n_batch, t_len)
    time_CS2 = torch.zeros_like(time_CS1)
    time_US = torch.zeros_like(time_CS1)
    vt_opt = torch.zeros_like(time_CS1)

    # Set the stimulus step inputs
    for b in range(n_batch):
        # Shuffle the indices to create a new second odor
        new_inds = torch.multinomial(torch.ones(n_ones), n_ones)
        r_kc2[b, :] = r_kc1[b, new_inds]

        # Convert stimulus time into range of indices
        stim_inds = st_times[b] + torch.arange(st_len)
        # Set the CS1 input times
        time_CS1[b, stim_inds] = 1
        # Set the CS2 input times
        time_CS2[b, (stim_inds + st_len)] = 1

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc1,
                         time_CS1.repeat(n_kc, 1, 1))
    r_kct += torch.einsum('bm, mbt -> bmt', r_kc2,
                          time_CS2.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    time_all_CS = [time_CS1, time_CS2]
    stim_ls = [time_all_CS, time_US]
    r_next = ([r_kc1, r_kc2], r_ext)

    return r_next, r_kct, r_extt, stim_ls, vt_opt


def classic_cond_test(*, t_len, st_times, st_len, r_in, n_batch, **kwargs):
    """ Runs a first-order test interval (CS+ and target valence).

    This function can be used for both first- and second-order tests, as well
    as extinction conditioning.

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
    """

    # TODO: Check how many odors are being passed. Only need the first.
    #  Look into this for all interval functions. What combinations of passing
    #  odors are there?
    #  eg. For the second order test, I need to check the r_kc2, NOT r_kc1.
    #  Therefore, maybe 2nd-order test needs its own function? Add an extra input
    #  to indicate which odor from r_in is relevant?
    # Set odors and context signals for each trial
    r_kc, r_ext = r_in
    n_kc = r_kc.shape[1]
    n_ext = r_ext.shape[1]

    # Initialize stimulus time matrices
    time_CS = torch.zeros(n_batch, t_len)
    time_US = torch.zeros_like(time_CS)
    vt_opt = torch.zeros_like(time_CS)

    # Set the stimulus step inputs
    for b in range(n_batch):
        # Convert stimulus time into range of indices
        stim_inds = st_times[b] + torch.arange(st_len)
        # Set the CS input times
        time_CS[b, stim_inds] = 1
        # Set the target valence times
        if r_ext[b, 0] == 1:
            vt_opt[b, (stim_inds + 1)] = 1
        else:
            vt_opt[b, (stim_inds + 1)] = -1

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    stim_ls = [time_CS, time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def extinct_test(*, t_len, st_times, st_len, r_in, n_batch, **kwargs):
    """ Runs an extinction interval.

    Consists of CS+ stimulus presentation without the relevant US. The
    extinction test always occurs after a first order test. In this case,
    the target valence is half that of a first order (CS+) test.

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
    """

    # Set odors and context signals for each trial
    r_kc, r_ext = r_in
    n_kc = r_kc.shape[1]
    n_ext = r_ext.shape[1]

    # Initialize stimulus time matrices
    time_CS = torch.zeros(n_batch, t_len)
    time_US = torch.zeros_like(time_CS)
    vt_opt = torch.zeros_like(time_CS)

    # Set the stimulus step inputs
    for b in range(n_batch):
        # Convert stimulus time into range of indices
        stim_inds = st_times[b] + torch.arange(st_len)
        # Set the CS input times
        time_CS[b, stim_inds] = 1
        # Set the target valence times
        if r_ext[b, 0] == 1:
            vt_opt[b, (stim_inds + 1)] = 1 / 2
        else:
            vt_opt[b, (stim_inds + 1)] = -1 / 2

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    stim_ls = [time_CS, time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def no_plasticity_trial(*, t_len, st_times, st_len, r_in, n_batch, dt,
                        p_ctrl=0.5, **kwargs):
    """ Runs a full no-plasticity trial (CS+ and US).

    In half of the training trials, the second odor is switched to prevent
    over-generalization.

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
        p_ctrl = fraction of control trials
    """

    # Generates a second set of stimulus times for the second presentation
    st_times2, _ = gen_st_times(dt, n_batch, T_range=(20, 30))

    # Set odors and context signals for each trial
    r_kc1, r_ext = r_in
    n_kc = r_kc1.shape[1]
    n_ext = r_ext.shape[1]
    # Define a second set of odors for generalization trials
    r_kc2 = r_kc1.clone()
    # r_kc2 = torch.zeros_like(r_kc1)
    n_ones = r_kc1.shape[1]

    # Initialize stimulus time matrices
    time_CS1 = torch.zeros(n_batch, t_len)
    time_CS2 = torch.zeros_like(time_CS1)
    time_US = torch.zeros_like(time_CS1)
    vt_opt = torch.zeros_like(time_CS1)

    # Determine whether CS+ is switched for a novel odor
    switch_inds = torch.rand(n_batch) < p_ctrl

    # Set the stimulus step inputs
    for b in range(n_batch):
        # Convert stimulus time into range of indices
        stim_inds1 = st_times[b] + torch.arange(st_len)
        stim_inds2 = st_times2[b] + torch.arange(st_len)
        # Set the CS input times
        time_CS1[b, stim_inds1] = 1

        # If it is a control trial, switch the odor (target valence is zero)
        if switch_inds[b]:
            new_inds = torch.multinomial(torch.ones(n_ones), n_ones)
            r_kc2[b, :] = r_kc1[b, new_inds]
            time_CS2[b, stim_inds2] = 1
        # If the odor is not switched, set the target valence
        else:
            time_CS1[b, stim_inds2] = 1
            if r_ext[b, 0] == 1:
                vt_opt[b, (stim_inds2 + 1)] = 1 / 2
            else:
                vt_opt[b, (stim_inds2 + 1)] = -1 / 2

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc1,
                         time_CS1.repeat(n_kc, 1, 1))
    r_kct += torch.einsum('bm, mbt -> bmt', r_kc2,
                          time_CS2.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    time_all_CS = [time_CS1, time_CS2]
    stim_ls = [time_all_CS, time_US]
    r_next = ([r_kc1, r_kc2], r_ext)

    return r_next, r_kct, r_extt, stim_ls, vt_opt


def continual_trail(*, t_len, st_times, st_len, r_in, n_batch, dt, **kwargs):
    """ Runs a continual learning trial (two CS- and two CS+).

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
    """

    # Draw the CS presentation times from a Poisson distribution
    st_times, st_len = gen_cont_times(dt, n_batch, **kwargs)
    n_odor = len(st_times)

    # Set odors and context signals
    r_kc1, r_ext1 = r_in
    # Define the number of Kenyon cells (n_kc) and dim of context (n_ext)
    n_kc = r_kc1.shape[1]
    n_ext = r_ext1.shape[1]
    # Initialize matrices to store additional odors and their context signals
    r_kcs = [0] * n_odor
    r_kcs[0] = r_kc1
    r_exts = [0] * n_odor
    # TODO: r_ext1 should be positive
    r_exts[0] = r_ext1

    # Initialize activity matrices
    r_kct = torch.zeros(n_batch, n_kc, t_len)
    r_extt = torch.zeros(n_batch, n_ext, t_len)

    # Initialize stimulus time matrices
    time_CS = torch.zeros(n_odor, n_batch, t_len)
    time_US = torch.zeros_like(time_CS)
    vt_opt = torch.zeros(n_batch, t_len)

    # Set the stimulus step inputs
    for i in range(n_odor):
        time_CS_odor = torch.zeros(n_batch, t_len)
        time_US_odor = torch.zeros_like(time_CS_odor)
        for b in range(n_batch):
            # Generate new odors
            if i > 0:
                # TODO: r_ext should be positive for the first odor, negative
                #  for the second odor and zero for any following odors
                r_kcs[i] = torch.zeros_like(r_kc1)
                r_exts[i] = torch.zeros_like(r_ext1)
                new_kc_inds = torch.multinomial(torch.ones(n_kc), n_kc)
                r_kcs[i][b, :] = r_kc1[b, new_kc_inds]
                r_exts[i][b, :] = torch.multinomial(torch.ones(n_ext), n_ext)

            for j, st in enumerate(st_times[i][b]):
                # Convert stimulus time into range of indices
                stim_inds = st + torch.arange(st_len)
                # Set the CS input times
                time_CS_odor[b, stim_inds] = 1

                if i < (n_odor / 2):
                    # Set the US input times
                    time_US_odor[b, (stim_inds + st_len)] = 1
                    # Set target valence on each presentation after the first
                    if j > 0:
                        if r_exts[i][b, 0] == 1:
                            vt_opt[b, (stim_inds + 1)] = 1
                        else:
                            vt_opt[b, (stim_inds + 1)] = -1

        # Calculate the input neuron activity time series (KC = CS, ext = US)
        r_kct += torch.einsum('bm, mbt -> bmt', r_kcs[i],
                              time_CS_odor.repeat(n_kc, 1, 1))
        r_extt += torch.einsum('bm, mbt -> bmt', r_exts[i],
                               time_US_odor.repeat(n_ext, 1, 1))

        # Save the CS and US stimuli time series
        time_CS[i, :, :] = time_CS_odor
        time_US[i, :, :] = time_US_odor

    # Combine the time matrices into a list
    stim_ls = [time_CS, time_US]
    r_next = (r_kcs, r_exts)

    return r_next, r_kct, r_extt, stim_ls, vt_opt


def gen_st_times(dt, n_batch, T_stim=2, T_range=(5, 15), **kwargs):
    """ Generates an array of stimulus presentation times for all trials

    Parameters
        dt = time step of simulations
        n_batch = number of trials in eval-batch
        T_stim = length of time each stimulus is presented
        T_range: tuple = range in which stimulus can be presented

    Returns
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
    """

    # Present the stimulus between 5-15s of each interval
    min_ind = int(T_range[0] / dt)
    max_ind = int((T_range[1] - T_stim) / dt)
    st_times = torch.randint(min_ind, max_ind, (n_batch,)) + min_ind
    st_len = int(T_stim / dt)

    return st_times, st_len


def gen_cont_times(dt, n_batch, T_stim=2, T_int=200, stim_mean=2, n_odor=4,
                   **kwargs):
    """ Generates stimulus presentation times for continual learning trials.

    Parameters
        dt = time step of simulations
        n_batch = number of trials in eval-batch
        T_stim = length of time each stimulus is presented
        T_int (kwarg) = length of trial (in seconds)
        stim_mean (kwarg) = average number of stimulus presentations per trial
        n_odor = number of odors in a trial

    Returns
        st_times = lists of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
    """

    # Poisson rate of stimulus presentations
    stim_rate = stim_mean / T_int

    # Initialize stimulus presentation times array
    st_times = [0] * n_odor
    st_len = int(T_stim / dt)

    # Generate a list of stimulus presentation times for each trial
    for b in range(n_odor):
        batch_times = [0] * n_batch
        for i in range(n_batch):
            trial_times = []
            last_time = 0
            while True:
                stim_isi = -torch.log(torch.rand(1)) / stim_rate
                next_time = last_time + stim_isi
                if next_time < (T_int - 2 * T_stim):
                    # Stimulus times are indices (not times)
                    trial_times.append((next_time / dt).int())
                    last_time += stim_isi
                # Ensure at least one presentation of each stimuli
                elif last_time == 0:
                    continue
                else:
                    break
            batch_times[i] = torch.stack(trial_times)
        st_times[b] = batch_times

    return st_times, st_len


def print_trial(network, plt_ttl: str, dt=0.5):
    """ Plots a figure similar to Figure 2 from Jiang 2020.

    Runs the network using a novel combination of stimuli, then prints the
    result. Top: time series of the various stimuli (CS and US), as well as
    the target valence and readout. Bottom: activity of eight randomly chosen
    mushroom body output neurons (MBONs).

    Paramters
        network = previously trained RNN
        plt_ttl = title of plot
        dt = time step of the simulation/plot
    """

    # Define plot font sizes
    label_font = 18
    title_font = 24
    legend_font = 12

    # Run the network
    r_out, vt, vt_opt, loss_hist, stim_ls = network.train_net(n_epoch=1, n_batch=1, p_ctrl=0)
    r_out = r_out.detach().numpy().squeeze()
    vt = vt.detach().numpy().squeeze()
    vt_opt = vt_opt.detach().numpy().squeeze()
    plot_CS = stim_ls[0].numpy().squeeze()
    plot_US = stim_ls[1].numpy().squeeze()
    plot_time = np.arange(plot_CS.size) * dt

    # Determine plot labels
    n_CS = plot_CS.shape[0]
    CS_labels = []
    for i in range(n_CS):
        pass
    # Plot the conditioning and test
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 4]})
    ax1.plot(plot_time, vt, label='Readout')
    ax1.plot(plot_time, vt_opt, label='Target')
    ax1.plot(plot_time, plot_CS, label='CS+')
    # Second-order conditioning involves an additional stimulus time series
    for i in range(plot_CS.shape[0]):
        plot_CS2 = stim_ls[2].numpy().squeeze()
        ax1.plot(plot_time, plot_CS2, label='CS2')
    ax1.plot(plot_time, plot_US, label='US')
    ax1.set_ylabel('Value', fontsize=label_font)
    ax1.set_title(plt_ttl, fontsize=title_font)
    ax1.legend(fontsize=legend_font)

    # Plot the activities of a few MBONs
    plot_neurs = np.random.choice(network.N_MBON, size=8, replace=False)
    r_max = np.max(r_out)
    for i, n in enumerate(plot_neurs):
        ax2.plot(plot_time, (r_out[n, :] / r_max) + (i * 2 / 3), '-k')
    ax2.set_xlabel('Time', fontsize=label_font)
    ax2.set_ylabel('Normalized Activity', fontsize=label_font)
    ax2.set_yticks([])
    fig.tight_layout();
