# Import the required packages
import torch
import torch.nn.functional as F


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


def first_order_cond_csp(t_len, st_times, st_len, r_in, n_batch, **kwargs):
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
    r_kcs, r_ext = r_in
    r_kc = r_kcs[0]
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
        # Set the US input times
        time_US[b, (stim_inds + st_len)] = 1

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    stim_ls = [[time_CS], time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def first_order_csm(t_len, st_times, st_len, r_in, n_batch, **kwargs):
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

    # Set odors and context signals for each trial
    r_kcs, r_ext = r_in
    r_kc = r_kcs[0]
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
    stim_ls = [[time_CS], time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def first_order_test(t_len, st_times, st_len, r_in, n_batch, **kwargs):
    """ Runs a first-order test interval (CS+ and target valence).

    This function can be used for both first--order tests and extinction
    conditioning. This interval must follow an interval with a single odor.
    i.e. the input r_kc must be of size (n_batch, n_kc)

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
    """

    # Set odors and context signals for each trial
    r_kcs, r_ext = r_in
    r_kc = r_kcs[0]
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
    stim_ls = [[time_CS], time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def extinct_test(t_len, st_times, st_len, r_in, n_batch, **kwargs):
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
    r_kcs, r_ext = r_in
    r_kc = r_kcs[0]
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
    stim_ls = [[time_CS], time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def second_order_cond(t_len, st_times, st_len, r_in, n_batch, **kwargs):
    """ Runs a first-order conditioning interval (CS alone).

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
    """

    # Set odors and context signals for each trial
    r_kcs, r_ext = r_in
    # The input odor is the second odor presented (CS1)
    r_kc2 = r_kcs[0]
    n_kc = r_kc2.shape[1]
    n_ext = r_ext.shape[1]
    # Initialize a second odor
    r_kc1 = torch.zeros_like(r_kc2)
    n_ones = r_kc2[0, :].sum().int()

    # Initialize stimulus time matrices
    time_CS1 = torch.zeros(n_batch, t_len)
    time_CS2 = torch.zeros_like(time_CS1)
    time_US = torch.zeros_like(time_CS1)
    vt_opt = torch.zeros_like(time_CS1)

    # Set the stimulus step inputs
    for b in range(n_batch):
        # Shuffle the indices to create a new second odor
        new_inds = torch.multinomial(torch.ones(n_kc), n_ones)
        # r_kc1[b, :] = r_kc2[b, new_inds]
        r_kc1[b, new_inds] = 1

        # Convert stimulus time into range of indices
        stim_inds = st_times[b] + torch.arange(st_len)
        # Set the CS1 input times
        time_CS1[b, stim_inds] = 1
        # Set the CS2 input times
        time_CS2[b, (stim_inds + st_len)] = 1
        # Set the target valence
        vt_opt[b, (stim_inds + st_len + 1)] = 1

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


def second_order_test(t_len, st_times, st_len, r_in, n_batch, **kwargs):
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

    # Set odors and context signals for each trial
    r_kcs, r_ext = r_in
    r_kc = r_kcs[1]
    n_kc = r_kc.shape[1]
    n_ext = r_ext.shape[1]

    # Initialize stimulus time matrices
    time_CS1 = torch.zeros(n_batch, t_len)
    time_CS2 = torch.zeros_like(time_CS1)
    time_US = torch.zeros_like(time_CS1)
    vt_opt = torch.zeros_like(time_CS1)

    # Set the stimulus step inputs
    for b in range(n_batch):
        # Convert stimulus time into range of indices
        stim_inds = st_times[b] + torch.arange(st_len)
        # Set the CS input times
        time_CS2[b, stim_inds] = 1
        # Set the target valence times
        if r_ext[b, 0] == 1:
            vt_opt[b, (stim_inds + 1)] = 1
        else:
            vt_opt[b, (stim_inds + 1)] = -1

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS2.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    time_all_CS = [time_CS1, time_CS2]
    stim_ls = [time_all_CS, time_US]

    return r_in, r_kct, r_extt, stim_ls, vt_opt


def no_plasticity_trial(t_len, st_times, st_len, r_in, n_batch, T_stim,
                        dt, p_ctrl=0., **kwargs):
    """ Runs a full no-plasticity trial (CS+ and US).

    In half of the training trials, the second odor is switched to prevent
    over-generalization.

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
        p_ctrl = fraction of control (generalization) trials
    """

    # Generates a second set of stimulus times for the second presentation
    st_times2, _ = gen_int_times(n_batch, dt, T_stim, T_range=(20, 30))

    # Set odors and context signals for each trial
    r_kcs, r_ext = r_in
    r_kc1 = r_kcs[0]
    n_kc = r_kc1.shape[1]
    n_ext = r_ext.shape[1]
    # Define a second set of odors for generalization trials
    # r_kc2 = r_kc1.clone()
    r_kc2 = torch.zeros_like(r_kc1)
    n_ones = r_kc1[0, :].sum().int()

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
            time_CS2[b, stim_inds2] = 1
            new_inds = torch.multinomial(torch.ones(n_kc), n_ones)
            # r_kc2[b, :] = r_kc1[b, new_inds]
            r_kc2[b, new_inds] = 1
        # If the odor is not switched, set the target valence
        else:
            time_CS1[b, stim_inds2] = 1
            if r_ext[b, 0] == 1:
                vt_opt[b, (stim_inds2 + 1)] = 1
            else:
                vt_opt[b, (stim_inds2 + 1)] = -1

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


def continual_trial(t_len, st_times, st_len, r_in, n_batch, dt, **kwargs):
    """ Runs a continual learning trial (two CS- and two CS+).

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
    """

    # Draw the CS presentation times from a Poisson distribution
    st_times, st_len = gen_cont_times(n_batch, dt, **kwargs)
    n_odor = len(st_times)

    # Set odors and context signals
    r_kc, r_ext0 = r_in
    r_kc0 = r_kc[0]
    # Define the number of Kenyon cells (n_kc) and dim of context (n_ext)
    n_kc = r_kc0.shape[1]
    n_ext = r_ext0.shape[1]
    n_ones = r_kc0[0, :].sum().int()
    # Initialize lists to store additional odors and their context signals
    r_kcs = [torch.tensor([0])] * n_odor
    r_exts = [torch.tensor([0])] * n_odor
    # The first US is appetitive, while the second is aversive
    r_exts[0] = torch.tensor([1, 0]).repeat(n_batch, 1)
    r_exts[1] = torch.tensor([0, 1]).repeat(n_batch, 1)

    # Initialize activity matrices
    r_kct = torch.zeros(n_batch, n_kc, t_len)
    r_extt = torch.zeros(n_batch, n_ext, t_len)

    # Initialize stimulus time matrices
    # time_CS = torch.zeros(n_odor, n_batch, t_len)
    # time_US = torch.zeros_like(time_CS)
    time_all_CS = []
    time_all_US = []
    vt_opt = torch.zeros(n_batch, t_len)

    # Set the stimulus step inputs
    for i in range(n_odor):
        # Initialize time matrices
        time_CS = torch.zeros(n_batch, t_len)
        time_US = torch.zeros_like(time_CS)
        append_US = False
        # Set neutral stimulus contexts
        r_kcs[i] = torch.zeros_like(r_kc0)
        if i > 1:
            r_exts[i] = torch.tensor([0, 0]).repeat(n_batch, 1)

        for b in range(n_batch):
            # Generate odors
            # new_kc_inds = torch.multinomial(torch.ones(n_kc), n_kc)
            # r_kcs[i][b, :] = r_kc0[b, new_kc_inds]
            new_kc_inds = torch.multinomial(torch.ones(n_kc), n_ones)
            r_kcs[i][b, new_kc_inds] = 1

            for j, st in enumerate(st_times[i][b]):
                # Convert stimulus time into range of indices
                stim_inds = st + torch.arange(st_len)
                # Set the CS input times
                time_CS[b, stim_inds] = 1

                if i < (n_odor / 2):
                    # Set the US input times
                    time_US[b, (stim_inds + st_len)] = 1
                    append_US = True
                    # Set target valence on each presentation after the first
                    if j > 0:
                        if r_exts[i][b, 0] == 1:
                            vt_opt[b, (stim_inds + 1)] = 1
                        else:
                            vt_opt[b, (stim_inds + 1)] = -1

        # Calculate the input neuron activity time series (KC = CS, ext = US)
        r_kct += torch.einsum('bm, mbt -> bmt', r_kcs[i],
                              time_CS.repeat(n_kc, 1, 1))
        r_extt += torch.einsum('bm, mbt -> bmt', r_exts[i],
                               time_US.repeat(n_ext, 1, 1))

        # Save the CS and US stimuli time series
        # time_all_CS[i, :, :] = time_CS
        # time_all_US[i, :, :] = time_US
        time_all_CS.append(time_CS)
        if append_US:
            time_all_US.append(time_US)

    # Combine the time matrices into a list
    stim_ls = [time_all_CS, time_all_US]
    r_next = (r_kcs, r_exts)

    return r_next, r_kct, r_extt, stim_ls, vt_opt


def gen_int_times(n_batch, dt, T_stim, T_range=(5, 15), **kwargs):
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
    max_ind = int((T_range[1] - 2 * T_stim) / dt)
    st_times = torch.randint(min_ind, max_ind, (n_batch,))
    st_len = int(T_stim / dt)

    return st_times, st_len


def gen_cont_times(n_batch, dt, T_stim=2, T_int=200, stim_mean=2, n_odor=4,
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
    st_times = [torch.tensor([0])] * n_odor
    st_len = int(T_stim / dt)

    # Generate a list of stimulus presentation times for each trial
    for i in range(n_odor):
        batch_times = [torch.tensor([0])] * n_batch
        for b in range(n_batch):
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
            batch_times[b] = torch.stack(trial_times)
        st_times[i] = batch_times

    return st_times, st_len
