# Import the required packages
import torch
import torch.nn.functional as F


def cond_err(vt, vt_opt):
    """ Calculates the readout error for conditioning tasks.

    Only accounts for error due to difference between target (optimal) readout
    and the MBON readout scalar.

    Parameters
        vt = time dependent valence output of network
        vt_opt = target valence (must be a torch tensor)

    Returns
        vt_loss = scalar readout error used in evaluation
    """

    # Calculate the MSE loss of the valence
    vt_loss = torch.mean((vt - vt_opt) ** 2)

    return vt_loss


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

    # Calculate the MSE loss of the valence (scalar)
    vt_loss = cond_err(vt, vt_opt)

    # Calculate DAN activity regularization term (scalar)
    rt_sum = torch.sum(F.relu(r_DAN - DAN_baseline) ** 2, dim=1)
    rt_loss = torch.mean(rt_sum) * lam

    # Calculate the combined loss
    loss = vt_loss + rt_loss

    return loss


def int_cond_cs(t_len, st_times, st_len, r_in, n_batch):
    """ Runs a first-order conditioning interval (CS+ and US).

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (CS+ = r_kc and US = r_ext)
        n_batch = number of trials in batch

    Returns
        r_kct = Kenyon cell (odor) activity time series
        r_extt = contextual signal time series
        stim_ls = conditioned (CS) and unconditioned (US) stimulus time series
        vt_opt = target readout for interval
    """

    # Set odors and context signals for this interval
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
        # Set the US input times
        time_US[b, (stim_inds + st_len)] = 1

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    stim_ls = [time_CS, time_US]

    return r_kct, r_extt, stim_ls, vt_opt


def int_cs_alone(t_len, st_times, st_len, r_in, n_batch):
    """ Runs a first-order conditioning interval (CS- alone).

    This function can be used for both the conditioning and test intervals
    of CS- trials. Only the CS stimulus is presented.

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (CS- = r_kc, US = r_ext = zeros)
        n_batch = number of trials in batch

    Returns
        r_kct = Kenyon cell (odor) activity time series
        r_extt = contextual signal time series
        stim_ls = conditioned (CS) and unconditioned (US) stimulus time series
        vt_opt = target readout for interval
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

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    stim_ls = [time_CS, time_US]

    return r_kct, r_extt, stim_ls, vt_opt


def int_test_cs(t_len, st_times, st_len, r_in, n_batch, f_tar=1.0):
    """ Runs a conditioning test interval (CS+ and target valence).

    This function can be used for both first- and second-order tests, extinction
    tests as well as extinction conditioning.

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (CS+ = r_kc, US = r_ext = zeros)
        n_batch = number of trials in batch
        f_tar = relative target valence magnitude (used for extinction tests)

    Returns
        r_kct = Kenyon cell (odor) activity time series
        r_extt = contextual signal time series
        stim_ls = conditioned (CS) and unconditioned (US) stimulus time series
        vt_opt = target readout for interval
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
        # Set the target valence
        if r_ext[b, 0] == 1:
            vt_opt[b, (stim_inds + 1)] = 1 * f_tar
        else:
            vt_opt[b, (stim_inds + 1)] = -1 * f_tar

    # Calculate the input neurons' activity time series (KC = CS, ext = US)
    r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                         time_CS.repeat(n_kc, 1, 1))
    r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                          time_US.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    stim_ls = [time_CS, time_US]

    return r_kct, r_extt, stim_ls, vt_opt


def int_cond_cs2(t_len, st_times, st_len, r_in, n_batch):
    """ Runs a second-order conditioning interval (CS2, CS1 and target valence).

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations
                (CS1 = r_kcs[0], CS2 = r_kcs[1], US = r_ext = zeros)
        n_batch = number of trials in batch

    Returns
        r_kct = Kenyon cell (odor) activity time series
        r_extt = contextual signal time series
        stim_ls = conditioned (CS) and unconditioned (US) stimulus time series
        vt_opt = target readout for interval
    """

    # Set odors and context signals for each trial
    r_kcs, r_ext = r_in
    # The input odor is the second odor presented (CS1)
    r_kc1 = r_kcs[0]
    r_kc2 = r_kcs[1]
    n_kc = r_kc1.shape[1]
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
        # Set the CS1 input times
        time_CS1[b, (stim_inds + st_len)] = 1
        # Set the CS2 input times
        time_CS2[b, stim_inds] = 1
        # Set the target valence
        # vt_opt[b, (stim_inds + st_len + 1)] = 1
        if r_ext[b, 0] == 1:
            vt_opt[b, (stim_inds + st_len + 1)] = 1
        else:
            vt_opt[b, (stim_inds + st_len + 1)] = -1

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

    return r_kct, r_extt, stim_ls, vt_opt


def tr_no_plasticity(t_len, st_times, st_len, r_in, n_batch, csm=False):
    """ Runs a trial without KC->MBON weight plasticity.

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trials in batch
        csm = indicates whether to run a control (CS-) trial
            True: present random (novel) odour during second interval
            False: present CS+ in both intervals
    """

    # Set odors and context signals
    r_kc, r_ext = r_in
    # Define the number of Kenyon cells (n_kc) and dim of context (n_ext)
    n_kc = r_kc.shape[1]
    n_ext = r_ext.shape[1]

    # Initialize activity matrices
    r_kct = torch.zeros(n_batch, n_kc, t_len)
    r_extt = torch.zeros(n_batch, n_ext, t_len)

    # Initialize stimulus time matrices
    vt_opt = torch.zeros(n_batch, t_len)
    time_CSp = torch.zeros_like(vt_opt)
    time_CSm = torch.zeros_like(vt_opt)
    time_US = torch.zeros_like(vt_opt)

    # For each stimulus presentation
    for i in range(2):
        # Initialize time matrices
        time_CS_int = torch.zeros_like(vt_opt)
        time_US_int = torch.zeros_like(vt_opt)

        for b in range(n_batch):
            stim_inds = st_times[i][b] + torch.arange(st_len)
            # Set the CS time
            time_CS_int[b, stim_inds] = 1
            # Set the CS+ and US time
            if i == 0:
                time_CSp[b, stim_inds] = 1
                time_US_int[b, stim_inds + st_len] = 1
            # Set the CS+/CS2 and target valence times
            if i == 1:
                # Switch the odor in half the trials (target valence is zero)
                if csm:
                    CSm_inds = torch.multinomial(torch.ones(n_kc), n_kc)
                    r_kc[b, :] = r_kc[b, CSm_inds]
                    r_ext[b, :] = 0
                    time_CSm[b, stim_inds] = 1
                # If the odor is not switched, set the target valence
                else:
                    time_CSp[b, stim_inds] = 1
                    if r_ext[b, 0] == 1:
                        vt_opt[b, (stim_inds + 1)] = 1
                    elif r_ext[b, 1] == 1:
                        vt_opt[b, (stim_inds + 1)] = -1

        # Calculate the stimulus time series (KC = CS, ext = US)
        r_kct += torch.einsum('bm, mbt -> bmt', r_kc,
                              time_CS_int.repeat(n_kc, 1, 1))
        r_extt += torch.einsum('bm, mbt -> bmt', r_ext,
                               time_US_int.repeat(n_ext, 1, 1))

    # Combine the time matrices into a list
    if csm:
        time_all_CS = [time_CSp, time_CSm]
    else:
        time_all_CS = time_CSp
    stim_ls = [time_all_CS, time_US]

    return r_kct, r_extt, stim_ls, vt_opt


def tr_continual(t_len, st_times, st_len, r_in, n_batch):
    """ Runs a continual learning trial (two CS- and two CS+).

    Parameters
        t_len = length of task interval (in indices)
        st_times = array of stimulus presentation times for each trial
        stim_len = length of stimulus presentation (in indices)
        r_in = input neuron activations (r_kc and r_ext)
        n_batch = number of trialsclassic_net in batch
    """

    # Draw the CS presentation times from a Poisson distribution
    # st_times, st_len = gen_cont_times(n_batch, dt, **kwargs)
    # n_odor = len(st_times)

    # Set odors and context signals
    r_kcs, r_exts = r_in
    n_odor = len(r_kcs)
    # Define the number of Kenyon cells (n_kc) and dim of context (n_ext)
    n_kc = r_kcs[0].shape[1]
    n_ext = r_exts[0].shape[1]

    # Initialize activity matrices
    r_kct = torch.zeros(n_batch, n_kc, t_len)
    r_extt = torch.zeros(n_batch, n_ext, t_len)

    # Initialize stimulus time matrices
    time_all_CS = []
    time_all_US = []
    vt_opt = torch.zeros(n_batch, t_len)

    # Set the stimulus step inputs
    for i in range(n_odor):
        # Initialize time matrices
        time_CS = torch.zeros(n_batch, t_len)
        time_US = torch.zeros_like(time_CS)
        append_US = False

        for b in range(n_batch):
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
        time_all_CS.append(time_CS)
        if append_US:
            time_all_US.append(time_US)

    # Combine the time matrices into a list
    stim_ls = [time_all_CS, time_all_US]

    return r_kct, r_extt, stim_ls, vt_opt


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
