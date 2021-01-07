# Import the required packages
import os
from network_classes.all_conditioning_rnn import ExtendedCondRNN
from common.common import *
from common.plotting import *


def imprint_false(net, W_in, T_vars, n_batch, n_act_dan, **kwargs):
    """ Imprints a false memory on the network, then tests the effects.

    A false memory is imprinted by artificially activating (using "act_dan"
    parameter of the network's forward function) a particular DAN neuron during
    presentation of an stimulus (without unconditioned stimulus pairing). Then,
    a novel stimulus is presented (with plasticity turned off) to test the
    effects of knocking out each MBON (using the "ko_wts" parameter of the
    network's forward function) on the DAN activity.

    Parameters
        net = trained network to evaluate
        W_in = initial weights to the trial
        T_vars: Tuple
            T_vars[0] = T_int = length of trial (in seconds)
            T_vars[1] = T_stim = length of time each stimulus is presented
            T_vars[2] = dt = time step of simulations
        n_batch = number of trials in mini-batch
        n_act_dan = index of DAN to be artificially activated

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

    # Generate odors and context (odor = KC = CS, context = ext = US)
    r_kc, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)
    r_kc_novel, _ = net.gen_r_kc_ext(n_batch, **kwargs)
    r_ext_zeros = torch.zeros_like(r_ext)
    n_int = net.n_mbon + 3
    trial_odors = [r_kc, r_kc_novel]

    # Lists to store activities, weights, readouts and target valences
    # In this example, everything is saved
    rts = []
    Wts = []
    wts = []
    vts = []
    vt_opts = []
    time_CS = torch.zeros(n_batch, t_len * n_int)
    time_CS_novel = torch.zeros_like(time_CS)
    time_US = torch.zeros_like(time_CS)

    # Iniitialize activity input
    rt_in = None
    # Initialize weights
    Wt_init, wt_init = net.init_w_kc_mbon(W_in, n_batch, (0, 1))
    Wts += Wt_init
    wts += wt_init

    # Set the parameters for the first interval
    T_range = (5, 15)
    r_in = (r_kc, r_ext_zeros)
    ud_wts = True
    ko_wts = []
    act_dan = [n_act_dan]

    # Run the first set of intervals (imprint memory and test MBONs)
    for i in range(n_int):
        if i > 0:
            ud_wts = False
            act_dan = []
        if i > 1:
            T_stim = T_int
            T_range = (0, 2 * T_stim + dt)
            r_in = (r_kc_novel, r_ext_zeros)
        if i > 2:
            ko_wts = [i - 3]

        # Calculate the CS stimulus presentation times
        st_times, st_len = gen_int_times(n_batch, dt, T_stim, T_range, **kwargs)
        # Calculate the interval inputs for a CS+ conditioning interval
        if i == 0:
            f_in = int_cond_cs(t_len, st_times, st_len, r_in, n_batch)
        elif i == 1:
            f_in = int_test_cs(t_len, st_times, st_len, r_in, n_batch)
        else:
            f_in = int_cs_alone(t_len, st_times, st_len, r_in, n_batch)
        r_kct, r_extt, stim_ls, vt_opt = f_in

        # Run the forward pass
        net_out = net(r_kct, r_extt, time_int, n_batch, W_in, rt_in, ud_wts,
                      ko_wts, act_dan, **kwargs)
        rt_int, (Wt_int, wt_int), vt_int = net_out
        # Pass the KC->MBON weights to the next interval
        W_in = (Wt_int[-1], wt_int[-1])
        # Pass the neuron activities to the next interval
        rt_in = rt_int[-1]

        # Append the interval outputs to lists
        rts += rt_int
        Wts += Wt_int[-1]
        wts += wt_int[-1]
        vts += vt_int
        vt_opts.append(vt_opt)

        # Store the stimuli time series
        if i > 1:
            time_CS_novel[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        else:
            time_CS[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        # Store US time series
        # time_US[:, i * t_len:(i + 1) * t_len] = stim_ls[1]

    # Save stimuli time series
    time_all_CS = [time_CS, time_CS_novel]
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

    return rts_trial, Wts_trial, wts_trial, vt_trial, vt_opt_trial, err_trial,\
        trial_odors, stim_list


def consolidate_false(net, W_in, T_vars, n_batch, n_cons, **kwargs):
    """ Imprints a false memory on the network, then tests the effects.

    A false memory is imprinted by artificially activating (using "act_dan"
    parameter of the network's forward function) a particular DAN neuron during
    presentation of an stimulus (without unconditioned stimulus pairing). Then,
    a novel stimulus is presented (with plasticity turned off) to test the
    effects of knocking out each MBON (using the "ko_wts" parameter of the
    network's forward function) on the DAN activity.

    Parameters
        net = trained network to evaluate
        W_in = initial weights to the trial
        T_vars: Tuple
            T_vars[0] = T_int = length of trial (in seconds)
            T_vars[1] = T_stim = length of time each stimulus is presented
            T_vars[2] = dt = time step of simulations
        n_batch = number of trials in mini-batch
        n_cons = number of consolidation intervals

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

    # Generate odors and context (odor = KC = CS, context = ext = US)
    r_kc_novel, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)
    r_kc = net.eval_odors[-1][0]
    r_ext_zeros = torch.zeros_like(r_ext)
    n_int = net.n_mbon + n_cons + 1
    trial_odors = [r_kc, r_kc_novel]

    # Lists to store activities, weights, readouts and target valences
    # In this example, everything is saved
    rts = []
    Wts = []
    wts = []
    vts = []
    vt_opts = []
    time_CS = torch.zeros(n_batch, t_len * n_int)
    time_CS_novel = torch.zeros_like(time_CS)
    time_US = torch.zeros_like(time_CS)

    # Iniitialize activity input
    rt_in = None
    # Initialize weights
    Wt_init, wt_init = net.init_w_kc_mbon(W_in, n_batch, (0, 1))
    Wts += Wt_init
    wts += wt_init

    # Set the parameters for the first interval
    T_range = (5, 15)
    r_in = ([r_kc, r_kc], r_ext_zeros)
    # r_in = (r_kc, r_ext_zeros)
    ud_wts = True
    ko_wts = []

    # Run the first set of intervals (imprint memory and test MBONs)
    for i in range(n_int):
        if i > (n_cons - 1):
            T_stim = T_int
            T_range = (0, 2 * T_stim + dt)
            r_in = (r_kc_novel, r_ext_zeros)
            ud_wts = False
        if i > n_cons:
            ko_wts = [i - n_cons - 1]

        # Calculate the CS stimulus presentation times
        st_times, st_len = gen_int_times(n_batch, dt, T_stim, T_range, **kwargs)
        # Calculate the interval inputs for a CS+ conditioning interval
        if i > (n_cons - 1):
            f_in = int_cs_alone(t_len, st_times, st_len, r_in, n_batch)
        else:
            f_in = int_cond_cs2(t_len, st_times, st_len, r_in, n_batch)
            # f_in = int_test_cs(t_len, st_times, st_len, r_in, n_batch)
        r_kct, r_extt, stim_ls, vt_opt = f_in

        # Run the forward pass
        net_out = net(r_kct, r_extt, time_int, n_batch, W_in, rt_in, ud_wts,
                      ko_wts, **kwargs)
        rt_int, (Wt_int, wt_int), vt_int = net_out
        # Pass the KC->MBON weights to the next interval
        W_in = (Wt_int[-1], wt_int[-1])
        # Pass the neuron activities to the next interval
        rt_in = rt_int[-1]

        # Append the interval outputs to lists
        rts += rt_int
        Wts += Wt_int[-1]
        wts += wt_int[-1]
        vts += vt_int
        vt_opts.append(vt_opt)

        # Store the stimuli time series
        if i > (n_cons - 1):
            time_CS_novel[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        else:
            # time_CS[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
            time_CS[:, i * t_len:(i + 1) * t_len] = stim_ls[0][0]
            time_CS[:, i * t_len:(i + 1) * t_len] += stim_ls[0][1]
        # Store US time series
        time_US[:, i * t_len:(i + 1) * t_len] = stim_ls[1]

    # Save stimuli time series
    time_all_CS = [time_CS, time_CS_novel]
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

    return rts_trial, Wts_trial, wts_trial, vt_trial, vt_opt_trial, err_trial,\
        trial_odors, stim_list


# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/mbon_sensitivity/second_order_only_nets/' \
                      'trained_nets/08_mbons/'

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# Network parameters
n_mbon = 8
n_hop = 1

# Load the minimum network
network = ExtendedCondRNN(n_mbon=n_mbon, n_hop=n_hop)
net_fname = 'second_order_no_extinct_5000ep_1hop_N02.pt'
fname = net_path + net_fname
network.load_state_dict(torch.load(fname))

# Identify the most aversive MBON
sorted_mbons = np.argsort(network.W_readout.detach().numpy().squeeze())
averse_mbon = sorted_mbons[-1]
# sorted_ext_wts = np.argsort(network.W_ext[:, 1].detach().numpy().squeeze())
# # averse_mbon = sorted_ext_wts[0]
# print(network.W_readout[0, averse_mbon])
# print(network.W_readout[0, sorted_mbons])
# print(network.W_ext[sorted_mbons, :])
# print(network.W_readout[0, sorted_ext_wts])
# print(network.W_ext[sorted_ext_wts, :])
# TODO: DANs corresponding to MBONs with positive weights form averse memories
#  when artificially activated, and vice versa
# TODO: doesn't appear as if W_ext that correspond to aversive signal is
#  correlate with the MBONs of W_readout that correspond to an aversive readout

# Imprint a false memory in the network, record from DANs
network.run_eval(imprint_false, n_act_dan=averse_mbon, pos_vt=False)

# Identify which DAN was most active during de-activations
# Length of trial in indices
tr_len = int(network.T_int / network.dt + 1)
# Pull the data
Wt_kc_mbon = network.eval_Wts[-1].numpy().squeeze()
rt = network.eval_rts[-1].numpy().squeeze()
vt = network.eval_vts[-1].numpy().squeeze()
vt_opt = network.eval_vt_opts[-1].numpy().squeeze()
CS_list = network.eval_CS_stim[-1]
US_list = network.eval_US_stim[-1]

# Create a matrix representation and plot
# DAN Activity
rt_dans = rt[-n_mbon:, :]
ss_dan_imp = rt_dans[sorted_mbons, (3 * tr_len - 1)::tr_len]
# MBON Activity
rt_mbons = rt[:n_mbon, :]
ss_mbon_imp = rt_mbons[sorted_mbons, (3 * tr_len - 1)::tr_len]
# KC->MBON Weights
print(network.W_readout[0, sorted_mbons])
avg_kc_mbon_diff_01 = np.mean(np.diff(Wt_kc_mbon[:, :, :2]),
                              axis=1)[sorted_mbons].T
# print(Wt_kc_mbon.shape)
kc_mbon_diff_01 = np.diff(Wt_kc_mbon[sorted_mbons, :, :2]).T.squeeze()
# print(avg_kc_mbon_diff_01)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.imshow(kc_mbon_diff_01, cmap='coolwarm', origin='lower')
ax.imshow(ss_dan_imp, cmap='coolwarm', origin='lower')
# ax.set_xlabel('MBON (Sorted)', fontsize=label_font)
ax.set_xlabel('Deactivated MBON (Sorted)', fontsize=label_font)
ax.set_xticks(np.arange(ss_dan_imp.shape[-1]))
ax.set_xticklabels(['None', '1', '2', '3', '4', '5', '6', '7', '8'])
# ax.set_ylabel('KC', fontsize=label_font)
ax.set_ylabel('DAN Activity (Sorted)', fontsize=label_font)
ax.set_yticks(np.arange(ss_dan_imp.shape[0]))
ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
plt.close()

# Plot the trial
plot_time = np.arange(US_list[0].numpy().squeeze().size) * network.dt
plt_ttl = 'Trial Intervals: CS+, Test, CS-, Test, CS2, Test'
CS_lbls = ['CS+', 'CS-']
US_lbls = ['US']
# Plot the conditioning and test
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                               gridspec_kw={'height_ratios': [1, 3]})
ax1.plot(plot_time, vt, label='Readout')
ax1.plot(plot_time, vt_opt, label='Target')
for i in range(len(CS_lbls)):
    ax1.plot(plot_time, CS_list[i].squeeze(), label='{}'.format(CS_lbls[i]))
for i in range(len(US_lbls)):
    ax1.plot(plot_time, US_list[i].squeeze(), label='{}'.format(US_lbls[i]))
ax1.set_ylabel('Value', fontsize=label_font)
ax1.set_title(plt_ttl, fontsize=title_font)
ax1.legend(fontsize=legend_font)

# Plot the activities of a few MBONs
mbon_max = np.max(rt[:n_mbon, :])
dan_max = np.max(rt[n_mbon:, :])
for i, n in enumerate(sorted_mbons):
    ax2.plot(plot_time, (rt[(n - n_mbon), :] / dan_max) + i, '-k')
ax2.set_xlabel('Time', fontsize=label_font)
ax2.set_ylabel('Normalized DAN Activity', fontsize=label_font)
ax2.set_yticks([1, 7])
ax2.set_yticklabels(['Most Negative W_readout', 'Most Positive W_readout'],
                    rotation='vertical', verticalalignment='center')
fig.tight_layout()
# plt.close()

# Consolidate and record from DANs
W_in = (torch.unsqueeze(network.eval_Wts[-1][:, :, -1], 0),
        torch.unsqueeze(network.eval_wts[-1][:, :, -1], 0))
n_cons = 6
network.run_eval(consolidate_false, W_in=W_in, pos_vt=False, n_cons=n_cons)

# Pull the data
Wt_kc_mbon = network.eval_Wts[-1].numpy().squeeze()
rt = network.eval_rts[-1].numpy().squeeze()
vt = network.eval_vts[-1].numpy().squeeze()
vt_opt = network.eval_vt_opts[-1].numpy().squeeze()
CS_list = network.eval_CS_stim[-1]
US_list = network.eval_US_stim[-1]

# Create a matrix representation and plot
rt_dans = rt[-n_mbon:, :]
ss_dan_con = rt_dans[sorted_mbons, ((n_cons + 1) * tr_len - 1)::tr_len]
rt_mbons = rt[:n_mbon, :]
ss_mbon_con = rt_mbons[sorted_mbons, ((n_cons + 1) * tr_len - 1)::tr_len]
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.imshow(ss_dan_con, cmap='coolwarm', origin='lower')
# ax.set_xlabel('Deactivated MBON (Sorted)', fontsize=label_font)
# ax.set_xticks(np.arange(ss_dan_con.shape[-1]))
# ax.set_xticklabels(['None', '1', '2', '3', '4', '5', '6', '7', '8'])
# ax.set_ylabel('DAN Activity (Sorted)', fontsize=label_font)
# ax.set_yticks(np.arange(ss_dan_con.shape[0]))
# ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
# plt.close()

# Plot the trial
plot_time = np.arange(US_list[0].numpy().squeeze().size) * network.dt
plt_ttl = 'Trial Intervals: CS+, Test, CS-, Test, CS2, Test'
CS_lbls = ['CS+', 'CS-']
US_lbls = ['US']
# Plot the conditioning and test
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                               gridspec_kw={'height_ratios': [1, 3]})
ax1.plot(plot_time, vt, label='Readout')
ax1.plot(plot_time, vt_opt, label='Target')
# Note: the number of stimuli plotted is determined by the label list
# This is determined by the output of the trial function
for i in range(len(CS_lbls)):
    ax1.plot(plot_time, CS_list[i].squeeze(), label='{}'.format(CS_lbls[i]))
for i in range(len(US_lbls)):
    ax1.plot(plot_time, US_list[i].squeeze(), label='{}'.format(US_lbls[i]))
ax1.set_ylabel('Value', fontsize=label_font)
ax1.set_title(plt_ttl, fontsize=title_font)
ax1.legend(fontsize=legend_font)

# Plot the activities of a few MBONs
mbon_max = np.max(rt[:n_mbon, :])
dan_max = np.max(rt[n_mbon:, :])
for i, n in enumerate(sorted_mbons):
    ax2.plot(plot_time, (rt[(n - n_mbon), :] / dan_max) + i, '-k')
ax2.set_xlabel('Time', fontsize=label_font)
ax2.set_ylabel('Normalized DAN Activity', fontsize=label_font)
ax2.set_yticks([1, 7])
ax2.set_yticklabels(['Most Negative W_readout', 'Most Positive W_readout'],
                    rotation='vertical', verticalalignment='center')
fig.tight_layout()
# plt.close()

# Look at changes after consolidation
# TODO: take difference of DAN activity matrices
ss_diff = ss_dan_imp - ss_dan_con
max_val = max(np.max(ss_diff), abs(np.min(ss_diff)))
max_val = 0.3
# ss_diff = ss_mbon_imp - ss_mbon_con
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
dan_diff_plot = ax.imshow(ss_diff, cmap='coolwarm', origin='lower',
                          vmin=-max_val, vmax=max_val)
ax.set_xlabel('Deactivated MBON (Sorted)', fontsize=label_font)
ax.set_xticks(np.arange(ss_diff.shape[-1]))
ax.set_xticklabels(['None', '1', '2', '3', '4', '5', '6', '7', '8'])
ax.set_ylabel('DAN Activity (Sorted)', fontsize=label_font)
ax.set_yticks(np.arange(ss_diff.shape[0]))
ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
fig.colorbar(dan_diff_plot, ax=ax)
# plt.close()

# TODO: try different scenarios (w/ and w/o US, other r_dan set to zero,
#  different values for artificial r_dan)
# TODO: why does the positive readout weight create an aversive memory?


plt.show()
