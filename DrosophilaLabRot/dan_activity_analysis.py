# Import the required packages
import os
from network_classes.all_conditioning_rnn import ExtendedCondRNN
from common.common import *
from common.plotting import *


def csp_csm_cs2(net, W_in, T_vars, n_batch, **kwargs):
    """ Runs a CS+, test, CS-, test, CS2, test trial.

    Parameters
        net = trained network to evaluate
        W_in = initial weights to the trial
        T_vars: Tuple
            T_vars[0] = T_int = length of trial (in seconds)
            T_vars[1] = T_stim = length of time each stimulus is presented
            T_vars[2] = dt = time step of simulations
        n_batch = number of trials in mini-batch

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
    # time_zeros = torch.zeros(n_batch, t_len)

    # Generate odors and context (odor = KC = CS, context = ext = US)
    r_kc_cs1, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)  # CS1+ odor
    r_kc_cs2, _ = net.gen_r_kc_ext(n_batch, **kwargs)  # CS2+ odor
    r_kc_csm, _ = net.gen_r_kc_ext(n_batch, **kwargs)  # CS- odor
    r_kc_novel, _ = net.gen_r_kc_ext(n_batch, **kwargs)  # novel odor
    r_ext_zeros = torch.zeros_like(r_ext)
    trial_odors = [r_kc_cs1, r_kc_cs2, r_kc_csm, r_kc_novel]
    int_list = [int_cond_cs,
                int_test_cs,
                int_cs_alone,
                int_cs_alone,
                int_cond_cs2,
                int_test_cs]
    n_int = len(int_list)
    r_in_list = [(r_kc_cs1, r_ext),
                 (r_kc_cs1, r_ext),
                 (r_kc_csm, r_ext_zeros),
                 (r_kc_csm, r_ext_zeros),
                 ([r_kc_cs1, r_kc_cs2], r_ext),
                 (r_kc_cs2, r_ext)]
    # wt_plast = [True, False, True, False, True, False]
    wt_plast = [True] * n_int

    # Lists to store activities, weights, readouts and target valences
    # In this example, everything is saved
    rts = []
    Wts = []
    wts = []
    vts = []
    vt_opts = []
    time_CS1 = torch.zeros(n_batch, t_len * n_int)
    time_CS2 = torch.zeros_like(time_CS1)
    time_CSm = torch.zeros_like(time_CS1)
    time_CS_novel = torch.zeros_like(time_CS1)
    time_US = torch.zeros_like(time_CS1)

    # Iniitialize activity input
    rt_in = None
    # Initialize weights
    Wt_init, wt_init = net.init_w_kc_mbon(W_in, n_batch, (0, 1))
    Wts += Wt_init
    wts += wt_init

    for i in range(n_int):
        # Calculate the CS stimulus presentation times
        st_times, st_len = gen_int_times(n_batch, dt, T_stim, **kwargs)
        # Calculate the interval inputs for a CS+ conditioning interval
        int_fnc = int_list[i]
        r_in = r_in_list[i]
        f_in = int_fnc(t_len, st_times, st_len, r_in, n_batch)
        r_kct, r_extt, stim_ls, vt_opt = f_in

        # Run the forward pass
        net_out = net(r_kct, r_extt, time_int, n_batch, W_in, rt_in,
                      wt_plast[i], **kwargs)
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
        if i in [0, 1]:
            time_CS1[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        elif i in [2, 3]:
            time_CSm[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        elif i == 4:
            time_CS1[:, i * t_len:(i + 1) * t_len] = stim_ls[0][0]
            time_CS2[:, i * t_len:(i + 1) * t_len] = stim_ls[0][1]
        elif i == 5:
            time_CS2[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        # Store US time series
        time_US[:, i * t_len:(i + 1) * t_len] = stim_ls[1]

    # Save stimuli time series
    time_all_CS = [time_CS1, time_CS2, time_CSm]
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

net_fname = 'second_order_no_extinct_5000ep_1hop_N01.pt'
fname = net_path + net_fname
network.load_state_dict(torch.load(fname))

# Evaluate the network
network.run_eval(csp_csm_cs2, pos_vt=True)
# Pull the data
Wt_kc_mbon = network.eval_Wts[-1].numpy().squeeze()
rt = network.eval_rts[-1].numpy().squeeze()
vt = network.eval_vts[-1].numpy().squeeze()
vt_opt = network.eval_vt_opts[-1].numpy().squeeze()
CS_list = network.eval_CS_stim[-1]
US_list = network.eval_US_stim[-1]
plot_time = np.arange(US_list[0].numpy().squeeze().size) * network.dt

# Sort the MBON->DAN weights by MBON readout strength
mbon_sorted = np.argsort(network.W_readout.detach().numpy())[0]

# Plot the trial
# plt_fname = 'knockout_so_5000ep'
plt_ttl = 'Trial Intervals: CS+, Test, CS-, Test, CS2, Test'
CS_lbls = ['CS1+', 'CS2+', 'CS-']
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
for i, n in enumerate(mbon_sorted):
    ax2.plot(plot_time, (rt[n, :] / mbon_max) + i + 1, '-k')
    ax2.plot(plot_time, (rt[(n - n_mbon), :] / dan_max) + i + n_mbon + 2, '-k')
    # ax2.plot(plot_time, np.ones(plot_time.size) * (i + n_mbon + 1), '#808080',
    #          ls='--', lw=1)
ax2.plot(plot_time, np.mean(rt[:n_mbon, :] / mbon_max, axis=0), '-r')
ax2.plot(plot_time, np.mean(rt[n_mbon:, :] / dan_max, axis=0) + n_mbon + 1, '-r')
for i in range(2 * (n_mbon + 1)):
    ax2.plot(plot_time, np.zeros(plot_time.size) + i, '#808080',
             ls='--', lw=1)
ax2.set_xlabel('Time', fontsize=label_font)
# ax2.set_ylabel('Normalized MBON Activity', fontsize=label_font)
ax2.set_yticks([0, 4.5, 9, 13.5])
ax2.set_yticklabels(['Avg MBON', 'MBONs', 'Avg DAN', 'DANs'], rotation='vertical',
                    verticalalignment='center')
fig.tight_layout()
plt.close()
# plt.show()

# For each of the 20 networks
n_nets = 20
avg_mbon_dan_wt = np.zeros((n_nets, n_mbon))
avg_abs_mbon_dan_wt = np.zeros((n_nets, n_mbon))
avg_mbon_rt = np.zeros((n_nets, n_mbon))
mbon_readout_wt = np.zeros((n_nets, n_mbon))
all_mbon_dan_wt = np.zeros((n_nets, n_mbon, n_mbon))
all_kc_mbon_Wt = np.zeros((n_nets, n_mbon, Wt_kc_mbon.shape[1],
                       Wt_kc_mbon.shape[2]))
n_stats = 3
avg_mbon_stats = np.zeros((n_stats, n_nets, n_mbon))
for i in range(n_nets):
    # Load the minimum network
    network = ExtendedCondRNN(n_mbon=n_mbon, n_hop=n_hop)

    net_fname = 'second_order_no_extinct_5000ep_1hop_N{}.pt'\
        .format(str(i + 1).zfill(2))
    fname = net_path + net_fname
    network.load_state_dict(torch.load(fname))

    # Evaluate the network
    network.run_eval(csp_csm_cs2, pos_vt=True)

    # Pull the data
    W_mbon_to_dan = network.W_recur[:n_mbon, -n_mbon:].detach().numpy()
    avg_mbon_dan_wt[i, :] = np.mean(W_mbon_to_dan, axis=0)
    avg_abs_mbon_dan_wt[i, :] = np.mean(abs(W_mbon_to_dan), axis=0)
    rt = network.eval_rts[-1].numpy().squeeze()
    avg_mbon_rt[i, :] = np.mean(rt[:n_mbon, :], axis=1)
    mbon_readout_wt[i, :] = network.W_readout.detach().numpy()
    all_mbon_dan_wt[i, :, :] = W_mbon_to_dan
    all_kc_mbon_Wt[i, :, :, :] = network.eval_Wts[-1].numpy().squeeze()
    avg_mbon_stats[0, i, :] = np.mean(W_mbon_to_dan, axis=0)
    avg_mbon_stats[1, i, :] = np.mean(abs(W_mbon_to_dan), axis=0)
    avg_mbon_stats[2, i, :] = np.mean(rt[:n_mbon, :], axis=1)

# Plot the comparisons
mbon_ttls = ['Avg MBON->DAN Wt', 'Avg MBON->DAN |Wt|',
             'Avg MBON rt']
cmap_list = ['coolwarm', 'Reds', 'Reds']
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 12), sharey=True)
# avg_wt_plt = ax1.matshow(avg_mbon_dan_wt, cmap='coolwarm')
# ax1.set_xlabel('MBON', fontsize=label_font)
# ax1.set_ylabel('Network', fontsize=label_font)
# fig.colorbar(avg_wt_plt, ax=ax1)
# avg_abs_wt_plt = ax2.matshow(avg_abs_mbon_dan_wt, cmap='Reds')
# ax2.set_xlabel('MBON', fontsize=label_font)
# fig.colorbar(avg_abs_wt_plt, ax=ax2)
# avg_rt_plt = ax3.matshow(avg_mbon_rt, cmap='Reds')
# ax3.set_xlabel('MBON', fontsize=label_font)
# fig.colorbar(avg_rt_plt, ax=ax3)
fig, axes = plt.subplots(1, 3, figsize=(15, 12), sharey=True)
for i in range(n_stats):
    plot_ax = axes[i].matshow(avg_mbon_stats[i], cmap=cmap_list[i])
    axes[i].set_xlabel('MBON', fontsize=label_font)
    axes[i].set_title(mbon_ttls[i], fontsize=label_font)
    fig.colorbar(plot_ax, ax=axes[i])
axes[0].set_ylabel('Network', fontsize=label_font)
plt.close()

# Calculate the correlations
flat1 = avg_mbon_dan_wt.flatten()
flat2 = avg_abs_mbon_dan_wt.flatten()
flat3 = avg_mbon_rt.flatten()
flat4 = mbon_readout_wt.flatten()
data_mat = np.stack((flat1, flat2, flat3, flat4))
corr_mat = np.corrcoef(data_mat)

# Plot the correlation matrix
corr_ttls = ['Avg Wt', 'Avg |Wt|', 'Avg rt', 'Out Wt']
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_ax = ax.matshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr_ttls)))
ax.set_xticklabels(corr_ttls)
ax.set_yticks(np.arange(n_stats))
ax.set_yticklabels(corr_ttls, rotation='vertical')
fig.colorbar(plot_ax, ax=ax)
plt.close()

# Plot the KC->MBON weights (all_kc_mbon_Wt)
# print(all_kc_mbon_Wt.shape)
# kc_mbon_Wt_diff_N1 = np.diff(all_kc_mbon_Wt[0], axis=-1)
# print(kc_mbon_Wt_diff_N1.shape)
#
# n_diffs = kc_mbon_Wt_diff_N1.shape[-1]
# fig, axes = plt.subplots(1, n_diffs, figsize=(8, 12))
# for i in range(n_diffs):
#     axes[i].matshow(kc_mbon_Wt_diff_N1[:, :, i].T, cmap='coolwarm',
#                     vmin=-0.05, vmax=0.05)

# Do MBONs that store the odor information affect plasticity especially?
# Average activity across all KCs
avg_kc_mbon_Wt = np.mean(all_kc_mbon_Wt, axis=2)
avg_kc_mbon_Wt_diff = np.mean(np.diff(all_kc_mbon_Wt, axis=-1), axis=-1)
avg_mbon_dan_wt = np.mean(all_mbon_dan_wt, axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.matshow(avg_mbon_dan_wt, cmap='coolwarm')
ax2.matshow(avg_kc_mbon_Wt[:, :, 1], cmap='coolwarm')
plt.close()

wt_shp = avg_kc_mbon_Wt.shape
corr_data = np.zeros((wt_shp[2], wt_shp[0] * wt_shp[1]))
corr_data[0, :] = avg_mbon_dan_wt.flatten()
for i in range(wt_shp[2] - 1):
    # corr_data[(i + 1), :] = avg_kc_mbon_Wt[:, :, (i + 1)].flatten()
    corr_data[(i + 1), :] = avg_kc_mbon_Wt_diff[:, :, i].flatten()
corr_matt = np.corrcoef(corr_data)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
corr_plot = ax.matshow(corr_matt, cmap='Reds', vmin=0)
fig.colorbar(corr_plot, ax=ax)
# plt.close()
# This plot indicates the memory trace is moving (I guess that's something)

plt.show()
