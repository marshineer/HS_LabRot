# Import the required packages
import os
from network_classes.all_conditioning_rnn import ExtendedCondRNN
from common.common import *
from common.plotting import *


def imprint_false(net, W_in, T_vars, n_batch, n_act_dan, mbon_inds, **kwargs):
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
        mbon_inds = sorted MBON indices (for knocking out MBONs in sequence)

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

    # Initialize activity input
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
        # First interval imprints memory (weights update)
        # Second interval tests whether memory is imprinted (no weight updates)
        if i > 0:
            ud_wts = False
            act_dan = []
        # Test novel stimulus with all MBONs (no weight updates)
        if i > 1:
            T_stim = T_int
            T_range = (0, 2 * T_stim + dt)
            r_in = (r_kc_novel, r_ext_zeros)
        # Test novel stimulus while knocking out each MBON individually
        if i > 2:
            # ko_wts = [i - 3]
            ko_wts = [mbon_inds[i - 3]]

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


def consolidate_false(net, W_in, T_vars, n_batch, n_cons, mbon_inds, **kwargs):
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
        mbon_inds = sorted MBON indices (for knocking out MBONs in sequence)

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
        # Run consolidation trials (weights update)
        # Test novel stimulus with all MBONs (no weight updates)
        if i >= n_cons:
            T_stim = T_int
            T_range = (0, 2 * T_stim + dt)
            r_in = (r_kc_novel, r_ext_zeros)
            ud_wts = False
        # Test novel stimulus while knocking out each MBON individually
        if i > n_cons:
            # ko_wts = [i - (n_cons + 1)]
            ko_wts = [mbon_inds[i - (n_cons + 1)]]

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
        if i >= n_cons:
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


# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# Network parameters
n_mbon = 10
n_hop = 1
rt_opt = 'r05'
net_str = '11'
save_plot = True

# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
# net_path = dir_path + '/data_store/network_compare/2nd_order_min_mbon/' \
#                       '{}_mbons/trained_nets/'.format(str(n_mbon).zfill(2))
# net_path = dir_path + '/data_store/network_compare/2nd_order_1hop_min_mbon/' \
#                       '{}_mbons/trained_nets/'.format(str(n_mbon).zfill(2))
net_path = dir_path + '/data_store/network_compare/' \
                      '2nd_order_no_extinction_1hop_min_mbon/{}_mbons/' \
                      'trained_nets/'.format(str(n_mbon).zfill(2))

# Load the minimum network
network = ExtendedCondRNN(n_mbon=n_mbon, n_hop=n_hop)
# net_fname = 'min_2nd_order_5000ep_3hop_N01.pt'
# Good nets = 2, 6, 9, 13, 15, 18
# net_fname = 'min_2nd_order_5000ep_1hop_N15.pt'
# Good nets = 4, 5, 6, 9, 10, 11, 12, 16, 19, 20
net_fname = 'min_2nd_order_only_5000ep_1hop_N{}.pt'.format(net_str)
fname = net_path + net_fname
network.load_state_dict(torch.load(fname))

# Identify the most aversive MBON
sorted_mbons = np.argsort(network.W_readout.detach().numpy().squeeze())
averse_mbon = sorted_mbons[-1]
# averse_mbon = sorted_mbons[0]
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
# TODO: try different scenarios (w/ and w/o US, other r_dan set to zero,
#  different values for artificial r_dan)
# TODO: why does the positive readout weight create an aversive memory?

# Imprint a false memory in the network, record from DANs
network.run_eval(imprint_false, n_act_dan=averse_mbon, mbon_inds=sorted_mbons,
                 pos_vt=False)

# Identify which DAN was most active during de-activations
# Length of trial in indices
tr_len = int(network.T_int / network.dt + 1)
# Pull the data
Wt_kc_mbon_imp = network.eval_Wts[-1].numpy().squeeze()
rt_imp = network.eval_rts[-1].numpy().squeeze()
vt_imp = network.eval_vts[-1].numpy().squeeze()
vt_opt_imp = network.eval_vt_opts[-1].numpy().squeeze()
CS_list_imp = network.eval_CS_stim[-1]
US_list_imp = network.eval_US_stim[-1]

# Plot the memory imprinting trial
plot_time = np.arange(US_list_imp[0].numpy().squeeze().size) * network.dt
CS_lbls = ['CS+', 'Novel']
# US_lbls = ['US']
# Plot the conditioning and test
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                               gridspec_kw={'height_ratios': [1, 3]})
ax1.plot(plot_time, vt_imp, label='Readout')
ax1.plot(plot_time, vt_opt_imp, label='Target')
for i in range(len(CS_lbls)):
    ax1.plot(plot_time, CS_list_imp[i].squeeze(), label='{}'.format(CS_lbls[i]))
# for i in range(len(US_lbls)):
#     ax1.plot(plot_time, US_list_imp[i].squeeze(), label='{}'.format(US_lbls[i]))
ax1.set_ylabel('Value', fontsize=label_font)
ax1.set_title('Imprint Memory Trial', fontsize=title_font)
ax1.legend(fontsize=legend_font, bbox_to_anchor=(1, 1.05), loc='upper left')

# Plot the activities of a few MBONs
mbon_max = np.max(rt_imp[:n_mbon, :])
dan_max = np.max(rt_imp[n_mbon:, :])
for i, n in enumerate(sorted_mbons):
    ax2.plot(plot_time, (rt_imp[(n - n_mbon), :] / dan_max) + i, '-k')
ax2.set_xlabel('Time', fontsize=label_font)
ax2.set_ylabel('Normalized DAN Activity', fontsize=label_font)
ax2.set_yticks([1, (n_mbon - 1)])
ax2.set_yticklabels(['Most Negative W_readout', 'Most Positive W_readout'],
                    rotation='vertical', verticalalignment='center')
fig.tight_layout()
# plt.close()
if save_plot:
    plot_path = dir_path + '/data_store/analysis_plots/mbon_knockout_imprint_' \
                           'trial_{}_N{}.png'.format(rt_opt, net_str)
    fig.savefig(plot_path, bbox_inches='tight')

# Consolidate and record from DANs
W_in = (torch.unsqueeze(network.eval_Wts[-1][:, :, -1], 0),
        torch.unsqueeze(network.eval_wts[-1][:, :, -1], 0))
n_cons = 4
network.run_eval(consolidate_false, W_in=W_in, pos_vt=False, n_cons=n_cons,
                 mbon_inds=sorted_mbons)

# Pull the data
Wt_kc_mbon_con = network.eval_Wts[-1].numpy().squeeze()
rt_con = network.eval_rts[-1].numpy().squeeze()
vt_con = network.eval_vts[-1].numpy().squeeze()
vt_opt_con = network.eval_vt_opts[-1].numpy().squeeze()
CS_list_con = network.eval_CS_stim[-1]
US_list_con = network.eval_US_stim[-1]

# Plot the consolidation trial
plot_time = np.arange(US_list_con[0].numpy().squeeze().size) * network.dt
CS_lbls = ['CS+', 'Novel']
# US_lbls = ['US']
# Plot the conditioning and test
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                               gridspec_kw={'height_ratios': [1, 3]})
ax1.plot(plot_time, vt_con, label='Readout')
ax1.plot(plot_time, vt_opt_con, label='Target')
# Note: the number of stimuli plotted is determined by the label list
# This is determined by the output of the trial function
for i in range(len(CS_lbls)):
    ax1.plot(plot_time, CS_list_con[i].squeeze(), label='{}'.format(CS_lbls[i]))
# for i in range(len(US_lbls)):
#     ax1.plot(plot_time, US_list_con[i].squeeze(), label='{}'.format(US_lbls[i]))
ax1.set_ylabel('Value', fontsize=label_font)
ax1.set_title('Consolidate Memory Trial', fontsize=title_font)
ax1.legend(fontsize=legend_font, bbox_to_anchor=(1, 1.05), loc='upper left')

# Plot the activities of a few MBONs
mbon_max = np.max(rt_con[:n_mbon, :])
dan_max = np.max(rt_con[n_mbon:, :])
for i, n in enumerate(sorted_mbons):
    ax2.plot(plot_time, (rt_con[(n - n_mbon), :] / dan_max) + i, '-k')
ax2.set_xlabel('Time', fontsize=label_font)
ax2.set_ylabel('Normalized DAN Activity', fontsize=label_font)
ax2.set_yticks([1, (n_mbon - 1)])
ax2.set_yticklabels(['Most Negative W_readout', 'Most Positive W_readout'],
                    rotation='vertical', verticalalignment='center')
fig.tight_layout()
# plt.close()
if save_plot:
    plot_path = dir_path + '/data_store/analysis_plots/mbon_knockout_' \
                           'consolidate_trial_{}_N{}.png'.format(rt_opt, net_str)
    fig.savefig(plot_path, bbox_inches='tight')

###############################################################################
# KC->MBON Weight Changes
wt_dt_min = -0.05
wt_dt_max = 0.05
# Imprinting Trial
# avg_kc_mbon_imp_diff_01 = np.mean(np.diff(Wt_kc_mbon_imp[:, :, :2]),
#                                   axis=1)[sorted_mbons].T
kc_mbon_imp_diff = np.diff(Wt_kc_mbon_imp[sorted_mbons, :, :2])
n_mbon, n_kc, n_imp_plots = kc_mbon_imp_diff.shape
x_labs = ['Imprint Memory']

fig, axes = plt.subplots(1, n_imp_plots, figsize=(4, 6), sharey=True)
for i in range(n_imp_plots):
    if n_imp_plots > 1:
        imp_wt_plt = axes[i].imshow(kc_mbon_imp_diff[:, :, i].T, cmap='coolwarm',
                                    origin='lower', aspect='auto',
                                    vmin=wt_dt_min, vmax=wt_dt_max)
        axes[n_imp_plots // 2].set_xlabel('MBON (Sorted)', fontsize=label_font)
        axes[0].set_ylabel('KC', fontsize=label_font)
    else:
        imp_wt_plt = axes.imshow(kc_mbon_imp_diff[:, :, i].T, cmap='coolwarm',
                                 origin='lower', aspect='auto',
                                 vmin=wt_dt_min, vmax=wt_dt_max)
axes.set_xlabel('MBON (Sorted)', fontsize=label_font)
# axes.set_xticks([])
axes.set_xticks(np.arange(n_mbon))
axes.set_xticklabels(np.arange(n_mbon) + 1)
axes.set_ylabel('KC', fontsize=label_font)
axes.set_yticks([])
# axes.set_yticks(np.arange(n_kc)[-1::25])
# axes.set_yticklabels((np.arange(n_kc) + 1)[::25])
axes.set_title(r'$\Delta W^{KC \rightarrow MBON}$'+' After\nImprinting Memory',
               fontsize=label_font)
if n_imp_plots > 1:
    fig.colorbar(imp_wt_plt, ax=axes[-1])
else:
    fig.colorbar(imp_wt_plt, ax=axes)
fig.tight_layout()
if save_plot:
    plot_path = dir_path + '/data_store/analysis_plots/mbon_knockout_imprint_' \
                           'weights_{}_N{}.png'.format(rt_opt, net_str)
    fig.savefig(plot_path, bbox_inches='tight')
plt.close()

# Consolidation Trial
# avg_kc_mbon_con_diff_01 = np.mean(np.diff(Wt_kc_mbon_con[:, :, :2]),
#                                   axis=1)[sorted_mbons].T
kc_mbon_con_diff = np.diff(Wt_kc_mbon_con[sorted_mbons, :, :(n_cons + 1)])
n_con_plots = kc_mbon_con_diff.shape[-1]

fig, axes = plt.subplots(1, n_con_plots, figsize=(3 * n_con_plots, 6), sharey=True)
for i in range(n_con_plots):
    con_wt_plt = axes[i].imshow(kc_mbon_con_diff[:, :, i].T, cmap='coolwarm',
                                origin='lower', aspect='auto',
                                vmin=wt_dt_min, vmax=wt_dt_max)
    axes[i].set_xticks(np.arange(n_mbon))
    axes[i].set_xticklabels(np.arange(n_mbon) + 1)
# axes[n_con_plots // 2].set_xlabel('MBON (Sorted)', fontsize=label_font)
fig.text(0.5, 0.04, 'MBON (Sorted)', fontsize=label_font, ha='center')
axes[0].set_yticks([])
axes[0].set_ylabel('KC', fontsize=label_font)
fig.suptitle(r'$\Delta W^{KC \rightarrow MBON}$'+' After Consolidation Intervals',
             fontsize=label_font, y=0.93)
fig.colorbar(con_wt_plt, ax=axes[-1])
# fig.tight_layout()
if save_plot:
    plot_path = dir_path + '/data_store/analysis_plots/mbon_knockout_consoli' \
                           'date_weights_{}_N{}.png'.format(rt_opt, net_str)
    fig.savefig(plot_path, bbox_inches='tight')
plt.close()

###############################################################################
# DAN and MBON Activity
# x_labels = ['None', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Imprint']
# x_labels = ['None', 'Most Negative MBON', 'Most Positive MBON']
x_labels = ['Most Aversive', 'Most Apetitive']
y_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# Imprinting Trial
rt_dans_imp = rt_imp[-n_mbon:, :]
ss_dan_imp = rt_dans_imp[sorted_mbons, (3 * tr_len - 1)::tr_len]
ss_dan_imp -= ss_dan_imp[:, 0].reshape(-1, 1)
rt_mbons_imp = rt_imp[:n_mbon, :]
ss_mbon_imp = rt_mbons_imp[sorted_mbons, (3 * tr_len - 1)::tr_len]
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.imshow(ss_dan_imp, cmap='coolwarm', origin='lower')
# ax.imshow(ss_mbon_imp, cmap='coolwarm', origin='lower')
# ax.set_xlabel('Deactivated MBON (Sorted)', fontsize=label_font)
# ax.set_xticks(np.arange(ss_dan_imp.shape[-1]))
# ax.set_xticklabels(x_labels)
# ax.set_ylabel('DAN Activity (Sorted)', fontsize=label_font)
# ax.set_yticks(np.arange(ss_dan_imp.shape[0]))
# ax.set_yticklabels(y_labels)
# plt.close()

# Consolidation Trial
rt_dans_con = rt_con[-n_mbon:, :]
ss_dan_con = rt_dans_con[sorted_mbons, ((n_cons + 1) * tr_len - 1)::tr_len]
ss_dan_con -= ss_dan_con[:, 0].reshape(-1, 1)
rt_mbons_con = rt_con[:n_mbon, :]
ss_mbon_con = rt_mbons_con[sorted_mbons, ((n_cons + 1) * tr_len - 1)::tr_len]
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.imshow(ss_dan_con, cmap='coolwarm', origin='lower')
# ax.set_xlabel('Deactivated MBON (Sorted)', fontsize=label_font)
# ax.set_xticks(np.arange(ss_dan_con.shape[-1]))
# ax.set_xticklabels(x_labels)
# ax.set_ylabel('DAN Activity (Sorted)', fontsize=label_font)
# ax.set_yticks(np.arange(ss_dan_con.shape[0]))
# ax.set_yticklabels(y_labels)
# plt.close()

# Look at changes after consolidation
ss_diff = ss_dan_imp - ss_dan_con
# ss_diff = ss_mbon_imp - ss_mbon_con
# max_val = max(np.max(ss_diff), abs(np.min(ss_diff)))
max_val = 0.15
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
dan_diff_plot = ax.imshow(ss_diff[:, 1:], cmap='coolwarm', origin='lower',
                          vmin=-max_val, vmax=max_val)
ax.set_xlabel('Deactivated MBON (Sorted)', fontsize=label_font)
# ax.set_xticks(np.arange(ss_diff.shape[-1]))
ax.set_xticks([0, ss_diff.shape[-1] - 2])
ax.set_xticklabels(x_labels)
ax.set_ylabel('DAN Activity (Sorted)', fontsize=label_font)
ax.set_yticks(np.arange(ss_diff.shape[0]))
ax.set_yticklabels(y_labels)
fig.colorbar(dan_diff_plot, ax=ax)
# plt.close()
if save_plot:
    plot_path = dir_path + '/data_store/analysis_plots/mbon_knockout_ss_dan' \
                           '_diffs_{}_N{}.png'.format(rt_opt, net_str)
    fig.savefig(plot_path, bbox_inches='tight')

fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
diff_mats = np.stack((ss_dan_imp, ss_dan_con, ss_diff))
for i in range(3):
    dan_plot = axes[i].imshow(diff_mats[i], cmap='coolwarm', origin='lower',
                              vmin=-max_val, vmax=max_val)
    axes[i].set_xlabel('Deactivated MBON (Sorted)', fontsize=label_font)
    # axes[i].set_xticks(np.arange(ss_diff.shape[-1]))
    axes[i].set_xticks([1, ss_diff.shape[-1] - 1])
    axes[i].set_xticklabels(x_labels)
fig.colorbar(dan_plot, ax=axes[-1], shrink=0.6)
axes[0].set_ylabel('DAN Activity (Sorted)', fontsize=label_font)
axes[0].set_yticks(np.arange(ss_diff.shape[0]))
axes[0].set_yticklabels(y_labels)
# fig.tight_layout()
plt.close()

plt.show()
