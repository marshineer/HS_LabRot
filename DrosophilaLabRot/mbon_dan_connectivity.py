# Import the required packages
import os
from network_classes.all_conditioning_rnn import ExtendedCondRNN
from common.trial_functions import second_order_trial
from common.common import *
from common.plotting import *

# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/mbon_sensitivity/second_order_only_nets/' \
                      'trained_nets/08_mbons/'
ctrl_path = dir_path + '/data_store/network_compare/2nd_order_no_train/' \
                       '08_mbons/trained_nets/'

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# Network parameters
n_mbon = 8
n_hop = 1

# Initialize the network
network = ExtendedCondRNN(n_mbon=n_mbon, n_hop=n_hop)

# Plot parameters
n_nets = 16
sq_dim = int(np.sqrt(n_nets))
trained_wts = np.zeros((n_mbon, n_mbon, n_nets))
trained_err = np.zeros((sq_dim, sq_dim))
trained_std_wts = np.zeros((sq_dim, sq_dim))
trained_avg_wts = np.zeros((sq_dim, sq_dim))
trained_max_wts = np.zeros((sq_dim, sq_dim))
control_wts = np.zeros((n_mbon, n_mbon, n_nets))
control_err = np.zeros((sq_dim, sq_dim))
control_std_wts = np.zeros((sq_dim, sq_dim))
control_avg_wts = np.zeros((sq_dim, sq_dim))
control_max_wts = np.zeros((sq_dim, sq_dim))
min_val = 10
max_val = -10

ct = 0
for i in range(n_nets):
    # # Skip the outlier
    # if i == 4:
    #     ct += 1

    # Trained nets
    net_fname = 'second_order_no_extinct_5000ep_1hop_N{}.pt'\
        .format(str(ct + 1).zfill(2))
    fname = net_path + net_fname
    network.load_state_dict(torch.load(fname))
    # Run a first order trial (CS+)
    network.run_eval(second_order_trial, task='2nd', pos_vt=True, n_batch=20)

    # Error
    trained_err[i // sq_dim, i % sq_dim] = np.mean(network.eval_err[0])
    # Plot the KC->MBON and DAN<->MBON weight matrices
    W_mbon_to_dan = network.W_recur[:n_mbon, -n_mbon:].detach().numpy()
    # Sort by readout weight
    sorted_mbons = np.argsort(network.W_readout.detach().numpy())
    W_mbon_to_dan = W_mbon_to_dan[:, sorted_mbons]
    W_mbon_to_dan = W_mbon_to_dan[sorted_mbons, :]
    # Calculate the max and min values of the weights
    min_val = min(min_val, np.min(W_mbon_to_dan))
    max_val = max(max_val, np.max(W_mbon_to_dan))
    trained_wts[:, :, i] = W_mbon_to_dan.squeeze()
    trained_std_wts[i // sq_dim, i % sq_dim] = np.std(W_mbon_to_dan)
    # avg_wts[i // sq_dim, i % sq_dim] = np.mean(W_mbon_to_dan)
    trained_avg_wts[i // sq_dim, i % sq_dim] = np.mean(abs(W_mbon_to_dan))
    trained_max_wts[i // sq_dim, i % sq_dim] = np.max(abs(W_mbon_to_dan))

    # W_kc_to_mbon = network.eval_Wts[-1].detach().numpy()
    # fig, axes = plt.subplots(3, 1, figsize=(15, 7))
    # for j in range(3):
    #     axes[j].matshow(W_kc_to_mbon[:, :, j], cmap='coolwarm')
    # plt.show()

    # Control nets
    net_fname = 'control_net_0ep_1hop_N{}.pt'\
        .format(str(i + 1).zfill(2))
    fname = ctrl_path + net_fname
    network.load_state_dict(torch.load(fname))
    # Run a first order trial (CS+)
    network.run_eval(second_order_trial, task='2nd', pos_vt=True, n_batch=20)

    # Error
    control_err[i // sq_dim, i % sq_dim] = np.mean(network.eval_err[0])
    # Plot the KC->MBON and DAN<->MBON weight matrices
    W_mbon_to_dan = network.W_recur[:n_mbon, -n_mbon:].detach().numpy()
    min_val = min(min_val, np.min(W_mbon_to_dan))
    max_val = max(max_val, np.max(W_mbon_to_dan))
    control_wts[:, :, i] = W_mbon_to_dan
    control_std_wts[i // sq_dim, i % sq_dim] = np.std(W_mbon_to_dan)
    # avg_wts[i // sq_dim, i % sq_dim] = np.mean(W_mbon_to_dan)
    control_avg_wts[i // sq_dim, i % sq_dim] = np.mean(abs(W_mbon_to_dan))
    control_max_wts[i // sq_dim, i % sq_dim] = np.max(abs(W_mbon_to_dan))

    # ct += 1

################## PLOT THE MBON->DAN WEIGHTS AS MATRICES ##################
ct = 0
fig, axes = plt.subplots(sq_dim, sq_dim, figsize=(12, 12))
curr_plot = None
for i in range(n_nets):
    curr_plot = axes[i // sq_dim, i % sq_dim].matshow(trained_wts[:, :, i],
                                                      cmap='coolwarm',
                                                      vmin=min_val, vmax=max_val)
    axes[i // sq_dim, i % sq_dim].set_xlabel('MBONs')
    axes[i // sq_dim, i % sq_dim].set_ylabel('DANs')
    axes[i // sq_dim, i % sq_dim].set_xticks([])
    axes[i // sq_dim, i % sq_dim].set_yticks([])
    # fig.colorbar(curr_plot, ax=axes[i // sq_dim, i % sq_dim])
fig.colorbar(curr_plot, ax=axes)
fig.suptitle('MBON -> DAN Weights', y=0.90)
plt.close()

###### PLOT THE MBON->DAN WEIGHT AVERAGE AND STD DEV ACROSS ALL NETWORKS ######
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
avg_plot = ax1.matshow(np.mean(trained_wts, axis=-1), cmap='coolwarm')
std_plot = ax2.matshow(np.std(trained_wts, axis=-1), cmap='Reds')
fig.colorbar(avg_plot, ax=ax1)
fig.colorbar(std_plot, ax=ax2)
# plt.close()

###### PLOT THE AVG ERROR FOR EACH NETWORK, AND COMPARE THAT TO THE
###### MBON->DAN WEIGHT VAR, AVG, ABS AVG AND MAX MAGNITUDE
fig, axes = plt.subplots(2, 4, figsize=(18, 6))
# Trained plots
err_plot = axes[0, 0].matshow(trained_err, cmap='Reds', vmin=0)
axes[0, 0].set_title('Network Error')
fig.colorbar(err_plot, ax=axes[0, 0])
std_wts_plot = axes[0, 1].matshow(trained_std_wts, cmap='Reds', vmin=0)
axes[0, 1].set_title('Weight Var')
fig.colorbar(std_wts_plot, ax=axes[0, 1])
# avg_wts_plot = axes[2].matshow(avg_wts, cmap='coolwarm',
#                                vmin=np.min(avg_wts), vmax=np.max(avg_wts))
avg_wts_plot = axes[0, 2].matshow(trained_avg_wts, cmap='Reds', vmin=0)
axes[0, 2].set_title('Avg Weight Magnitude')
fig.colorbar(avg_wts_plot, ax=axes[0, 2])
max_wts_plot = axes[0, 3].matshow(trained_max_wts, cmap='Reds', vmin=0)
axes[0, 3].set_title('Max Weight Magnitude')
fig.colorbar(max_wts_plot, ax=axes[0, 3])

# Control plots
err_plot = axes[1, 0].matshow(control_err, cmap='Reds', vmin=0, vmax=0.01)
axes[1, 0].set_title('Network Error')
fig.colorbar(err_plot, ax=axes[1, 0])
std_wts_plot = axes[1, 1].matshow(control_std_wts, cmap='Reds', vmin=0)
axes[1, 1].set_title('Weight Var')
fig.colorbar(std_wts_plot, ax=axes[1, 1])
# avg_wts_plot = axes[2].matshow(avg_wts, cmap='coolwarm',
#                                vmin=np.min(avg_wts), vmax=np.max(avg_wts))
avg_wts_plot = axes[1, 2].matshow(control_avg_wts, cmap='Reds', vmin=0)
axes[1, 2].set_title('Avg Weight Magnitude')
fig.colorbar(avg_wts_plot, ax=axes[1, 2])
max_wts_plot = axes[1, 3].matshow(control_max_wts, cmap='Reds', vmin=0)
axes[1, 3].set_title('Max Weight Magnitude')
fig.colorbar(max_wts_plot, ax=axes[1, 3])

plt.close()



# Calculate a correlation matrix across the network's average error, weight
# variance, average weight magnitude, and maximum weight magnitude
# Include untrained networks as a control
n_all_nets = 20
trained_stats = np.zeros((4, n_all_nets))
control_stats = np.zeros((4, n_all_nets))
stat_labels = ['Avg Err', 'Wt Var', 'Abs Wt Avg', 'Wt Max']

for i in range(n_all_nets):
    # Trained nets
    net_fname = 'second_order_no_extinct_5000ep_1hop_N{}.pt'\
        .format(str(i + 1).zfill(2))
    fname = net_path + net_fname
    network.load_state_dict(torch.load(fname))

    # Run a first order trial (CS+)
    network.run_eval(second_order_trial, task='2nd', pos_vt=True, n_batch=20)

    # Assign stats
    trained_stats[0, i] = np.mean(network.eval_err[0])
    W_kc_to_mbon = network.eval_Wts[-1].detach().numpy()
    W_mbon_to_dan = network.W_recur[:n_mbon, -n_mbon:].detach().numpy()
    trained_stats[1, i] = np.var(W_mbon_to_dan)
    trained_stats[2, i] = np.mean(abs(W_mbon_to_dan))
    trained_stats[3, i] = np.max(abs(W_mbon_to_dan))

    # Control nets
    net_fname = 'control_net_0ep_1hop_N{}.pt'\
        .format(str(i + 1).zfill(2))
    fname = ctrl_path + net_fname
    network.load_state_dict(torch.load(fname))

    # Run a first order trial (CS+)
    network.run_eval(second_order_trial, task='2nd', pos_vt=True, n_batch=20)

    # Assign stats
    control_stats[0, i] = np.mean(network.eval_err[0])
    W_kc_to_mbon = network.eval_Wts[-1].detach().numpy()
    W_mbon_to_dan = network.W_recur[:n_mbon, -n_mbon:].detach().numpy()
    control_stats[1, i] = np.var(W_mbon_to_dan)
    control_stats[2, i] = np.mean(abs(W_mbon_to_dan))
    control_stats[3, i] = np.max(abs(W_mbon_to_dan))

# Trained nets
trained_mat = np.corrcoef(trained_stats)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
trained_plot = ax.matshow(trained_mat, cmap='Reds')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(stat_labels)
ax.set_yticks(np.arange(4))
ax.set_yticklabels(stat_labels)
ax.set_title('Relationship Between Network Error and\n'
             'MBON->DAN Weight Statistics', fontsize=label_font, y=1.1)
cbar = fig.colorbar(trained_plot, ax=ax)
cbar.set_label('Correlation', fontsize=label_font)
fig.tight_layout()
fig.suptitle('Trained Network', fontsize=title_font, y=1.05)
plt.close()

# Control nets
control_mat = np.corrcoef(control_stats)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
control_plot = ax.matshow(control_mat, cmap='Reds')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(stat_labels)
ax.set_yticks(np.arange(4))
ax.set_yticklabels(stat_labels)
ax.set_title('Relationship Between Network Error and\n'
             'MBON->DAN Weight Statistics', fontsize=label_font, y=1.1)
cbar = fig.colorbar(control_plot, ax=ax)
cbar.set_label('Correlation', fontsize=label_font)
fig.tight_layout()
fig.suptitle('Control Network', fontsize=title_font, y=1.05)
plt.close()

# # All nets
# all_mat = np.corrcoef(np.concatenate((trained_stats, control_stats), axis=0))
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# all_plot = ax.matshow(all_mat, cmap='Reds')
# ax.set_xticks(np.arange(8))
# ax.set_xticklabels(stat_labels + stat_labels)
# ax.set_yticks(np.arange(8))
# ax.set_yticklabels(stat_labels + stat_labels)
# ax.set_title('Relationship Between Network Error and\n'
#              'MBON->DAN Weight Statistics', fontsize=label_font, y=1.1)
# cbar = fig.colorbar(all_plot, ax=ax)
# cbar.set_label('Correlation', fontsize=label_font)
# fig.tight_layout()
# fig.suptitle('Control Network', fontsize=title_font, y=1.05)
# plt.close()

plt.show()
