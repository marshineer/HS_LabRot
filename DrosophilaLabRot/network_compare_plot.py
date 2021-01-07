# Import the required packages
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from network_classes.base_rnn import FirstOrderCondRNN
from network_classes.all_conditioning_rnn import ExtendedCondRNN
from common.trial_functions import *

# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/network_compare/'

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# Create a list of paths for each network type
#  (path, prefix, n_epoch, n_hop)
p_list = [('2nd_order_no_train/20_mbons/', 'control_net', 0, 1),
          ('1st_order_paper/', 'first_order', 2000, 0),
          ('2nd_order_paper/', 'second_order', 5000, 0),
          ('2nd_order_no_extinction/', 'second_order_only', 5000, 0),
          ('2nd_order_2hop_0fbn/', 'second_order_2hop', 5000, 2),
          ('2nd_order_1hop_0fbn/', 'second_order_1hop', 5000, 1),
          ('2nd_order_1hop_no_extinction/', 'second_order_only_1hop', 5000, 1),
          ('2nd_order_1hop_no_ltp/', '2nd_order_no_ltp', 5000, 1),
          ('2nd_order_1hop_min_mbon/', 'min_2nd_order', 5000, 1)]
p_names = ['Untrained Net',
           '1st-order Conditioning Net',
           'All Classical Conditioning Net\n(Control)',
           '2nd-order Conditioning Net\n(Only CS2 Training)',
           'All Classical Conditioning Net\n(No Feedback, 2 hop)',
           'All Classical Conditioning Net\n(No Feedback, 1 hop)',
           '2nd-order Conditioning Net\n(Only CS2, 1hop)',
           'All Classical Conditioning Net\n(No LTP, 1 hop)',
           'All Classical Conditioning Net\n(8 MBONs)']

# Set which task to plot
#  '1' = first-order conditioning
#  '2' = extinction
#  '3' = second-order conditioning
task_type = '1'
if task_type == '1':
    task = 'CS+'
    plt_ttl = 'First-order Conditioning Task'
    plt_name = 'plots/first_order_compare.png'
elif task_type == '2':
    task = 'extinct'
    plt_ttl = 'Extinction Task'
    plt_name = 'plots/extinction_compare.png'
elif task_type == '3':
    task = '2nd'
    plt_ttl = 'Second-order Conditioning Task'
    plt_name = 'plots/second_order_compare.png'

# Set network parameters
T_int = 30
T_stim = 2
dt = 0.5
n_mbon = 20
# Set evaluation parameters
n_trials = 50
n_nets = 20
err_list = []
log_err_list = []
log_min = 100
log_max = -100

# For each type of network
for i, path in enumerate(p_list):
    # Define a matrix to store the data
    task_err = np.zeros((n_trials, n_nets))
    net_err = np.zeros(n_nets)
    net_ftype, net_fname, n_ep, n_hop = p_list[i]
    # Initialize the network
    n_hopp = n_hop
    if n_hop == 0:
        n_hopp = 3
    if i == (len(p_list) - 1):
        n_mbon = 6
    if i == 0:
        network = FirstOrderCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hopp,
                                    n_mbon=n_mbon)
    else:
        network = ExtendedCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hopp,
                                  n_mbon=n_mbon)

    # For each instance of the network
    print(i)
    for j in range(n_nets):
        # Load the network
        fsuff = '_{}ep_{}hop_N{}'.format(n_ep, n_hop, str(j + 1).zfill(2))
        full_path = net_path + net_ftype + 'trained_nets/'
        fname = full_path + net_fname + fsuff + '.pt'
        network.load_state_dict(torch.load(fname))

        # Evaluate the network
        network.run_eval(second_order_trial, n_batch=n_trials, task=task)
        vts = network.eval_vts[0]
        vt_opts = network.eval_vt_opts[0]
        net_err[j] = network.eval_err[0]
        for k in range(n_trials):
            task_err[k, j] = cond_err(vts[k, :], vt_opts[k, :])

    # err_list.append(task_err.flatten())
    # log_err_list.append(np.log10(task_err.flatten()))
    # log_min = min(np.min(np.log10(task_err)), log_min)
    # log_max = max(np.max(np.log10(task_err)), log_max)
    err_list.append(net_err.flatten())
    log_err_list.append(np.log10(net_err.flatten()))
    log_min = min(np.min(np.log10(net_err)), log_min)
    log_max = max(np.max(np.log10(net_err)), log_max)

# print(log_min)
# print(log_max)
# print(np.arange(np.floor(log_min), np.ceil(log_max) + 1))

# Plot the box plot comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# ax.boxplot(err_list)
ax.boxplot(log_err_list)
ax.set_xticks(np.arange(len(p_list)) + 1)
ax.set_xticklabels(p_names, rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel('Readout Error (log 10)', fontsize=label_font)
ax.set_yticks(np.arange(np.floor(log_min), np.ceil(log_max) + 1))
ax.set_title('Network Performance Comparison of\n'+plt_ttl, fontsize=title_font)
fig.tight_layout()
plt.show()

# Save the losses plot
plot_path = net_path + plt_name
fig.savefig(plot_path, bbox_inches='tight')
