# Import the required packages
import os
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from network_classes.paper_tasks.trial_functions import second_order_trial
from common.common import *
from common.plotting import *

# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/mbon_sensitivity/second_order_only_nets/' \
                      'trained_nets/08_mbons/'

# Network parameters
n_mbon = 8
n_fbn = 60
n_hop = 1

# Plot parameters
n_nets = 16
net_wts = np.zeros((n_mbon, n_mbon, n_nets))
sq_dim = int(np.sqrt(n_nets))
net_err = np.zeros((sq_dim, sq_dim))
std_wts = np.zeros((sq_dim, sq_dim))
avg_wts = np.zeros((sq_dim, sq_dim))
max_wts = np.zeros((sq_dim, sq_dim))
min_val = 10
max_val = -10

# Load the minimum network
network = ExtendedCondRNN(n_mbon=n_mbon, n_fbn=n_fbn, n_hop=n_hop)

fig, axes = plt.subplots(sq_dim, sq_dim, figsize=(12, 12))
ct = 0
for i in range(n_nets):
    # print(i // 3)
    # print(i % 3)
    if i == 4:
        ct += 1

    net_fname = 'second_order_no_extinct_5000ep_1hop_N{}.pt'\
        .format(str(ct + 1).zfill(2))
    fname = net_path + net_fname
    network.load_state_dict(torch.load(fname))

    # Run a first order trial (CS+)
    network.run_eval(second_order_trial, task='2nd', pos_vt=True, n_batch=20)

    # Error
    # print(type(network.eval_err))
    # print(len(network.eval_err))
    net_err[i // sq_dim, i % sq_dim] = np.mean(network.eval_err[0])

    # Plot the KC->MBON and DAN<->MBON weight matrices
    W_kc_to_mbon = network.eval_Wts[-1].detach().numpy()
    # print(W_kc_to_mbon.shape)
    W_mbon_to_dan = network.W_recur[:n_mbon, -n_mbon:].detach().numpy()
    net_wts[:, :, i] = W_mbon_to_dan
    std_wts[i // sq_dim, i % sq_dim] = np.std(W_mbon_to_dan)
    # avg_wts[i // sq_dim, i % sq_dim] = np.mean(W_mbon_to_dan)
    avg_wts[i // sq_dim, i % sq_dim] = np.mean(abs(W_mbon_to_dan))
    max_wts[i // sq_dim, i % sq_dim] = np.max(abs(W_mbon_to_dan))
    min_val = min(min_val, np.min(W_mbon_to_dan))
    max_val = max(max_val, np.max(W_mbon_to_dan))
    # print(W_mbon_to_dan.shape)

    # fig, axes = plt.subplots(3, 1, figsize=(15, 7))
    # for j in range(W_kc_to_mbon.shape[-1]):
    #     axes[j].imshow(W_kc_to_mbon[:, :, j], cmap='coolwarm')
    # plt.show()

    ct += 1

ct = 0
for i in range(n_nets):
    curr_plot = axes[i // sq_dim, i % sq_dim].matshow(net_wts[:, :, i],
                                                      cmap='coolwarm',
                                                      vmin=min_val, vmax=max_val)
    axes[i // sq_dim, i % sq_dim].set_xlabel('MBONs')
    axes[i // sq_dim, i % sq_dim].set_ylabel('DANs')
    axes[i // sq_dim, i % sq_dim].set_xticks([])
    axes[i // sq_dim, i % sq_dim].set_yticks([])
    # fig.colorbar(curr_plot, ax=axes[i // sq_dim, i % sq_dim])
fig.colorbar(curr_plot, ax=axes)
fig.suptitle('MBON -> DAN Weights', y=0.90)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# avg_plot = ax1.matshow(np.mean(net_wts, axis=-1), cmap='coolwarm')
# std_plot = ax2.matshow(np.std(net_wts, axis=-1), cmap='Reds')
# fig.colorbar(avg_plot, ax=ax1)
# fig.colorbar(std_plot, ax=ax2)

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
err_plot = axes[0].matshow(net_err, cmap='Reds', vmin=0)
axes[0].set_title('Network Error')
fig.colorbar(err_plot, ax=axes[0])
std_wts_plot = axes[1].matshow(std_wts, cmap='Reds', vmin=0)
axes[1].set_title('Weight Var')
fig.colorbar(std_wts_plot, ax=axes[1])
# avg_wts_plot = axes[2].matshow(avg_wts, cmap='coolwarm',
#                                vmin=np.min(avg_wts), vmax=np.max(avg_wts))
avg_wts_plot = axes[2].matshow(avg_wts, cmap='Reds', vmin=0)
axes[2].set_title('Avg Weight Magnitude')
fig.colorbar(avg_wts_plot, ax=axes[2])
max_wts_plot = axes[3].matshow(max_wts, cmap='Reds', vmin=0)
axes[3].set_title('Max Weight Magnitude')
fig.colorbar(max_wts_plot, ax=axes[3])

plt.show()
