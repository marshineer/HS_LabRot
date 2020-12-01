# Import the required packages
import torch.optim as optim
from network_classes.paper_extensions.knockout_recur import NoRecurFirstO,\
    NoRecurContinual
from common.common import *
from common.plotting import *

# Set the training and plotting parameters
net_type = input('Input network type (knockout_r_fo, knockout_r_cl): ')
save_net = input('Do you wish to save this network? y/n ')
T_int = int(input('Input length of training interval: '))
T_stim = int(input('Input length of stimulus presentation: '))
n_ep = int(input('Input number of epochs to train for: '))

# Initialize the network
if net_type == 'knockout_r_fo':
    network = NoRecurFirstO(T_int=T_int, T_stim=T_stim)
elif net_type == 'knockout_r_cl':
    n_stim_avg = int(input('Input average number of stimulus presentations: '))
    n_stim = int(input('Input number of odors per trial: '))
    network = NoRecurContinual(n_trial_avg=n_stim_avg, n_trial_odors=n_stim,
                           T_int=T_int, T_stim=T_stim)

# Print the parameter shapes to check
for param in network.parameters():
    print(param.shape)

# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(network.parameters(), lr=lr)

# Train the network
net_path = '/home/marshineer/Dropbox/Ubuntu/lab_rotations/sprekeler/' \
           'DrosophilaLabRot/data_store/extension_nets/'
if net_type == 'knockout_r_fo':
    plt_title = 'First-order (No Recurrence) Training Losses'
    net_fname = 'trained_knockout_fo_{}ep'.format(n_ep)
elif net_type == 'knockout_r_cl':
    plt_title = 'Continual (No Recurrence) Training Losses'
    net_fname = 'trained_knockout_cl_{}stim_{}avg_{}ep'.format(n_stim,
                                                               n_stim_avg,
                                                               n_ep)

# Set network parameters and train
loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)
fname = net_path + 'trained_nets/' + net_fname + '.pt'
if save_net == 'y':
    torch.save(network.state_dict(), fname)

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# Plot the loss function
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].plot(loss_hist)
axes[0].set_xlabel('Epoch', fontsize=label_font)
axes[0].set_ylabel('Loss', fontsize=label_font)
axes[1].plot(loss_hist[10:])
axes[1].set_xlabel('Epoch', fontsize=label_font)
axes[1].set_ylabel('Loss', fontsize=label_font)
fig.suptitle(plt_title, fontsize=title_font, y=1.05)
fig.tight_layout()

# Save the losses plot
plot_path = net_path + 'loss_plots/' + net_fname + '_losses.png'
if save_net == 'y':
    plt.savefig(plot_path, bbox_inches='tight')

# Show the plot
plt.show()

# # Plot a single trial
# save_plot = input('Do you wish to save the plot? y/n ')
# if net_type == 'knockout_r_fo':
#     plot_type = input('Input type of trial to run (CS+ or CS-): ')
#
# p_ctrl = 0
# if net_type == 'knockout_r_fo':
#     # First-order conditioning with no recurrence
#     if plot_type == 'CS+':
#         p_ctrl = 0
#         trial_ls = [first_order_cond_csp, first_order_test]
#         plt_fname = 'knockout_fo_csp_{}ep'.format(n_ep)
#         plt_ttl = 'First-order (No Recurrence) Conditioning (CS+)'
#         plt_lbl = (['CS+'], ['US'])
#     elif plot_type == 'CS-':
#         p_ctrl = 1
#         trial_ls = [first_order_csm, first_order_test]
#         plt_fname = 'knockout_fo_csm_{}ep'.format(n_ep)
#         plt_ttl = 'First-order (No Recurrence) Conditioning (CS-)'
#         plt_lbl = (['CS-'], ['US'])
# elif net_type == 'knockout_r_cl':
#     # Continual learning with no recurrence
#     trial_ls = [continual_trial]
#     plt_fname = 'knockout_recur_cl_{}stim_{}avg_{}ep'.format(n_stim, n_stim_avg,
#                                                              n_ep)
#     plt_ttl = 'Continual Learning (No Recurrence)'
#     cs_lbl = []
#     for j in range(n_stim):
#         cs_lbl.append('CS{}'.format(j + 1))
#     plt_lbl = (cs_lbl, ['US1', 'US2'])
#
# # Plot the trial
# T_vars = (network.T_int, network.T_stim, network.dt)
# fig = plot_trial(network, trial_ls, plt_ttl, plt_lbl, T_vars, p_ctrl=p_ctrl,
#                  pos_vt=True)
#
# # Save the losses plot
# plot_path = net_path + 'trial_plots/' + plt_fname + '_trial.png'
# if save_plot == 'y':
#     fig.savefig(plot_path, bbox_inches='tight')
