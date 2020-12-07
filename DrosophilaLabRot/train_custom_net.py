# Import the required packages
import torch.optim as optim
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from network_classes.paper_tasks.continual_rnn import ContinualRNN
# from network_classes.paper_extensions.knockout_recur import NoRecurFirstO,\
#     NoRecurContinual, NoRecurExtended
from common.common import *
from common.plotting import *

# Define the path for saving the trained networks and loss plots
net_path = '/home/marshineer/Dropbox/Ubuntu/lab_rotations/sprekeler/' \
           'DrosophilaLabRot/data_store/extension_nets/'

# Set the training and plotting parameters
net_type = input('Input network type\n'
                 'First-order without recurrence = 1\n'
                 'All classical conditioning without recurrence = 2\n'
                 'Only 2nd-order training without recurrence = 3\n'
                 'Continual learning without recurrence = 4: ')
save_net = input('Do you wish to save this network? y/n ')
T_int = int(input('Input length of training interval: '))
T_stim = int(input('Input length of stimulus presentation: '))
n_ep = int(input('Input number of epochs to train for: '))
n_hop = int(input('Input number of hops in network: '))

# Initialize the network
if net_type == '1':
    network = FirstOrderCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop)
    # network = NoRecurFirstO(T_int=T_int, T_stim=T_stim)
elif net_type == '2':
    network = ExtendedCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop)
    # network = NoRecurExtended(T_int=T_int, T_stim=T_stim)
elif net_type == '3':
    network = ExtendedCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop)
    # network = NoRecurExtended(T_int=T_int, T_stim=T_stim)
elif net_type == '4':
    n_stim_avg = int(input('Input average number of stimulus presentations: '))
    n_stim = int(input('Input number of odors per trial: '))
    network = ContinualRNN(n_trial_avg=n_stim_avg, n_trial_odors=n_stim,
                           T_int=T_int, T_stim=T_stim, n_hop=n_hop)
    # network = NoRecurContinual(n_trial_avg=n_stim_avg, n_trial_odors=n_stim,
    #                            T_int=T_int, T_stim=T_stim)

# Print the parameter shapes to check
for param in network.parameters():
    print(param.shape)

# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(network.parameters(), lr=lr)

# Train the network
# Set network parameters and train
if net_type == '1':
    plt_title = 'First-order (No Recurrence) Training Losses'
    net_fname = 'trained_knockout_fo'
    loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)
elif net_type == '2':
    plt_title = 'All Classical (No Recurrence) Training Losses'
    net_fname = 'trained_knockout_ac'
    loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)
elif net_type == '3':
    plt_title = 'Second-order (No Recurrence) Training Losses'
    net_fname = 'trained_knockout_so'
    loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep, p_extinct=0)
elif net_type == '4':
    plt_title = 'Continual (No Recurrence) Training Losses'
    net_fname = 'trained_knockout_cl_{}stim_{}avg'.format(n_stim, n_stim_avg)
    loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)
# Save the network
fsuff = '_{}ep_{}hop'.format(n_ep, n_hop)
if save_net == 'y':
    fname = net_path + 'trained_nets/' + net_fname + fsuff + '.pt'
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
axes[1].plot(loss_hist[5:])
axes[1].set_xlabel('Epoch', fontsize=label_font)
axes[1].set_ylabel('Loss', fontsize=label_font)
fig.suptitle(plt_title, fontsize=title_font, y=1.05)
fig.tight_layout()

# Save the losses plot
if save_net == 'y':
    plot_path = net_path + 'loss_plots/' + net_fname + '_losses' + fsuff + '.png'
    plt.savefig(plot_path, bbox_inches='tight')

# Show the plot
plt.show()
