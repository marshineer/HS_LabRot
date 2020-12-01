# Import the required packages
import torch.optim as optim
import matplotlib.pyplot as plt
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from network_classes.paper_tasks.no_plasticity_rnn import NoPlasticityRNN
from network_classes.paper_tasks.continual_rnn import ContinualRNN
from common.common import *

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# Set the training and plotting parameters
net_type = input('Input network type (first, all_classic, no_plast or '
                 'continual: ')
T_int = int(input('Input length of training interval: '))
n_ep = int(input('Input number of epochs to train for: '))

# Initialize the network
if net_type == 'first':
    network = FirstOrderCondRNN(T_int=T_int)
elif net_type == 'all_classic':
    network = ExtendedCondRNN()
elif net_type == 'no_plast':
    n_odors = int(input('Input number of odors for network: '))
    network = NoPlasticityRNN(n_odors=n_odors, T_int=T_int)
elif net_type == 'continual':
    n_stim_avg = int(input('Input average number of stimulus presentations: '))
    n_stim = int(input('Input number of odors per trial: '))
    network = ContinualRNN(n_trial_avg=n_stim_avg, n_trial_odors=n_stim,
                           T_int=T_int)

# Print the parameter shapes to check
for param in network.parameters():
    print(param.shape)

# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(network.parameters(), lr=lr)

# Train the network
net_path = '/home/marshineer/Dropbox/Ubuntu/lab_rotations/sprekeler/' \
           'DrosophilaLabRot/data_store/paper_nets/'
if net_type == 'first':
    # Classical training
    plt_title = 'First-order Conditioning Training Losses'
    net_fname = 'trained_first_order_{}ep'.format(n_ep)
if net_type == 'all_classic':
    # Extinction and second-order training
    plt_title = 'Classical Conditioning Training Losses'
    net_fname = 'trained_all_classic_{}ep'.format(n_ep)
elif net_type == 'no_plast':
    # No plasticity training
    plt_title = 'No Plasticity Conditioning Training Losses'
    net_fname = 'trained_no_plasticity_{}odor_{}ep'.format(n_odors, n_ep)
elif net_type == 'continual':
    # Continual learning training
    plt_title = 'Continual Learning Training Losses'
    net_fname = 'trained_continual_{}stim_{}avg_{}ep'.format(n_stim, n_stim_avg, n_ep)

# Set network parameters and train
loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)
fname = net_path + 'trained_nets/' + net_fname + '.pt'
torch.save(network.state_dict(), fname)

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
plt.savefig(plot_path, bbox_inches='tight')

# Show the plot
plt.show()
