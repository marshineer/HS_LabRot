# Import the required packages
import os
import torch.optim as optim
from network_classes.no_plasticity_rnn import NoPlasticityRNN
from common.trial_functions import no_plasticity_trial
from common.common import *
from common.plotting import *


# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/paper_nets/'

# Set the network parameters
n_ep = 5000
T_int = 40
n_odor = 10

# Initialize the network
network = NoPlasticityRNN(T_int=T_int, n_odors=n_odor)
# Print the parameter shapes to check
for param in network.parameters():
    print(param.shape)

# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(network.parameters(), lr=lr)

# Determine user inputs (load network and save plots)
load_net = input('Do you wish to load an existing network? y/n ')
save_plot = input('Do you wish to save the trial plot? y/n ')

# Set the network filepath
o_str = str(n_odor).zfill(2)
net_fname = 'trained_no_plasticity_{}odor_{}ep'.format(o_str, n_ep)
fname = net_path + 'trained_nets/' + net_fname + '.pt'
if load_net == 'y':
    # Load the existing network
    network.load_state_dict(torch.load(fname))
else:
    # Train and save the network
    loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)
    torch.save(network.state_dict(), fname)

    # Plot the loss function
    label_font = 18
    title_font = 24
    legend_font = 12
    plt_title = 'No Plasticity Conditioning Training Losses'
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

# Plot and save trials for the network
plot_list = ['CS+', 'CS-']
plot_lbls = [['CS+'], ['CS+', 'CS-']]
plot_fsuff = ['csp', 'csm']
for i in range(len(plot_list)):
    network.run_eval(no_plasticity_trial, task=plot_list[i], pos_vt=True)
    plt_ttl = 'No Plasticity ({} Trial)'.format(plot_list[i])
    plt_lbl = (plot_lbls[i], ['US'])

    # Plot the trial
    fig, _ = plot_trial(network, plt_ttl, plt_lbl)

    # Save the losses plot
    if save_plot == 'y':
        plt_fname = 'no_plasticity_{}odor_{}ep_{}_trial.png'\
            .format(o_str, n_ep, plot_fsuff[i])
        plot_path = net_path + 'trial_plots/' + plt_fname
        fig.savefig(plot_path, bbox_inches='tight')
    else:
        plt.show()
