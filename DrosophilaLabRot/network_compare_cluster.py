# Import the required packages
import os
import torch.optim as optim
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from network_classes.paper_tasks.continual_rnn import ContinualRNN
from common.common import *

# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/network_compare/'

load_net = 'n'
save_train = 'y'
# Set comparison parameters
net_type = '6'
n_hop = 1
n_mbon = 20
# Set the network parameters
T_int = 30
T_stim = 2
if net_type == '4':
    n_stim_avg = 2
    n_stim = 4
lr = 0.001  # learning rate
# Set the number of networks to train
n_nets = 20
print('')

# Set the network parameters
if net_type == '1':
    # Standard first-order conditioning
    net_fname = 'first_order'
    net_ftype = '1st_order_paper/'
    p_ext = None
    ltp = True
    n_ep = 2000
elif net_type == '2':
    if n_hop == 0:
        # Standard all classical conditioning
        net_fname = 'second_order_only'
        net_ftype = '2nd_order_no_extinction/'
    elif n_hop == 1:
        # All classical conditioning with no recurrence (one-hop)
        net_fname = 'second_order_only_1hop'
        net_ftype = '2nd_order_1hop_no_extinction/'
    p_ext = 0
    ltp = True
    n_ep = 5000
elif net_type == '3':
    if n_hop == 0:
        # Standard second-order conditioning
        net_fname = 'second_order'
        net_ftype = '2nd_order_paper/'
    elif n_hop == 1:
        # Second-order conditioning with no recurrence (one-hop)
        net_fname = 'second_order_1hop'
        net_ftype = '2nd_order_1hop_0fbn/'
    elif n_hop == 2:
        # Second-order conditioning with no recurrence (two_hop)
        net_fname = 'second_order_2hop'
        net_ftype = '2nd_order_2hop_0fbn/'
    p_ext = 0.5
    ltp = True
    n_ep = 5000
elif net_type == '4':
    # Continual learning with no recurrence
    net_fname = 'continual_{}stim_{}avg'.format(n_stim, n_stim_avg)
    p_ext = None
    ltp = True
    n_ep = 5000
elif net_type == '5':
    # Minimal network
    net_fname = 'min_2nd_order'
    net_ftype = '2nd_order_1hop_min_mbon/'
    p_ext = 0
    ltp = True
    n_ep = 5000
elif net_type == '6':
    # No LTP network
    net_fname = '2nd_order_no_ltp'
    net_ftype = '2nd_order_1hop_no_ltp/'
    p_ext = 0.5
    ltp = False
    n_ep = 5000

for j in range(n_nets):
    # Initialize the network
    if net_type == '1':
        network = FirstOrderCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                                    n_mbon=n_mbon)
    elif net_type in ['2', '3', '5', '6']:
        network = ExtendedCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                                  n_mbon=n_mbon)
    elif net_type == '4':
        network = ContinualRNN(n_trial_avg=n_stim_avg, n_trial_odors=n_stim,
                               T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                               n_mbon=n_mbon)
    # Define the model's optimizer
    optimizer = optim.RMSprop(network.parameters(), lr=lr)

    # Load or train the network
    fsuff = '_{}ep_{}hop_N{}'.format(n_ep, n_hop, str(j + 1).zfill(2))
    full_path = net_path + net_ftype + 'trained_nets/'
    fname = full_path + net_fname + fsuff + '.pt'
    loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep, ltp=ltp,
                                  p_extinct=p_ext)
    if save_train == 'y':
        torch.save(network.state_dict(), fname)
