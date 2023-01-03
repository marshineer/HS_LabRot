# Import the required packages
import os
import torch.optim as optim
from network_classes.base_rnn import FirstOrderCondRNN
from network_classes.all_conditioning_rnn import ExtendedCondRNN
from network_classes.continual_rnn import ContinualRNN
from common.common import *

# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/network_compare/'

# Set comparison parameters
# These are the only two parameters that need to be set
# net_type determines what type of network and how it is trained
# 0-hop means there is neither feedback neurons nor MBON->DAN connections
# 1-hop means there are no feedback neurons
# 2-hop means there are feedback neurons, but they only act as a relay
# 3-hop is the standard setup from the paper (i.e. full recurrence)
# Descriptions of each net_type option
# '1' = Extended network, untrained (only initialized)
# '2' = First-order network, trained on first-order tasks
# '3' = Extended network, trained on all classical conditioning tasks
# '4' = Extended network, trained on all classical conditioning tasks (2-hop)
# '5' = Extended network, trained on all classical conditioning tasks (1-hop)
# '6' = Extended network, trained on all classical conditioning tasks (no hop)
# '7' = Extended network, trained only on 2nd-order conditioning (CS2) task
# '8' = Extended network, trained only on CS2 task (1-hop)
# '9' = Extended network, all classical conditioning tasks (no LTP)
# '10' = Extended network, all classical conditioning tasks (no LTP, 1-hop)
# '11' = Extended network (min MBONs), all classical conditioning tasks
# '12' = Extended network (min MBONs), all classical conditioning tasks (1-hop)
# '13' = Extended network (min MBONs), trained on CS2 task (1-hop)
# '14' = Continual network, trained on continual learning task (1-hop)
save_train = 'y'
net_type = '13'

# Set the network parameters
T_int = 30
T_stim = 2
lr = 0.001  # learning rate
# Set the number of networks to train
n_nets = 20

# Set the network parameters
if net_type == '1':
    # Control network
    p_ext = 0.5
    ltp = True
    n_ep = 0
    n_hop = 1
    n_mbon = 20
    net_fname = 'control_net'
    net_ftype = '2nd_order_no_train/{}_mbons/'.format(str(n_mbon).zfill(2))
elif net_type == '2':
    # Standard first-order conditioning
    p_ext = None
    ltp = True
    n_ep = 2000
    n_hop = 1
    n_mbon = 20
    net_fname = 'first_order'
    net_ftype = '1st_order_paper/'
elif net_type == '3':
    # Standard all classical conditioning
    p_ext = 0.5
    ltp = True
    n_ep = 5000
    n_hop = 3
    n_mbon = 20
    net_fname = 'second_order'
    net_ftype = '2nd_order_paper/'
elif net_type == '4':
    # All classical conditioning (no recurrence, two-hop)
    p_ext = 0.5
    ltp = True
    n_ep = 5000
    n_hop = 2
    n_mbon = 20
    net_fname = 'second_order_2hop'
    net_ftype = '2nd_order_2hop_0fbn/'
elif net_type == '5':
    # All classical conditioning (no recurrence, one-hop)
    p_ext = 0.5
    ltp = True
    n_ep = 5000
    n_hop = 1
    n_mbon = 8
    net_fname = 'second_order_1hop'
    net_ftype = '2nd_order_1hop_0fbn/'
elif net_type == '6':
    # No MBON->DAN connections at all
    p_ext = 0.5
    ltp = True
    n_ep = 5000
    n_hop = 0
    n_mbon = 20
    net_fname = '2nd_order_0hop'
    net_ftype = '2nd_order_0hop_0fbn/'
elif net_type == '7':
    # Second-order conditioning only (i.e. no extinction)
    p_ext = 0
    ltp = True
    n_ep = 5000
    n_hop = 3
    n_mbon = 20
    net_fname = 'second_order_only'
    net_ftype = '2nd_order_no_extinction/'
elif net_type == '8':
    # Second-order conditioning only (no recurrence, one-hop)
    p_ext = 0
    ltp = True
    n_ep = 5000
    n_hop = 1
    n_mbon = 20
    net_fname = 'second_order_only_1hop'
    net_ftype = '2nd_order_1hop_no_extinction/'
elif net_type == '9':
    # No LTP network
    p_ext = 0.5
    ltp = False
    n_ep = 5000
    n_hop = 3
    n_mbon = 20
    net_fname = '2nd_order_no_ltp'
    net_ftype = '2nd_order_no_ltp/'
elif net_type == '10':
    # No LTP network (no recurrence, one-hop)
    p_ext = 0.5
    ltp = False
    n_ep = 5000
    n_hop = 1
    n_mbon = 20
    net_fname = '2nd_order_no_ltp'
    net_ftype = '2nd_order_1hop_no_ltp/'
elif net_type == '11':
    # Minimal network all classical conditioning
    p_ext = 0.5
    ltp = True
    n_ep = 5000
    n_hop = 3
    n_mbon = 8
    net_fname = 'min_2nd_order'
    net_ftype = '2nd_order_min_mbon/{}_mbons/'.format(str(n_mbon).zfill(2))
elif net_type == '12':
    # Minimal network all classical conditioning (no recurrence, one-hop)
    p_ext = 0.5
    ltp = True
    n_ep = 5000
    n_hop = 1
    n_mbon = 8
    net_fname = 'min_2nd_order'
    net_ftype = '2nd_order_1hop_min_mbon/{}_mbons/'.format(str(n_mbon).zfill(2))
elif net_type == '13':
    # Minimal network CS2 trained (no recurrence, one-hop)
    p_ext = 0
    ltp = True
    n_ep = 5000
    n_hop = 1
    n_mbon = 8
    net_fname = 'min_2nd_order_only'
    net_ftype = '2nd_order_no_extinction_1hop_min_mbon/{}_mbons/'\
        .format(str(n_mbon).zfill(2))
elif net_type == '14':
    # Continual learning with (no recurrence, one-hop)
    p_ext = None
    ltp = True
    n_ep = 5000
    n_hop = 1
    n_mbon = 20
    n_stim_avg = 2
    n_stim = 4
    net_fname = 'continual_{}stim_{}avg'.format(n_stim, n_stim_avg)

for j in range(n_nets):
    # Initialize the network
    if net_type == '1':
        network = FirstOrderCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                                    n_mbon=n_mbon)
    elif net_type == '4':
        network = ContinualRNN(n_trial_avg=n_stim_avg, n_trial_odors=n_stim,
                               T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                               n_mbon=n_mbon)
    else:
        network = ExtendedCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
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
