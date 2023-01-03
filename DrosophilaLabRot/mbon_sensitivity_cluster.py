# Import the required packages
import numpy as np
import os
import torch.optim as optim
from network_classes.base_rnn import FirstOrderCondRNN
from network_classes.all_conditioning_rnn import ExtendedCondRNN
from network_classes.continual_rnn import ContinualRNN
from common.trial_functions import first_order_trial
from common.trial_functions import second_order_trial
from common.trial_functions import continual_trial
from common.common import *

# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/mbon_sensitivity/'

# The only user inputs required are n_mbon and net_type
# Set the number of MBONs to test
n_mbon_0 = 6
n_vals = 5
inc_type = 'lin'
# n_mbon = 7

# Set the type of network to train
# All networks use a 1-hop MBON->DAN connection (no feedback neurons)
# '1' = First-order conditioning with no recurrence
# '2' = Second-order conditioning with no recurrence
# '3' = All classical conditioning with no recurrence
# '4' = Continual learning with no recurrence
net_type = '3'
# Set the number of networks to train
n_nets = 20

# Set the parameters for saving/loading data
load_net = 'y'
save_train = 'y'
save_test = 'y'
# Set the network parameters
n_ep = 5000
n_trial = 200
T_int = 30
T_stim = 2
n_hop = 1
n_stim_avg = 2
n_stim = 4
lr = 0.001  # learning rate
print('')

# Set the network parameters
if net_type == '1':
    # First-order conditioning with no recurrence
    net_fname = 'first_order'
    net_ftype = 'first_order_nets/'
    f_trial = second_order_trial
    trial_task = ['CS+']
    p_ext = None
elif net_type == '2':
    # Second-order conditioning with no recurrence
    net_fname = 'second_order_no_extinct'
    net_ftype = 'second_order_only_nets/'
    f_trial = second_order_trial
    trial_task = ['extinct', '2nd']
    p_ext = 0.0
elif net_type == '3':
    # All classical conditioning with no recurrence
    net_fname = 'second_order'
    net_ftype = 'second_order_nets/'
    f_trial = second_order_trial
    trial_task = ['extinct', '2nd']
    p_ext = 0.5
elif net_type == '4':
    # Continual learning with no recurrence
    net_fname = 'continual_{}stim_{}avg'.format(n_stim, n_stim_avg)
    trial_ls = [continual_trial]
    p_ext = None

for i in range(n_vals):
    # Set the number of MBONs
    if inc_type == 'exp':
        n_mbon = n_mbon_0 ** (i + 1)
    elif inc_type == 'lin':
        n_mbon = n_mbon_0 + i
    print('Current MBONs:', n_mbon)

    # Initialize variables to store training data
    header = ''
    train_loss = torch.zeros(n_ep, n_nets)
    test_loss = np.zeros((n_trial * len(trial_task), n_nets))

    for j in range(n_nets):
        # Update the header for this network's loss data
        header += 'Net {}, '.format(j)
        # Initialize the network
        if net_type == '1':
            network = FirstOrderCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                                        n_mbon=n_mbon)
        elif net_type == '2':
            network = ExtendedCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                                      n_mbon=n_mbon)
        elif net_type == '3':
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
        full_path = net_path + net_ftype + \
            'trained_nets/{}_mbons/'.format(str(n_mbon).zfill(2))
        fname = full_path + net_fname + fsuff + '.pt'
        if load_net == 'y':
            network.load_state_dict(torch.load(fname))
        else:
            print('\nTraining network with {} MBONs'.format(n_mbon))
            loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep,
                                          p_extinct=p_ext)
            # Save the training losses
            train_loss[:, j] = torch.tensor(loss_hist)
            # Save the network
            if save_train == 'y':
                torch.save(network.state_dict(), fname)

        # Run the test on the network
        print('Evaluating network with {} MBONs'.format(n_mbon))
        for k, task in enumerate(trial_task):
            network.run_eval(second_order_trial, n_trial=n_trial, task=task)
            # network.run_eval(f_trial, n_trial=n_trial, task=task, pos_vt=None)
            # Save the test losses
            test_loss[(k * n_trial):(k + 1) * n_trial, j] = \
                torch.tensor(network.eval_err).detach().numpy()

    # Save the loss data to csv format
    fsuff = '_{}ep_{}hop_{}mbons_error.csv'.format(n_ep, n_hop,
                                                   str(n_mbon).zfill(2))
    if save_train == 'y':
        full_path = net_path + net_ftype + 'train_data/'
        ftrain = full_path + net_fname + fsuff
        np.savetxt(ftrain, train_loss, delimiter=",", header=header)
    if save_test == 'y':
        full_path = net_path + net_ftype + 'test_data/'
        ftest = full_path + net_fname + fsuff
        np.savetxt(ftest, test_loss, delimiter=",", header=header)
