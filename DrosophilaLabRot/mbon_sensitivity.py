# Import the required packages
import torch.optim as optim
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from network_classes.paper_tasks.continual_rnn import ContinualRNN
from common.common import *
from common.plotting import *

# Define the path for saving the trained networks and loss plots
net_path = '/home/marshineer/Dropbox/Ubuntu/lab_rotations/sprekeler/' \
           'DrosophilaLabRot/data_store/mbon_sensitivity/'

# Set the training and plotting parameters
net_type = input('Input network type\n'
                 'First-order without recurrence = 1\n'
                 'All classical conditioning without recurrence = 2\n'
                 'Only 2nd-order training without recurrence = 3\n'
                 'Continual learning without recurrence = 4: ')
save_net = input('Do you wish to save this network? y/n ')
# Set the network parameters
T_int = int(input('Input length of training interval: '))
T_stim = int(input('Input length of stimulus presentation: '))
n_ep = int(input('Input number of epochs to train for: '))
n_ep_test = int(input('Input number of epochs to test over: '))
n_hop = int(input('Input number of hops in network: '))
n_seed = int(input('Input the initialization seed for the network weights: '))
if net_type == '4':
    n_stim_avg = int(input('Input average number of stimulus presentations: '))
    n_stim = int(input('Input number of odors per trial: '))
lr = 0.001  # learning rate
# Set the parameters for sensitivity training
n_mbon_0 = int(input('Set the initial number of MBONs in the network: '))
n_vals = int(input('Set the number of MBON values to test: '))
inc_type = input('Set the type of increment for the MBONs (lin = linear, '
                 'exp = exponential): ')

# Initialize variables to store training data
header = ''
train_loss = torch.zeros(n_ep, n_vals)
# test_loss = torch.zeros(n_ep_test, n_vals)
test_loss = np.zeros((n_ep_test, n_vals))
n_mbon = n_mbon_0
n_mbon_vec = np.zeros(n_vals)

for i in range(n_vals):
    # Update the header for this network's loss data
    header += '{} MBONs, '.format(n_mbon)
    # Initialize the network
    if net_type == '1':
        net_fname = 'knockout_fo'
        network = FirstOrderCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                                    n_mbon=n_mbon, n_seed=n_seed)
    elif net_type == '2':
        net_fname = 'knockout_ac'
        network = ExtendedCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                                  n_mbon=n_mbon, n_seed=n_seed)
    elif net_type == '3':
        net_fname = 'knockout_so'
        network = ExtendedCondRNN(T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                                  n_mbon=n_mbon, n_seed=n_seed)
    elif net_type == '4':
        net_fname = 'knockout_cl_{}stim_{}avg'.format(n_stim, n_stim_avg)
        network = ContinualRNN(n_trial_avg=n_stim_avg, n_trial_odors=n_stim,
                               T_int=T_int, T_stim=T_stim, n_hop=n_hop,
                               n_mbon=n_mbon, n_seed=n_seed)
    # Define the model's optimizer
    optimizer = optim.RMSprop(network.parameters(), lr=lr)

    # Set the time step of the network
    dt = network.dt
    # Train the network
    print('\nTraining network with {} MBONs'.format(n_mbon))
    if net_type == '1':
        # First-order conditioning with no recurrence
        trial_ls = [first_order_cond_csp, first_order_test,
                    first_order_csm, first_order_test]
        plt_ttl = '1st-order Conditioning'
        loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)
    elif net_type == '2':
        # All classical conditioning with no recurrence
        trial_ls = [first_order_cond_csp, first_order_test, extinct_test,
                    first_order_cond_csp, second_order_cond, second_order_test]
        plt_ttl = 'All Classical Conditioning'
        loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)
    elif net_type == '3':
        # Second-order conditioning with no recurrence
        trial_ls = [first_order_cond_csp, second_order_cond, second_order_test]
        plt_ttl = '2nd-order Conditioning'
        loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep, p_extinct=0)
    elif net_type == '4':
        # Continual learning with no recurrence
        trial_ls = [continual_trial]
        plt_ttl = 'Continual Learning'
        loss_hist = network.run_train(opti=optimizer, n_epoch=n_ep)

    # Save the training losses
    train_loss[:, i] = torch.tensor(loss_hist)
    # Save the network
    fsuff = '_{}ep_{}hop_{}MBONs'.format(n_ep, n_hop, n_mbon)
    fname = net_path + 'trained_nets/' + net_fname + fsuff + '.pt'
    torch.save(network.state_dict(), fname)

    # Run the test on the network
    network.run_eval(trial_ls=trial_ls, T_int=T_int, T_stim=T_stim, dt=dt,
                     n_batch=30, n_epoch=n_ep_test)
    # Save the test losses
    test_loss[:, i] = torch.tensor(network.eval_loss).detach().numpy()

    # Save the number of MBONs
    n_mbon_vec[i] = n_mbon
    # Increment the number of MBONs
    if inc_type == 'exp':
        n_mbon = n_mbon_0 ** (i + 2)
    elif inc_type == 'lin':
        n_mbon += 1

# Save the loss data to csv format
if save_net:
    fsuff = '_' + inc_type
    ftrain = net_path + 'loss_data/' + net_fname + fsuff + '_training_losses.csv'
    np.savetxt(ftrain, train_loss, delimiter=",", header=header)
    ftest = net_path + 'loss_data/' + net_fname + fsuff + '_testing_losses.csv'
    np.savetxt(ftest, test_loss, delimiter=",", header=header)

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# Calculate statistics on losses
loss_avg = np.mean(test_loss, axis=0)
loss_std = np.std(test_loss, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.errorbar(n_mbon_vec, loss_avg, loss_std, None, '-o')
ax.set_xlabel('Number of MBONs', fontsize=label_font)
ax.set_ylabel('Average Loss', fontsize=label_font)
ax.set_title('MBON Sensitivity ({})'.format(plt_ttl), fontsize=title_font)

# Save the losses plot
if save_net == 'y':
    plot_path = net_path + 'loss_plots/' + net_fname + fsuff + '_testing_losses.png'
    plt.savefig(plot_path, bbox_inches='tight')

# Show the plot
plt.show()
