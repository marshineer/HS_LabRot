# Import the required packages
from network_classes.paper_extensions.knockout_recur import NoRecurFirstO, \
    NoRecurContinual, NoRecurExtended
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
if net_type == '1':
    plot_type = input('Input type of trial to run (CS+ or CS-): ')
elif net_type == '2':
    plot_type = input('Input type of trial to run (ext or 2nd): ')
save_plot = input('Do you wish to save the plot? y/n ')
fsuff = input('Enter a custom filename suffix, if desired: ')
T_int = int(input('Input length of training interval: '))
T_stim = int(input('Input length of stimulus presentation: '))
n_ep = int(input('Input number of epochs network was trained for: '))

# Load the network
if net_type == '1':
    network = NoRecurFirstO(T_int=T_int)
    net_fname = 'trained_knockout_fo_{}ep'.format(n_ep)
if net_type == '2':
    network = NoRecurExtended(T_int=T_int)
    net_fname = 'trained_knockout_ac_{}ep'.format(n_ep)
if net_type == '3':
    network = NoRecurExtended(T_int=T_int)
    net_fname = 'trained_knockout_so_{}ep'.format(n_ep)
elif net_type == '4':
    n_stim_avg = int(input('Input average number of stimulus'
                           'presentations: '))
    n_stim = int(input('Input number of odors per trial: '))
    network = NoRecurContinual(n_trial_avg=n_stim_avg, n_trial_odors=n_stim,
                               T_int=T_int, T_stim=T_stim)
    net_fname = 'trained_knockout_cl_{}stim_{}avg_{}ep'.format(n_stim,
                                                               n_stim_avg,
                                                               n_ep)

# Define network parameters and load network
fname = net_path + 'trained_nets/' + net_fname + fsuff + '.pt'
network.load_state_dict(torch.load(fname))

# Plot trials for trained networks
if net_type == '1':
    # First-order conditioning with no recurrence
    if plot_type == 'CS+':
        trial_ls = [first_order_cond_csp, first_order_test]
        plt_fname = 'knockout_fo_csp_{}ep'.format(n_ep)
        plt_ttl = 'First-order (No Recurrence) Conditioning (CS+)'
        plt_lbl = (['CS+'], ['US'])
    elif plot_type == 'CS-':
        trial_ls = [first_order_csm, first_order_test]
        plt_fname = 'knockout_fo_csm_{}ep'.format(n_ep)
        plt_ttl = 'First-order (No Recurrence) Conditioning (CS-)'
        plt_lbl = (['CS-'], ['US'])
elif net_type == '2':
    # All classical conditioning with no recurrence
    if plot_type == 'ext':
        trial_ls = [first_order_cond_csp, first_order_test, extinct_test]
        plt_fname = 'knockout_all_ex_{}ep'.format(n_ep)
        plt_ttl = 'Extinction (No Recurrence) Conditioning'
        plt_lbl = (['CS+'], ['US'])
    elif plot_type == '2nd':
        trial_ls = [first_order_cond_csp, second_order_cond, second_order_test]
        plt_fname = 'knockout_all_so_{}ep'.format(n_ep)
        plt_ttl = 'Second-order (No Recurrence) Conditioning'
        plt_lbl = (['CS1', 'CS2'], ['US'])
elif net_type == '3':
    # Second-order conditioning with no recurrence
    trial_ls = [first_order_cond_csp, second_order_cond, second_order_test]
    plt_fname = 'knockout_so_{}ep'.format(n_ep)
    plt_ttl = 'Second-order (No Recurrence) Conditioning'
    plt_lbl = (['CS1', 'CS2'], ['US'])
elif net_type == '4':
    # Continual learning with no recurrence
    trial_ls = [continual_trial]
    plt_fname = 'knockout_cl_{}stim_{}avg_{}ep'.format(n_stim, n_stim_avg, n_ep)
    plt_ttl = 'Continual Learning (No Recurrence)'
    cs_lbl = []
    for j in range(n_stim):
        cs_lbl.append('CS{}'.format(j + 1))
    plt_lbl = (cs_lbl, ['US1', 'US2'])

# Plot the trial
T_vars = (network.T_int, network.T_stim, network.dt)
fig = plot_trial(network, trial_ls, plt_ttl, plt_lbl, T_vars, pos_vt=True)

# Save the losses plot
plot_path = net_path + 'trial_plots/' + plt_fname + '_trial' + fsuff + '.png'
if save_plot == 'y':
    fig.savefig(plot_path, bbox_inches='tight')
