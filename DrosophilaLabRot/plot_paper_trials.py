# Import the required packages
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from network_classes.paper_tasks.no_plasticity_rnn import NoPlasticityRNN
from network_classes.paper_tasks.continual_rnn import ContinualRNN
from common.common import *
from common.plotting import *

# Define the path for saving the trained networks and loss plots
net_path = '/home/marshineer/Dropbox/Ubuntu/lab_rotations/sprekeler/' \
           'DrosophilaLabRot/data_store/paper_nets/'

# Set the training and plotting parameters
man_input = input('Manually select the trial to plot? y/n ')
save_plot = input('Do you wish to save the plot? y/n ')
if man_input == 'y':
    net_type = input('Input network type (first, extinct, second, no_plast or '
                     'continual: ')
    if net_type == 'first' or net_type == 'no_plast':
        plot_type = input('Input type of trial to run (CS+ or CS-): ')
elif man_input == 'n':
    net_type_list = ['first', 'first', 'extinct', 'second', 'no_plast', 'no_plast',
                     'continual']
    plot_type_list = ['CS+', 'CS-', '-', '-', 'CS+', 'CS-', '-']
else:
    raise Exception('This is not a valid response')

if man_input == 'y':
    n_loops = 1
else:
    n_loops = len(net_type_list)
for i in range(n_loops):
    if man_input == 'n':
        net_type = net_type_list[i]
        plot_type = plot_type_list[i]

    # Load the network
    if net_type == 'first':
        network = FirstOrderCondRNN()
        if man_input == 'y':
            n_ep = int(input('Input number of epochs to train for: '))
        else:
            n_ep = 2000
        net_fname = 'trained_first_order_{}ep'.format(n_ep)
    elif net_type == 'extinct' or net_type == 'second':
        network = ExtendedCondRNN()
        if man_input == 'y':
            n_ep = int(input('Input number of epochs to train for: '))
        else:
            n_ep = 5000
        net_fname = 'trained_all_classic_{}ep'.format(n_ep)
    elif net_type == 'no_plast':
        network = NoPlasticityRNN(T_int=40)
        if man_input == 'y':
            n_ep = int(input('Input number of epochs to train for: '))
            n_odors = int(input('Input number of odors for network: '))
        else:
            n_odors = 10
            n_ep = 2000
        net_fname = 'trained_no_plasticity_{}odor_{}ep'.format(n_odors, n_ep)
    elif net_type == 'continual':
        network = ContinualRNN(T_int=200)
        if man_input == 'y':
            n_ep = int(input('Input number of epochs to train for: '))
            n_stim_avg = int(input('Input average number of stimulus'
                                   'presentations: '))
            n_stim = int(input('Input number of odors per trial: '))
        else:
            n_stim_avg = 2
            n_stim = 4
            n_ep = 5000
        net_fname = 'trained_continual_{}stim_{}avg_{}ep'.format(n_stim,
                                                                 n_stim_avg,
                                                                 n_ep)

    # Define network parameters and load network
    fname = net_path + 'trained_nets/' + net_fname + '.pt'
    network.load_state_dict(torch.load(fname))

    # Plot trials for trained networks
    if net_type == 'first':
        # Classical conditioning
        if plot_type == 'CS+':
            trial_ls = [first_order_cond_csp, first_order_test]
            plt_fname = 'first_order_csp_{}ep'.format(n_ep)
            plt_ttl = 'First-order Conditioning (CS+)'
            plt_lbl = (['CS+'], ['US'])
        elif plot_type == 'CS-':
            trial_ls = [first_order_csm, first_order_test]
            plt_fname = 'first_order_csp_{}ep'.format(n_ep)
            plt_ttl = 'First-order Conditioning (CS-)'
            plt_lbl = (['CS-'], ['US'])
    elif net_type == 'extinct':
        # Extinction conditioning
        trial_ls = [first_order_cond_csp, first_order_test, extinct_test]
        plt_fname = 'extinction_{}ep'.format(n_ep)
        plt_ttl = 'Extinction Conditioning'
        plt_lbl = (['CS+'], ['US'])
    elif net_type == 'second':
        # Second-order conditioning
        trial_ls = [first_order_cond_csp, second_order_cond, second_order_test]
        plt_fname = 'second_order_{}ep'.format(n_ep)
        plt_ttl = 'Second-order Conditioning'
        plt_lbl = (['CS1', 'CS2'], ['US'])
    elif net_type == 'no_plast':
        # No plasticity
        if plot_type == 'CS+':
            trial_ls = [no_plasticity_trial]
            plt_fname = 'no_plasticity_csp_{}odor_{}ep'.format(n_odors, n_ep)
            plt_ttl = 'No Plasticity (CS+)'
            plt_lbl = (['CS+', 'CS-'], ['US'])
        elif plot_type == 'CS-':
            trial_ls = [no_plasticity_trial]
            plt_fname = 'no_plasticity_csm_{}odor_{}ep'.format(n_odors, n_ep)
            plt_ttl = 'No Plasticity (CS-)'
            plt_lbl = (['CS+', 'CS-'], ['US'])
    elif net_type == 'continual':
        # Continual learning
        trial_ls = [continual_trial]
        plt_fname = 'continual_{}stim_{}avg_{}ep'.format(n_stim, n_stim_avg,
                                                         n_ep)
        plt_ttl = 'Continual Learning'
        cs_lbl = []
        for j in range(n_stim):
            cs_lbl.append('CS{}'.format(j + 1))
        plt_lbl = (cs_lbl, ['US1', 'US2'])

    # Plot the trial
    T_vars = (network.T_int, network.T_stim, network.dt)
    fig = plot_trial(network, trial_ls, plt_ttl, plt_lbl, T_vars, pos_vt=True)

    # Save the losses plot
    plot_path = net_path + 'trial_plots/' + plt_fname + '_trial.png'
    if save_plot == 'y':
        fig.savefig(plot_path, bbox_inches='tight')
