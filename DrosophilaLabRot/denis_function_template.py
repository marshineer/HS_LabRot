# Import the required packages
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from common.common import *
from common.plotting import *
from common.common import *


def trial_fnc_ex(net, W_in, T_vars, n_batch, **kwargs):
    """

    """

    # Set the time variables
    T_int, T_stim, dt = T_vars
    time_int = torch.arange(0, T_int + dt / 10, dt)
    t_len = time_int.shape[0]
    time_zeros = torch.zeros(n_batch, t_len)

    # Generate odors and context (odor = KC = CS, context = ext = US)
    r_kc_csp, r_ext = net.gen_r_kc_ext(n_batch, **kwargs)  # CS+ odor
    r_kc_csm, _ = net.gen_r_kc_ext(n_batch, **kwargs)  # CS- odor
    r_kc_novel, _ = net.gen_r_kc_ext(n_batch, **kwargs)  # novel odor
    r_ext_zeros = torch.zeros_like(r_ext)
    trial_odors = [r_kc_csp, r_kc_csm, r_kc_novel]
    int_list = [int_cond_cs,
                int_cs_alone,
                int_cond_cs2,
                int_cond_cs2,
                int_test_cs,
                int_cs_alone,
                int_cs_alone]
    n_int = len(int_list)
    r_in_list = [(r_kc_csp, r_ext),
                 (r_kc_csm, r_ext_zeros),
                 ([r_kc_csp, r_kc_csp], r_ext_zeros),
                 ([r_kc_csm, r_kc_csm], r_ext_zeros),
                 (r_kc_csp, r_ext_zeros),
                 (r_kc_csm, r_ext_zeros),
                 (r_kc_novel, r_ext_zeros)]
    # These are the default values, so we don't need to include them,
    #  but I've shown them here for clarity
    wt_arg_list = {'ud_wts': True, 'ko_wts': None}

    # Lists to store activities, weights, readouts and target valences
    # In this example, everything is saved
    rts = []
    Wts = []
    wts = []
    vts = []
    vt_opts = []
    time_CSp = torch.zeros(n_batch, t_len * n_int)
    time_CSm = torch.zeros_like(time_CSp)
    time_CS_novel = torch.zeros_like(time_CSp)
    time_US = torch.zeros_like(time_CSp)

    for i in range(n_int):
        # Calculate the CS stimulus presentation times
        st_times, st_len = gen_int_times(n_batch, dt, T_stim, **kwargs)
        # Calculate the interval inputs for a CS+ conditioning interval
        int_fnc = int_list[i]
        r_in = r_in_list[i]
        f_in = int_fnc(t_len, st_times, st_len, r_in, n_batch)
        r_kct, r_extt, stim_ls, vt_opt = f_in

        # Run the forward pass
        net_out = net(r_kct, r_extt, time_int, n_batch, W_in, None, wt_arg_list,
                      **kwargs)
        rt_int, (Wt_int, wt_int), vt_int = net_out
        # Pass the KC->MBON weights to the next interval
        W_in = (Wt_int[-1], wt_int[-1])

        # Append the interval outputs to lists
        rts += rt_int
        Wts += Wt_int[-1]
        wts += wt_int[-1]
        vts += vt_int
        vt_opts.append(vt_opt)

        # TODO: Maybe use torch.cat here instead
        # Store the CS+ odor time series
        if i in [0, 4]:
            time_CSp[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        elif i == 2:
            time_CSp[:, i * t_len:(i + 1) * t_len] = stim_ls[0][0]
            time_CSp[:, i * t_len:(i + 1) * t_len] += stim_ls[0][1]
        else:
            time_CSp[:, i * t_len:(i + 1) * t_len] = time_zeros
        # Store the CS- odor time series
        if i in [1, 5]:
            time_CSm[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        elif i == 3:
            time_CSm[:, i * t_len:(i + 1) * t_len] = stim_ls[0][0]
            time_CSm[:, i * t_len:(i + 1) * t_len] += stim_ls[0][1]
        else:
            time_CSm[:, i * t_len:(i + 1) * t_len] = time_zeros
        # Store the novel odor time series
        if i == 6:
            time_CS_novel[:, i * t_len:(i + 1) * t_len] = stim_ls[0]
        else:
            time_CS_novel[:, i * t_len:(i + 1) * t_len] = time_zeros
        # Store US time series
        time_US[:, i * t_len:(i + 1) * t_len] = stim_ls[1]
        time_all_CS = [time_CSp, time_CSm, time_CS_novel]
        stim_list = [time_all_CS, time_US]

    # Calculate the trial error
    vt_trial = torch.stack(vts, dim=-1).detach()
    vt_opt_trial = torch.cat(vt_opts, dim=-1).detach()
    err_trial = cond_err(vt_trial, vt_opt_trial).item()

    # Save the recurrent neuron activites
    rts_trial = torch.stack(rts, dim=-1).detach()
    # Save the KC->MBON weights from the end of each interval
    Wts_trial = torch.stack(Wts, dim=-1).detach()
    wts_trial = torch.stack(wts, dim=-1).detach()

    return rts_trial, Wts_trial, wts_trial, vt_trial, vt_opt_trial, err_trial,\
        trial_odors, stim_list


# Run a network using the above trial function
# Initialize the network
network = ExtendedCondRNN(n_hop=1)

# Print the parameter shapes to check
for param in network.parameters():
    print(param.shape)

# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(network.parameters(), lr=lr)

# Load the network (you will have to set the absolute path)
net_path = '/home/marshineer/Dropbox/Ubuntu/lab_rotations/sprekeler/' \
           'DrosophilaLabRot/data_store/extension_nets/'
# This is the filename for trained networks where recurrence is knocked out
# The "so" indicates that it is trained only on second-order tasks
# For other files:
#   fo = first-order only
#   ac = all classical (conditioning)
#   cl = continual (learning)
net_fname = 'trained_knockout_so_5000ep_1hop'
fname = net_path + 'trained_nets/' + net_fname + '.pt'
network.load_state_dict(torch.load(fname))

# Plot a single trial
# save_plot = input('Do you wish to save the plot? y/n ')
save_plot = 'n'
# Second-order conditioning with no recurrence
network.run_eval(trial_fnc_ex)
plt_fname = 'knockout_so_5000ep'
plt_ttl = 'Second-order (No Recurrence) Conditioning'
plt_lbl = (['CS+', 'CS-', 'Novel CS'], ['US'])

# Plot the trial
T_vars = (network.T_int, network.T_stim, network.dt)
fig = plot_given(network, plt_ttl, plt_lbl, pos_vt=None)

# Save the losses plot
plot_path = net_path + 'trial_plots/' + plt_fname + '_trial.png'
if save_plot == 'y':
    fig.savefig(plot_path, bbox_inches='tight')

print(network.eval_vts[0].shape)