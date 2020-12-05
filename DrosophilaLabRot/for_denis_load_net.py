# Import the required packages
import torch.optim as optim
from network_classes.paper_extensions.knockout_recur import NoRecurFirstO, \
    NoRecurContinual, NoRecurExtended
from common.common import *
from common.plotting import *

# Sets the network parameters
T_int = 30
T_stim = 2

# Initialize the network
network = NoRecurExtended(T_int=T_int, T_stim=T_stim)

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
net_fname = 'trained_knockout_so_5000ep_one_hop'
fname = net_path + 'trained_nets/' + net_fname + '.pt'
network.load_state_dict(torch.load(fname))

# Plot a single trial
save_plot = input('Do you wish to save the plot? y/n ')
# Second-order conditioning with no recurrence
trial_ls = [first_order_cond_csp, second_order_cond, second_order_test]
plt_fname = 'knockout_so_5000ep'
plt_ttl = 'Second-order (No Recurrence) Conditioning'
plt_lbl = (['CS1', 'CS2'], ['US'])

# Plot the trial
T_vars = (network.T_int, network.T_stim, network.dt)
fig = plot_trial(network, trial_ls, plt_ttl, plt_lbl, T_vars, pos_vt=True)

# Save the losses plot
plot_path = net_path + 'trial_plots/' + plt_fname + '_trial.png'
if save_plot == 'y':
    fig.savefig(plot_path, bbox_inches='tight')
