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
n_odor = 1

# Initialize the network
network = NoPlasticityRNN(T_int=T_int, n_odors=n_odor)
# Print the parameter shapes to check
for param in network.parameters():
    print(param.shape)
# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(network.parameters(), lr=lr)

# Determine user inputs (load network and save plots)
# save_plot = input('Do you wish to save the trial plot? y/n ')

# Set the network filepath
o_str = str(n_odor).zfill(2)
net_fname = 'trained_no_plasticity_{}odor_{}ep'.format(o_str, n_ep)
fname = net_path + 'trained_nets/' + net_fname + '.pt'
# Load the existing network
network.load_state_dict(torch.load(fname))

# # Plot and save trials for the network
# plot_list = ['CS+', 'CS-']
# plot_lbls = [['CS+'], ['CS+', 'CS-']]
# plot_fsuff = ['csp', 'csm']
# for i in range(len(plot_list)):
#     network.run_eval(no_plasticity_trial, task=plot_list[i], pos_vt=True)
#     plt_ttl = 'No Plasticity ({} Trial)'.format(plot_list[i])
#     plt_lbl = (plot_lbls[i], ['US'])
#
#     # Plot the trial
#     fig, _ = plot_trial(network, plt_ttl, plt_lbl)
#
#     # Save the losses plot
#     if save_plot == 'y':
#         plt_fname = 'no_plasticity_{}odor_{}ep_{}_trial.png'\
#             .format(o_str, n_ep, plot_fsuff[i])
#         plot_path = net_path + 'trial_plots/' + plt_fname
#         fig.savefig(plot_path, bbox_inches='tight')
#     else:
#         plt.show()

n_tr = 100
n_ba = 20
network.run_eval(no_plasticity_trial, task='CS+', n_batch=n_ba, n_trial=n_tr,
                 pos_vt=True)
csp_err = np.stack(network.eval_err)
network.run_eval(no_plasticity_trial, task='CS-', n_batch=n_ba, n_trial=n_tr,
                 pos_vt=True)
csm_err = np.stack(network.eval_err)
train_err = network.run_train(opti=optimizer, n_batch=n_ba, n_epoch=n_tr)
avg_err = np.mean(np.stack((csp_err, csm_err), axis=0), axis=0)
print(avg_err.shape)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(csp_err)
ax.plot(csm_err)
ax.plot(avg_err)
ax.plot(train_err)

plt.show()
