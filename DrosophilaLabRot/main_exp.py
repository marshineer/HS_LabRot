import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from network_classes.paper_tasks.no_plasticity_rnn import NoPlasticityRNN
from network_classes.paper_tasks.continual_rnn import ContinualRNN
from common.common import *
from common.plotting import *

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# Initialize the network
net_type = 'first'
if net_type == 'first':
    network = FirstOrderCondRNN()
elif net_type == 'extend':
    network = ExtendedCondRNN()
elif net_type == 'no_plast':
    network = NoPlasticityRNN()
elif net_type == 'continual':
    network = ContinualRNN()

for param in network.parameters():
    print(param.shape)

# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(network.parameters(), lr=lr)

train_bool = True
if train_bool:
    if net_type == 'first' or net_type == 'extend':
        # Classical conditioning
        loss_hist = network.run_train(opti=optimizer, n_epoch=5)
    elif net_type == 'no_plast':
        # No plasticity
        loss_hist = network.run_train(opti=optimizer, T_int=40, n_epoch=1001)
    elif net_type == 'continual':
        # Continual learning
        loss_hist = network.run_train(opti=optimizer, T_int=200, n_epoch=3000)

if train_bool:
    # Plot the loss function
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].plot(loss_hist)
    axes[0].set_xlabel('Epoch', fontsize=label_font)
    axes[0].set_ylabel('Loss', fontsize=label_font)
    axes[1].plot(loss_hist[5:])
    axes[1].set_xlabel('Epoch', fontsize=label_font)
    axes[1].set_ylabel('Loss', fontsize=label_font)
    fig.tight_layout();

trial_ls = [first_order_cond_csp, first_order_test]
plt_ttl = 'First-order conditioning (CS+)'
plt_lbl = (['CS+'], ['US'])
T_int = 30
T_stim = 2
dt = 0.5
T_vars = (T_int, T_stim, dt)
plot_trial(network, trial_ls, plt_ttl, plt_lbl, T_vars, p_ctrl=0)

