import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from not_used.base_rnn_V2 import FirstOrderCondRNN

# Initialize the network
classic_net = FirstOrderCondRNN()
for param in classic_net.parameters():
    print(param.shape)

# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(classic_net.parameters(), lr=lr)

loss_hist = classic_net.train_net(opti=optimizer, n_epoch=2000)

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12
# Plot the loss function
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].plot(loss_hist)
axes[0].set_xlabel('Epoch', fontsize=label_font)
axes[0].set_ylabel('Loss', fontsize=label_font)
axes[1].plot(loss_hist[5:])
axes[1].set_xlabel('Epoch', fontsize=label_font)
axes[1].set_ylabel('Loss', fontsize=label_font)
fig.tight_layout()
plt.show()

loss_hist = classic_net.train_net(opti=optimizer, n_epoch=200)
print(np.mean(loss_hist), np.std(loss_hist))
