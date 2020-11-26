# Import packages
import torch.optim as optim


# Define the type of network to run
# 'classic', 'no plasticity', 'continual'
net_type = 'classic'
if net_type == 'classic':
    import network_classes.paper_tasks.all_conditioning_rnn as rnn
    task = 'first-order'
    # task = 'all_tasks'
elif net_type == 'no plasticity':
    import network_classes.paper_tasks.no_plasticity_rnn as rnn
elif net_type == 'continual':
    import network_classes.paper_tasks.continual_rnn as rnn

# Initialize the network
main_net = rnn.DrosophilaRNN()
for param in main_net.parameters():
    print(param.shape)

# Define the model's optimizer
lr = 0.001
optimizer = optim.RMSprop(main_net.parameters(), lr=lr)

# Press Shift+F10 to execute it or replace it with your code.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    r_out, Wt, vt, vt_opt, loss_hist, _ = rnn.train_net(main_net, optimizer,
                                                        task=task,
                                                        n_epochs=2000)
