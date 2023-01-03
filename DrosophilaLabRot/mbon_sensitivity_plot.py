# Import the required packages
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path for saving the trained networks and loss plots
dir_path = os.path.dirname(__file__)
net_path = dir_path + '/data_store/mbon_sensitivity/'

# Define plot font sizes
label_font = 18
title_font = 24
legend_font = 12

# The only user inputs required are net_type and inc_type
# Set which sensitivity study to plot
#  '1' = first-order network
#  '2' = second-order network
#  '3' = second-order network (no extinction)
net_type = '3'
# Set whether plot should be exponential or linear
# The networks must have been run for each case
inc_type = 'lin'
if inc_type == 'exp':
    n_mbon_0 = 2
    n_vals = 5
    mbon_list = (np.ones(n_vals) * n_mbon_0) ** (np.arange(n_vals) + 1)
elif inc_type == 'lin':
    n_mbon_0 = 6
    n_vals = 7
    mbon_list = np.append(np.arange(n_vals) + n_mbon_0, 16)
    # mbon_list = np.append(np.arange(n_vals) + n_mbon_0, (16, 32))
    print(mbon_list)
    # print(np.append(mbon_list, 16))
    # print(np.concatenate(mbon_list, np.array(16)))

# Initialize values
n_vals = mbon_list.size
err_avg = np.zeros(n_vals)
err_std = np.zeros(n_vals)
mbon_vec = np.zeros(n_vals)
err_list = []

# For each mbon value, calculate the average error over all networks
for i, n_mbon in enumerate(mbon_list):
    # Set the number of MBONs
    n_mbon = int(n_mbon)

    # Set the file path and name
    if net_type == '1':
        csv_path = 'first_order_nets/test_data/'
        fname = 'first_order_2000ep_1hop_{}mbons_error.csv'\
            .format(str(n_mbon).zfill(2))
        plt_name = 'first_order_2000ep_1hop_mbon_sensitivity'
        plt_ttl = 'First-order Conditioning'
    elif net_type == '2':
        csv_path = 'second_order_nets/test_data/'
        fname = 'second_order_5000ep_1hop_{}mbons_error.csv'\
            .format(str(n_mbon).zfill(2))
        plt_name = 'second_order_5000ep_1hop_mbon_sensitivity'
        plt_ttl = 'Second-order Conditioning'
    elif net_type == '3':
        csv_path = 'second_order_only_nets/test_data/'
        fname = 'second_order_no_extinct_5000ep_1hop_{}mbons_error.csv'\
            .format(str(n_mbon).zfill(2))
        plt_name = 'second_order_no_extinct_5000ep_1hop_mbon_sensitivity'
        plt_ttl = 'Second-order, no Extinction'

    # Load the data
    data_path = net_path + csv_path + fname
    test_data = np.loadtxt(data_path, delimiter=',', skiprows=1)

    # Calculate the average error and its std dev
    err_avg[i] = np.mean(test_data)
    err_std[i] = np.std(test_data)
    mbon_vec[i] = n_mbon
    avg_net_err = np.mean(test_data, axis=0)
    err_list.append(avg_net_err)

# Plot the sensitivity results
n_points = mbon_vec.size
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# ax.errorbar(mbon_vec, err_avg, err_std, None, '-o')
# ax.plot(np.arange(n_points) + 1, err_avg, '-x')
# ax.boxplot(err_list)
# ax.set_xticks(np.arange(n_points) + 1)
# ax.set_xticklabels(mbon_vec)
ax.boxplot(err_list, labels=mbon_vec, showfliers=False)
ax.set_xlabel('Number of MBONs', fontsize=label_font)
ax.set_ylabel('Average Readout MSE', fontsize=label_font)
# ax.set_title('MBON Sensitivity ({})'.format(plt_ttl), fontsize=title_font)
fig.tight_layout()

# Save the losses plot
save_plot = False
if save_plot:
    plot_path = net_path + csv_path + plt_name + '_{}.png'.format(inc_type)
    plt.savefig(plot_path, bbox_inches='tight')

# Show the plot
plt.show()
