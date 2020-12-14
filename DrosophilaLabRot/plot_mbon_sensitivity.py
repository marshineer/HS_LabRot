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

# Set which sensitivity study to plot
#  '1' = first-order network
#  '2' = second-order network
#  '3' = second-order network (no extinction)
net_type = '3'
# Set the parameters
n_mbon_0 = 2  # Initial number of MBONs
n_vals = 5  # Number of values tested

# Initialize values
err_avg = np.zeros(n_vals)
err_std = np.zeros(n_vals)
mbon_vec = np.zeros(n_vals)

# For each mbon value, calculate the average error over all networks
for i in range(n_vals):
    # Set the number of MBONs
    n_mbon = n_mbon_0 ** (i + 1)
    # n_mbon = n_mbon_0 + i

    # Set the file path and name
    if net_type == '1':
        csv_path = 'first_order_nets/test_data/'
        fname = 'first_order_2000ep_1hop_{}mbons_error.csv'\
            .format(str(n_mbon).zfill(2))
        plt_name = 'first_order_2000ep_1hop_mbon_sensitivity.png'
        plt_ttl = 'First-order Conditioning'
    elif net_type == '2':
        csv_path = 'second_order_nets/test_data/'
        fname = 'second_order_5000ep_1hop_{}mbons_error.csv'\
            .format(str(n_mbon).zfill(2))
        plt_name = 'second_order_5000ep_1hop_mbon_sensitivity.png'
        plt_ttl = 'Second-order Conditioning'
    elif net_type == '3':
        csv_path = 'second_order_only_nets/test_data/'
        fname = 'second_order_no_extinct_5000ep_1hop_{}mbons_error.csv'\
            .format(str(n_mbon).zfill(2))
        plt_name = 'second_order_no_extinct_5000ep_1hop_mbon_sensitivity.png'
        plt_ttl = 'Second-order, no Extinction'

    # Load the data
    data_path = net_path + csv_path + fname
    test_data = np.loadtxt(data_path, delimiter=',', skiprows=1)

    # Calcualte the average error and its std dev
    err_avg[i] = np.mean(test_data)
    err_std[i] = np.std(test_data)
    mbon_vec[i] = n_mbon

# Plot the sensitivity results
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.errorbar(mbon_vec, err_avg, err_std, None, '-o')
ax.set_xlabel('Number of MBONs', fontsize=label_font)
ax.set_ylabel('Average Loss', fontsize=label_font)
ax.set_title('MBON Sensitivity ({})'.format(plt_ttl), fontsize=title_font)

# Save the losses plot
plot_path = net_path + csv_path + plt_name
plt.savefig(plot_path, bbox_inches='tight')

# Show the plot
plt.show()
