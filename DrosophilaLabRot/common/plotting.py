# Import the required packages
import numpy as np
import matplotlib.pyplot as plt


def plot_trial(network, trial_ls, plt_ttl, plt_lbl, T_vars=None, **kwargs):
    """ Plots a figure similar to Figure 2 from Jiang 2020.

    Runs the network using a novel combination of stimuli, then prints the
    result. Top: time series of the various stimuli (CS and US), as well as
    the target valence and readout. Bottom: activity of eight randomly chosen
    mushroom body output neurons (MBONs).

    Parameters
        network = previously trained RNN
        trial_ls = list of interval functions that compose a trial
        plt_ttl = title of plot
        plt_lbl = labels for CS and US legends
        T_vars: Tuple
            T_vars[0] = T_int = length of trial (in seconds)
            T_vars[1] = T_stim = length of time each stimulus is presented
            T_vars[2] = dt = time step of simulations
    """

    # Define plot font sizes
    label_font = 18
    title_font = 24
    legend_font = 12

    # Set labels
    CS_labels = plt_lbl[0]
    US_labels = plt_lbl[1]

    # Set the time variables
    if T_vars is None:
        T_int, T_stim, dt = network.T_int, network.T_stim, network.dt
    else:
        T_int, T_stim, dt = T_vars
    n_int = len(trial_ls)
    plot_time = np.arange((int(T_int / dt) + 1) * n_int) * dt

    # Run the network
    network.run_eval(trial_ls=trial_ls, T_int=T_int, T_stim=T_stim, dt=dt,
                     n_batch=1, **kwargs)
    r_out = network.eval_rts[-1].numpy().squeeze()
    vt = network.eval_vts[-1].numpy().squeeze()
    vt_opt = network.eval_vt_opts[-1].numpy().squeeze()
    CS_list = network.eval_CS_stim[-1]
    US_list = network.eval_US_stim[-1]

    # Plot the conditioning and test
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 3]})
    ax1.plot(plot_time, vt, label='Readout')
    ax1.plot(plot_time, vt_opt, label='Target')
    for i in range(len(CS_list)):
        ax1.plot(plot_time, CS_list[i].squeeze(), label='{}'.format(CS_labels[i]))
    for i in range(len(US_list)):
        ax1.plot(plot_time, US_list[i].squeeze(), label='{}'.format(US_labels[i]))
    ax1.set_ylabel('Value', fontsize=label_font)
    ax1.set_title(plt_ttl, fontsize=title_font)
    ax1.legend(fontsize=legend_font)

    # Plot the activities of a few MBONs
    plot_neurs = np.random.choice(network.n_mbon, size=8, replace=False)
    r_max = np.max(r_out)
    for i, n in enumerate(plot_neurs):
        ax2.plot(plot_time, (r_out[n, :] / r_max) + i, '-k')
    ax2.set_xlabel('Time', fontsize=label_font)
    ax2.set_ylabel('Normalized MBON Activity', fontsize=label_font)
    ax2.set_yticks([])
    fig.tight_layout()
    plt.show()

    # Return the figure
    return fig
