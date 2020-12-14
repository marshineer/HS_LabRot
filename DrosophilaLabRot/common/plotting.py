# Import the required packages
import numpy as np
import matplotlib.pyplot as plt


def plot_trial(network, plt_ttl, plt_lbl, plt_mbons=8, **kwargs):
    """ Plots a figure similar to Figure 2 from Jiang 2020.

    Runs the network using a novel combination of stimuli, then prints the
    result. Top: time series of the various stimuli (CS and US), as well as
    the target valence and readout. Bottom: activity of randomly chosen
    mushroom body output neurons (MBONs).

    Parameters
        network = previously trained RNN
        plt_ttl = title of plot
        plt_lbl = labels for CS and US legends
        plt_mbons = number of MBONs to plot activity for
    """

    # Define plot font sizes
    label_font = 18
    title_font = 24
    legend_font = 12

    # Set labels
    CS_lbls = plt_lbl[0]
    US_lbls = plt_lbl[1]

    # Pull the data
    rt = network.eval_rts[-1].numpy().squeeze()
    vt = network.eval_vts[-1].numpy().squeeze()
    vt_opt = network.eval_vt_opts[-1].numpy().squeeze()
    CS_list = network.eval_CS_stim[-1]
    US_list = network.eval_US_stim[-1]
    plot_time = np.arange(US_list[0].numpy().squeeze().size) * network.dt

    # Plot the conditioning and test
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 3]})
    ax1.plot(plot_time, vt, label='Readout')
    ax1.plot(plot_time, vt_opt, label='Target')
    for i in range(len(CS_lbls)):
        ax1.plot(plot_time, CS_list[i].squeeze(), label='{}'.format(CS_lbls[i]))
    for i in range(len(US_lbls)):
        ax1.plot(plot_time, US_list[i].squeeze(), label='{}'.format(US_lbls[i]))
    ax1.set_ylabel('Value', fontsize=label_font)
    ax1.set_title(plt_ttl, fontsize=title_font)
    ax1.legend(fontsize=legend_font)

    # Plot the activities of a few MBONs
    plot_neurs = np.random.choice(network.n_mbon, size=plt_mbons, replace=False)
    r_max = np.max(rt)
    for i, n in enumerate(plot_neurs):
        ax2.plot(plot_time, (rt[n, :] / r_max) + i, '-k')
    ax2.set_xlabel('Time', fontsize=label_font)
    ax2.set_ylabel('Normalized MBON Activity', fontsize=label_font)
    ax2.set_yticks([])
    fig.tight_layout()
    # plt.show()

    # Return the figure
    return fig
