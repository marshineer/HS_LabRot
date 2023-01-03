# Import the required packages
import numpy as np
import matplotlib.pyplot as plt


def plot_trial(network, plt_ttl, plt_lbl, plt_mbons=8, **kwargs):
    """ Plots a figure similar to Figure 2 from Jiang 2020.

    Plots the results of the last trial evaluated by the network. Top: time
    series of the various stimuli (CS and US), as well as the target valence
    and readout. Bottom: activity of randomly chosen mushroom body output
    neurons (MBONs).

    Parameters
        network = previously trained RNN
        plt_ttl = title of plot
        plt_lbl = labels for CS and US legends
        plt_mbons = number of MBONs to plot activity for
    """

    # Define plot font sizes
    label_font = 24
    title_font = 28
    legend_font = 18

    # Define plot characteristics
    fig_w = 12
    if type(network).__name__ == 'FirstOrderCondRNN':
        fig_w = 8

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
    plt.rc('xtick', labelsize=legend_font)
    plt.rc('ytick', labelsize=legend_font)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 3]})
    ax1.plot(plot_time, vt, label='Readout')
    ax1.plot(plot_time, vt_opt, label='Target')
    # # Note: the number of stimuli plotted is determined by the label list
    # # This is determined by the output of the trial function
    # for i in range(len(CS_lbls)):
    #     ax1.plot(plot_time, CS_list[i].squeeze(), label='{}'.format(CS_lbls[i]))
    # for i in range(len(US_lbls)):
    #     ax1.plot(plot_time, US_list[i].squeeze(), label='{}'.format(US_lbls[i]))
    ax1.set_ylabel('Valence', fontsize=label_font)
    ax1.set_yticks([])
    ax1.set_title(plt_ttl, fontsize=title_font)
    if type(network).__name__ != 'ContinualRNN':
        ax1.set_ylim(-0.1, 1.1)

    # Plot the activities of a few MBONs
    plot_neurs = np.random.choice(network.n_mbon, size=plt_mbons, replace=False)
    if type(network).__name__ == 'ContinualRNN':
        r_max = np.max(rt[-network.n_dan:, :])
        for i, n in enumerate(plot_neurs):
            ax2.plot(plot_time, (rt[-(n + 1), :] / r_max) + i, '-k')
        ax2.set_ylabel('Normalized DAN Activity', fontsize=label_font)
        print('')
    else:
        r_max = np.max(rt)
        for i, n in enumerate(plot_neurs):
            ax2.plot(plot_time, (rt[n, :] / r_max) + i, '-k')
        ax2.set_ylabel('Normalized MBON Activity', fontsize=label_font)
    ax2.set_xlabel('Time', fontsize=label_font)
    ax2.set_yticks([])
    # plt.show()

    # Plot the US and CS as vertical bars
    # Note: the number of stimuli plotted is determined by the label list
    # This is determined by the output of the trial function
    # l_stim = int(network.T_stim / network.dt)
    l_stim = network.T_stim
    cs_colours = ['indigo', 'c', 'gray', 'gray']
    us_colours = ['g', 'r']
    for i in range(len(CS_lbls)):
        CSi_st = (np.where(np.diff(CS_list[i].squeeze()) == 1)[0] + 1) // 2
        for j in range(CSi_st.size):
            if j == 0:
                label_j = CS_lbls[i]
            else:
                label_j = '_nolegend_'
            for ax in [ax1, ax2]:
                ax.axvspan(CSi_st[j], CSi_st[j] + l_stim, alpha=0.2,
                           color=cs_colours[i], label=label_j)
    for i in range(len(US_lbls)):
        USi_st = (np.where(np.diff(US_list[i].squeeze()) == 1)[0] + 1) // 2
        for j in range(USi_st.size):
            if j == 0:
                label_j = US_lbls[i]
            else:
                label_j = '_nolegend_'
            for ax in [ax1, ax2]:
                ax.axvspan(USi_st[j], USi_st[j] + l_stim, alpha=0.2,
                           color=us_colours[i], label=label_j)
    ax1.legend(fontsize=legend_font, bbox_to_anchor=(1, 1.05), loc='upper left')
    fig.tight_layout()

    # Return the figure
    return fig, (ax1, ax2)
