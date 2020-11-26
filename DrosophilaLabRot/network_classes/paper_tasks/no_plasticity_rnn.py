# Import the required packages
import torch
from torch.autograd import Variable
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN


class NoPlasticityRNN(FirstOrderCondRNN):
    def __init__(self, *, n_odors=10):
        super().__init__()
        # Set the static KC->MBON weights
        W_kc_mbon_max = 0.05
        self.W_kc_mbon = Variable(torch.rand(self.n_mbon, self.n_kc) *
                                  W_kc_mbon_max, requires_grad=False)
        # Generate a static list of odors for the network to train on
        self.train_odors = torch.multinomial(torch.ones(n_odors, self.n_kc),
                                             self.n_ones)
        # Set the number of task intervals
        self.n_int = 1

    def wt_update(self, W_kc_mbon, wt, dt, r_bar_kc, r_bar_dan, r_kc, r_dan,
                  n_batch, **kwargs):
        """ Returns directly the static KC->MBON plasticity variables

        Since this class has no plasticity, the KC->MBON weights and plasticity
        variables are not updated. Therefore, the values are returned directly.

        Parameters
            W_kc_MBON: list = KC->MBON weight matrices
            wt = dynamic plasticity update
            dt = time step of simulation
            r_bar_kc = eligibility trace of Kenyon cell activity
            r_bar_dan = eligibility trace of dopaminergic cell activity
            r_kc = current activity of Kenyon cells
            r_dan = current activity of dopamine cells
            n_batch = number of trials in mini-batch
        """

        return W_kc_mbon, wt, r_bar_kc, r_bar_dan

    def gen_stim_times(self, T_stim, T_int, dt, n_epoch, n_batch):
        """ Generates an array of stimulus presentation times for all trials

        Parameters
            T_stim = length of time each stimulus is presented
            T_int = length of task intervals
            dt = time step of simulations
            n_epoch = number of epochs to train over
            n_batch = number of trials in mini-batch

        Returns
            Array of stimulus presentation times
        """

        # Present the stimuli between 5-15s and 20-30s of the interval
        stim_min1 = 5
        stim_max1 = 15 - T_stim
        stim_range1 = int((stim_max1 - stim_min1) / dt)
        stim_offset1 = int(stim_min1 / dt)
        stim_min2 = 20
        stim_max2 = 30 - T_stim
        stim_range2 = int((stim_max2 - stim_min2) / dt)
        stim_offset2 = int(stim_min2 / dt)

        # Initialize stimulus presentation times array
        stim_times = torch.zeros(n_epoch, n_batch, 2)

        for i in range(n_batch):
            # Randomly determine the time of each stimulus presentation
            stim_times[:, i, 0] = torch.multinomial(torch.ones(stim_range1),
                                                    n_epoch, replacement=True)
            stim_times[:, i, 0] += stim_offset1
            stim_times[:, i, 1] = torch.multinomial(torch.ones(stim_range2),
                                                    n_epoch, replacement=True)
            stim_times[:, i, 1] += stim_offset2

        return stim_times

    def gen_inputs(self, stim_times, stim_len, time_len, n_batch, **kwargs):
        """ Generates inputs for first-order conditioning tasks.

        All trials are CS+ or control trials where CS+ is switched out for a
        neutral CS in the second presentation (CS- trials). In the case where the
        CS is switched, the target valence is zero. To account for the sequential
        nature of numerical simulations, the target valence is set to begin one
        time step after stimulus onset.

        The mix of conditions is listed as follows:
            probability of trials where CS+ is switched = 0.5

        Parameters
            stim_times = indices of stimulus presentations for each interval
            stim_len = length of stimulus presentation (in indices)
            time_len = size of time vector
            n_batch = number of trials in mini-batch
            p_omit = probability of omitting either CS or US from trials

        Returns
            r_kct_ls = odor (KC) input time series arrays for each interval
            r_extt_ls = context (ext) input time series arrays for each interval
            vt_opt = target valence for plotting and loss calculations
            ls_stims = stimulus time series for plotting
        """

        # Set stimulus presentation time dtype
        stim_times = stim_times.int()

        # # Define a set of 10 odors (CS)
        # # n_ones = int(self.n_kc * 0.1)
        # if self.train_odors is None:
        #     # odor_list = torch.zeros(self.n_train_odors, self.n_kc)
        #     # odor_inds = torch.multinomial(torch.ones(self.n_train_odors,
        #     #                                          self.n_kc), self.n_ones)
        #     # for n in range(self.n_train_odors):
        #     #     # Define an odor (CS)
        #     #     odor_list[n, odor_inds[n, :]] = 1
        #     # self.train_odors = odor_list
        #
        #     # odor_list = torch.zeros(self.n_train_odors, self.n_kc)
        #     odor_inds = torch.multinomial(torch.ones(self.n_train_odors,
        #                                              self.n_kc), self.n_ones)
        #     self.train_odors = odor_inds

        # # Conditioned stimuli (CS) = odors
        # odor_select = torch.multinomial(torch.ones(self.n_train_odors),
        #                                 n_batch, replacement=True)
        # r_kc = torch.zeros(n_batch, self.N_kc)
        # for b in range(n_batch):
        #     # Define an odor (CS) for each trial
        #     r_kc[b, :] = self.train_odors[odor_select[b], :]
        # # Unconditioned stimuli (US) = context
        # r_ext = torch.multinomial(torch.ones(n_batch, self.N_ext), self.N_ext)
        # Generate odors and context signals for each trial
        r_kc, r_ext = self.gen_r_kc_ext(n_batch)

        # Determine whether CS2+ is switched (switch on half of trials)
        switch_inds = torch.rand(n_batch) < 0.5

        # Initialize activity matrices
        r_kct = torch.zeros(n_batch, self.n_kc, time_len)
        r_extt = torch.zeros(n_batch, self.n_ext, time_len)
        time_CS_both = torch.zeros(n_batch, time_len)
        time_US = torch.zeros_like(time_CS_both)
        vt_opt = torch.zeros_like(time_CS_both)

        # For each stimulus presentation
        for i in range(2):
            # Initialize time matrices
            time_CS = torch.zeros(n_batch, time_len)

            for b in range(n_batch):
                stim_inds = stim_times[b, i] + torch.arange(stim_len)
                # Set the CS time
                time_CS[b, stim_inds] = 1
                # Set the US time
                if i == 0:
                    time_US[b, stim_inds + stim_len] = 1
                # Set the CS+/CS2 and target valence times
                if i == 1:
                    # Switch the odor in half the trials (target valence is zero)
                    if switch_inds[b]:
                        CS2_inds = torch.multinomial(torch.ones(self.n_kc),
                                                     self.n_ones)
                        r_kc[b, CS2_inds] = 1
                    # If the odor is not switched, set the target valence
                    else:
                        if r_ext[b, 0] == 1:
                            vt_opt[b, (stim_inds + 1)] = 1
                        else:
                            vt_opt[b, (stim_inds + 1)] = -1

            # Calculate the stimulus time series (KC = CS, ext = US)
            r_kct += torch.einsum('bm, mbt -> bmt', r_kc,
                                  time_CS.repeat(self.n_kc, 1, 1))
            r_extt += torch.einsum('bm, mbt -> bmt', r_ext,
                                   time_US.repeat(self.n_ext, 1, 1))
            time_CS_both += time_CS

        # Make a list of stimulus times to plot
        ls_stims = [time_CS_both, time_US]

        return [r_kct], [r_extt], vt_opt, ls_stims

    def gen_r_kc_ext(self, n_batch):
        """ Generates neuron activities for context and odor inputs.

        Parameters
            n_batch = number of trials in mini-batch

        Returns
            r_kc = odor (KC) inputs
            r_ext = context (ext) inputs
        """

        # Conditioned stimuli (CS) = odors
        odor_select = torch.multinomial(torch.ones(self.n_train_odors),
                                        n_batch, replacement=True)
        r_kc = torch.zeros(n_batch, self.n_kc)
        for b in range(n_batch):
            # Define an odor (CS) for each trial
            r_kc[b, :] = self.train_odors[odor_select[b], :]
        # Unconditioned stimuli (US) = context
        r_ext = torch.multinomial(torch.ones(n_batch, self.n_ext), self.n_ext)

        return r_kc, r_ext
