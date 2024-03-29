# Import the required packages
from network_classes.base_rnn import FirstOrderCondRNN
from common.common import *


class ExtendedCondRNN(FirstOrderCondRNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set the number of task intervals
        self.n_int = 3

    def gen_inputs(self, T_vars, n_batch, p_omit=0.5, p_extinct=0.5, **kwargs):
        """ Generates inputs for extinction and second-order tasks.

        Trials are either extinction or second-order conditioning. No strictly
        first-order conditioning trials are included. In half of trials, CS or US
        are omitted (pg 28 of Jiang 2020) to prevent over-fitting. Of the trials
        where CS or US is omitted, a second parameter determines the relative
        fractions of CS or US trials omitted (p_omit_CS).

        There are no explicit first-order conditioning tasks included, since
        first-order conditioning is a necessary part of both extinction and
        second-order conditioning. See Figure 2 of Jiang 2020 to determine
        sequencing of stimuli during training. To account for the sequential
        nature of numerical simulations, the target valence is set to begin one
        time step after stimulus onset.

        The mix of conditions is listed as follows:
            probability of extinction trials = p_extinct = 0.5
            probability of second-order conditioning trials = 1 - p_extinct = 0.5
            probability of control (US omitted = CS-) trials = p_omit * 0.3
            probability of control (CS omitted) trials = p_omit * 0.7
        Note: extinction and second-order trials overlap arbitrarily with
              control trials

        Parameters
            T_vars: Tuple
                T_vars[0] = T_int = length of trial (in seconds)
                T_vars[1] = T_stim = length of time each stimulus is presented
                T_vars[2] = dt = time step of simulations
                T_vars[3] = time_len = size of time vector
            n_batch = number of trials in mini-batch
            p_omit = probability of omitting either CS or US from trials
            p_extinct = probability of trials being extinction trials

        Returns
            r_kct_ls = odor (KC) input time series arrays for each interval
            r_extt_ls = context (ext) input time series arrays for each interval
            vt_opt = target valence for plotting and loss calculations
            ls_stims = stimulus time series for plotting
        """

        # Set the range over which stimuli can be presented
        T_range = (5, 15)
        # Set the time variables
        T_stim, dt, time_len = T_vars[1:]

        # Generate odors and context signals for each trial
        r_kc1, r_ext = self.gen_r_kc_ext(n_batch)
        r_kc2, _ = self.gen_r_kc_ext(n_batch)

        # Determine whether trials are extinction or second-order
        p_extinct = p_extinct
        extinct_inds = torch.rand(n_batch) < p_extinct

        # Determine whether CS or US are randomly omitted
        omit_inds = torch.rand(n_batch) < p_omit
        # If omitted, determine which one is omitted
        p_omit_CS = 0.7
        x_omit_CS = torch.rand(n_batch)
        omit_CS_inds = torch.logical_and(omit_inds, x_omit_CS < p_omit_CS)
        omit_US_inds = torch.logical_and(omit_inds, x_omit_CS > p_omit_CS)

        # Initialize lists to store inputs and target valence
        r_kct_ls = []
        r_extt_ls = []
        vals = []
        # ls_CS1 = []
        # ls_CS2 = []
        # ls_US = []

        # For each interval
        for i in range(self.n_int):
            # Define a binary CS and US time series to mulitply the inputs by
            time_CS1 = torch.zeros(n_batch, time_len)
            time_CS2 = torch.zeros_like(time_CS1)
            time_US = torch.zeros_like(time_CS1)
            # Define the target valences
            val_int = torch.zeros_like(time_CS1)

            # Calculate the stimulus presentation times and length
            st_times, st_len = gen_int_times(n_batch, dt, T_stim, T_range)

            # Set the inputs for each trial
            for b in range(n_batch):
                stim_inds = st_times[b] + torch.arange(st_len)
                # Set the inputs for extinction trials
                if extinct_inds[b]:
                    # Set the CS input times
                    if not omit_CS_inds[b]:
                        time_CS1[b, stim_inds] = 1
                    # Set the US input times
                    if i == 0 and not omit_US_inds[b]:
                        time_US[b, stim_inds + st_len] = 1
                    # Set the target valence times
                    if i > 0 and not omit_inds[b]:
                        if r_ext[b, 0] == 1:
                            val_int[b, (stim_inds + 1)] = 1 / i
                        elif r_ext[b, 1] == 1:
                            val_int[b, (stim_inds + 1)] = -1 / i
                # Set the inputs for second-order conditioning trials
                else:
                    # Set the CS1 and CS2 input times
                    if not omit_CS_inds[b]:
                        if i == 0:
                            time_CS1[b, stim_inds] = 1
                        if i == 1:
                            time_CS1[b, stim_inds + st_len] = 1
                            time_CS2[b, stim_inds] = 1
                        if i == 2:
                            time_CS2[b, stim_inds] = 1
                    # Set the US input times
                    if i == 0 and not omit_US_inds[b]:
                        time_US[b, stim_inds + st_len] = 1
                    # Set the target valence times
                    if i > 0 and not omit_inds[b]:
                        if r_ext[b, 0] == 1:
                            val_int[b, (stim_inds + (i % 2) * st_len + 1)] = 1
                        elif r_ext[b, 1] == 1:
                            val_int[b, (stim_inds + (i % 2) * st_len + 1)] = -1

            # Calculate the stimulus time series (KC = CS, ext = US)
            r_kct = torch.einsum('bm, mbt -> bmt', r_kc1,
                                 time_CS1.repeat(self.n_kc, 1, 1))
            r_kct += torch.einsum('bm, mbt -> bmt', r_kc2,
                                  time_CS2.repeat(self.n_kc, 1, 1))
            r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                                  time_US.repeat(self.n_ext, 1, 1))

            r_kct_ls.append(r_kct)
            r_extt_ls.append(r_extt)
            vals.append(val_int)
            # ls_CS1 += time_CS1
            # ls_CS2 += time_CS2
            # ls_US += time_US

        # Concatenate target valences
        vt_opt = torch.cat((vals[0], vals[1], vals[2]), dim=-1)

        # # Make a list of stimulus times to plot
        # ls_stims = [torch.cat(ls_CS1), torch.cat(ls_US), torch.cat(ls_CS2)]

        # return r_kct_ls, r_extt_ls, vt_opt, ls_stims
        return r_kct_ls, r_extt_ls, vt_opt
