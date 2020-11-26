# Import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN


class ContinualRNN(FirstOrderCondRNN):
    def __init__(self, *, n_trial_odors=4):
        super().__init__()
        # Set the number of task intervals
        self.n_int = 1
        # Set the number of odors to train over on each trial
        self.n_trial_odors = n_trial_odors
        # Add a non-specific potentiation parameter
        self.beta = nn.Parameter(torch.ones(self.n_kc) * 0.01,
                                 requires_grad=True)

    def calc_dw(self, r_bar_kc, r_bar_dan, r_kc, r_dan, n_batch, nps=True,
                **kwargs):
        """ Calculates the dynamic weight update (see Eq 4).

        Parameters
            r_bar_kc = eligibility trace of Kenyon cell activity
            r_bar_dan = eligibility trace of dopaminergic cell activity
            r_kc = current activity of Kenyon cells
            r_dan = current activity of dopamine cells
            n_batch = number of trials in mini-batch
            nps = indicates whether non-specific potentiation is included

        Returns
            dw = increment of dynamic plasticity variable wt
        """

        # Calculate the LTD/LTP terms
        prod1 = torch.einsum('bd, bk -> bdk', r_bar_dan, r_kc)
        prod2 = torch.einsum('bd, bk -> bdk', r_dan, r_bar_kc)

        # Include non-specific potentiation (unless control condition)
        if nps:
            # Rectify the potentiation parameter
            beta = F.relu(self.beta.clone())
            # Constrain the potentiation parameter to be positive
            prod3 = torch.einsum('bd, bk -> bdk', r_bar_dan,
                                 beta.repeat(n_batch, 1))
        else:
            prod3 = torch.zeros_like(prod2)

        return prod1 - prod2 + prod3

    def init_w_kc_mbon(self, W_in, n_batch, e_tup):
        """ Initializes the KC->MBON weights for the task.

        KC->MBON weights are reset at the beginning of each epoch.

        Parameters
            W0 = specified initial weight values or None
            n_batch = number of trials in mini-batch
            e_tup: tuple = (current epoch, total training epochs)

        Returns
            W_kc_MBON: list = initial KC->MBON weight matrix
            wt = initial dynamic plasticity update
        """

        wt0 = torch.ones(n_batch, self.n_mbon, self.n_kc) * self.kc_mbon_max
        if W_in is None:
            W_in = (wt0.clone(), wt0.clone())
        # Calculate the saturation parameter and modify initial weights
        x_sat = min(1, (e_tup[0] / (e_tup[1] / 2)))
        Wt = W_in[0]
        wt = (1 - x_sat) * wt0 + x_sat * W_in[1]
        W_in = (Wt, wt)

        return W_in

    def gen_stim_times(self, T_stim, T_int, dt, n_epoch, n_batch):
        """ Generates an array of stimulus presentation times for all trials.

        Parameters
            T_stim = length of time each stimulus is presented
            T_int = length of task intervals
            dt = time step of simulations
            n_epoch = number of epochs to train over
            n_batch = number of trials in mini-batch

        Returns
            Array of stimulus presentation times
        """

        # Poisson rate of stimulus presentations
        stim_rate = 2 / T_int

        # Initialize stimulus presentation times array
        #         stim_times = torch.zeros(n_epoch, n_batch, n_odors)
        stim_times = [0] * n_epoch

        # Generate a list of stimulus presentation times for each trial
        for e in range(n_epoch):
            batch_times = [0] * n_batch
            for b in range(n_batch):
                odor_times = [0] * self.n_trial_odors
                for i in range(self.n_trail_odors):
                    trial_times = []
                    last_time = 0
                    while True:
                        stim_isi = -torch.log(torch.rand(1)) / stim_rate
                        next_time = last_time + stim_isi
                        if next_time < (T_int - 2 * T_stim):
                            # Stimulus times are indices (not times)
                            trial_times.append((next_time / dt).int())
                            last_time += stim_isi
                        # Ensure at least one presentation of each stimuli
                        elif last_time == 0:
                            continue
                        else:
                            break
                    odor_times[i] = torch.stack(trial_times)
                batch_times[b] = odor_times
            stim_times[e] = batch_times

        return stim_times

    def gen_inputs(self, stim_times, stim_len, time_len, n_batch, **kwargs):
        """ Generates inputs for first-order conditioning tasks.

        All trials are either CS+, CS- (US omitted) or CS omitted (control trials
        to avoid over-fitting). Of the trials where CS or US is omitted, a second
        parameter determines the relative fractions of CS or US trials omitted
        (p_omit_CS). See Fig. 2 of Jiang 2020 to determine sequencing of stimuli
        during training. To account for the sequential nature of numerical
        simulations, the target valence begins one time step after stimulus
        onset. Details provided in Jiang 2020 -> Methods -> Conditioning Tasks.

        The mix of conditions is listed as follows:
            probability of CS+ trials = 1 - p_omit
            probability of CS- trials = p_omit * 0.3
            probability of control trials = p_omit * 0.7

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

        # # Number of active neurons in an odor
        # n_ones = int(self.N_kc * 0.1)

        # Initialize activity matrices
        r_kct = torch.zeros(n_batch, self.n_kc, time_len)
        r_extt = torch.zeros(n_batch, self.n_ext, time_len)

        # Initialize lists and arrays to store stimulus time series
        ls_CS = []
        time_US_all = torch.zeros(n_batch, time_len)
        vt_opt = torch.zeros(n_batch, time_len)

        # For each batch, randomly generate different odors and presentation times
        for i in range(self.n_trial_odors):
            # Initialize the CS time matrix
            time_CS = torch.zeros(n_batch, time_len)
            time_US = torch.zeros_like(time_CS)

            # # Conditioned stimuli (CS) = odors
            # r_kc = torch.zeros(n_batch, self.N_kc)
            # # Unconditioned stimuli (US) = context
            # r_ext = torch.multinomial(torch.ones(n_batch, self.N_ext),
            #                           self.N_ext)
            # Generate odors and context signals for each trial
            r_kc, r_ext = self.gen_r_kc_ext(n_batch)

            # For each trial
            for b in range(n_batch):
                # # Define an odor (CS)
                # r_kc_inds = torch.multinomial(torch.ones(self.N_kc), self.n_ones)
                # r_kc[b, r_kc_inds] = 1

                for j, st in enumerate(stim_times[b][i]):
                    # Set the CS input times
                    stim_inds = st + torch.arange(stim_len)
                    time_CS[b, stim_inds] = 1

                    # For CS+ odors, set US and the valence
                    if i < (self.n_trial_odors / 2):
                        # Set the US input times
                        time_US[b, (stim_inds + stim_len)] = 1
                        # Set a target valence on every presentation but the first
                        if j > 0:
                            if r_ext[b, 0] == 1:
                                vt_opt[b, (stim_inds + 1)] = 1
                            else:
                                vt_opt[b, (stim_inds + 1)] = -1

            # Calculate the stimulus time series (KC = CS, ext = US)
            r_kct += torch.einsum('bm, mbt -> bmt', r_kc,
                                  time_CS.repeat(self.n_kc, 1, 1))
            r_extt += torch.einsum('bm, mbt -> bmt', r_ext,
                                   time_US.repeat(self.n_ext, 1, 1))
            ls_CS += time_CS
            time_US_all += time_US

        # Make a list of stimulus times to plot
        ls_stims = ls_CS + [time_US_all]

        return [r_kct], [r_extt], vt_opt, ls_stims
