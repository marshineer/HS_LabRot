# Import the required packages
# import torch
import torch.nn as nn
# import torch.nn.functional as F
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from network_classes.paper_tasks.common import *


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

    def gen_inputs(self, T_vars, n_batch, **kwargs):
        """ Generates inputs for continual learning task.

        Trials consist of the presentation of four different odors, with
        presentation times drawn from a Poisson distribution with a mean of two.
        The first two odors are conditioned stimuli, with the first
        corresponding to a positive valence (approach behaviour) and the second
        corresponding to a negative valence (avoidance behaviour). The last two
        are neutral odors and have zero associated valence. The conditioned
        stimuli are trained to respond (i.e. have a non-zero target valence) to
        each presentation of the odor AFTER the first.

        Parameters
            T_vars: Tuple
                T_vars[0] = T_int = length of trial (in seconds)
                T_vars[1] = T_stim = length of time each stimulus is presented
                T_vars[2] = dt = time step of simulations
                T_vars[3] = time_len = size of time vector
            n_batch = number of trials in mini-batch

        Returns
            r_kct_ls = odor (KC) input time series arrays for each interval
            r_extt_ls = context (ext) input time series arrays for each interval
            vt_opt = target valence for plotting and loss calculations
            ls_stims = stimulus time series for plotting
        """

        # Set the time variables
        T_int, T_stim, dt, time_len = T_vars
        # Average number of stimulus presentations
        st_mean = 2
        # Calculate the stimulus presentation times and length
        st_times, st_len = gen_cont_times(n_batch, dt, T_stim, T_int, st_mean,
                                          self.n_trial_odors)

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

            # Generate odors and context signals for each trial
            if i == 0:
                r_kc, r_ext = self.gen_r_kc_ext(n_batch, pos_val=True)
            elif i == 1:
                r_kc, r_ext = self.gen_r_kc_ext(n_batch, pos_val=False)
            else:
                r_kc, r_ext = self.gen_r_kc_ext(n_batch)

            # For each trial
            for b in range(n_batch):
                for j, st in enumerate(st_times[i][b]):
                    # Set the CS input times
                    stim_inds = st + torch.arange(st_len)
                    time_CS[b, stim_inds] = 1

                    # For CS+ odors, set US and the valence
                    if i < (self.n_trial_odors / 2):
                        # Set the US input times
                        time_US[b, (stim_inds + st_len)] = 1
                        # Set target valence on every presentation but the first
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
