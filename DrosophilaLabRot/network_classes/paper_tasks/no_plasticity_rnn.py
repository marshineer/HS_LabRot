# Import the required packages
from torch.autograd import Variable
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from common.common import *


class NoPlasticityRNN(FirstOrderCondRNN):
    def __init__(self, *, n_odors=10, **kwargs):
        super().__init__(**kwargs)
        # Set the static KC->MBON weights
        W_kc_mbon_max = 0.05
        self.W_kc_mbon = Variable(torch.rand(self.n_mbon, self.n_kc) *
                                  W_kc_mbon_max, requires_grad=False)
        # Generate a static list of odors for the network to train on
        odor_inds = torch.multinomial(torch.ones(n_odors, self.n_kc),
                                      self.n_ones)
        self.train_odors = Variable(odor_inds)
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

        return r_bar_kc, r_bar_dan

    def gen_inputs(self, T_vars, n_batch, **kwargs):
        """ Generates inputs for task without KC->MBON plasticity.

        All trials are CS+ or control trials where CS+ is switched out for a
        neutral CS in the second presentation (CS- trials). In the case where the
        CS is switched, the target valence is zero. To account for the sequential
        nature of numerical simulations, the target valence is set to begin one
        time step after stimulus onset.

        The mix of conditions is listed as follows:
            probability of trials where CS+ is switched = 0.5

        Parameters
            T_vars: Tuple
                T_vars[0] = T_int = length of trial (in seconds)
                T_vars[1] = T_stim = length of time each stimulus is presented
                T_vars[2] = dt = time step of simulations
                T_vars[3] = time_len = size of time vector
            n_batch = number of trials in mini-batch
            p_omit = probability of omitting either CS or US from trials

        Returns
            r_kct_ls = odor (KC) input time series arrays for each interval
            r_extt_ls = context (ext) input time series arrays for each interval
            vt_opt = target valence for plotting and loss calculations
            ls_stims = stimulus time series for plotting
        """

        # Set the range over which stimuli can be presented
        T_range = [(5, 15), (20, 30)]
        # Set the time variables
        T_stim, dt, time_len = T_vars[1:]

        # Generate odors and context signals for each trial
        r_kc, r_ext = self.gen_r_kc_ext(n_batch, **kwargs)

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

            # Calculate the stimulus presentation times and length
            st_times, st_len = gen_int_times(n_batch, dt, T_stim, T_range[i])

            for b in range(n_batch):
                stim_inds = st_times[b] + torch.arange(st_len)
                # Set the CS time
                time_CS[b, stim_inds] = 1
                # Set the US time
                if i == 0:
                    time_US[b, stim_inds + st_len] = 1
                # Set the CS+/CS2 and target valence times
                if i == 1:
                    # Switch the odor in half the trials (target valence is zero)
                    if switch_inds[b]:
                        # CS2_inds = torch.multinomial(torch.ones(self.n_kc),
                        #                              self.n_ones)
                        # r_kc[b, CS2_inds] = 1
                        CS2_inds = torch.multinomial(torch.ones(self.n_kc),
                                                     self.n_kc)
                        r_kc[b, :] = r_kc[b, CS2_inds]
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
