# Import the required packages
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from network_classes.paper_tasks.common import cond_loss


class FirstOrderCondRNN(nn.Module):
    def __init__(self, *, n_kc=200, n_mbon=20, n_fbn=60, n_ext=2, n_out=1,
                 f_ones=0.1, n_seed=None):
        super().__init__()
        # Set the seeds
        if n_seed is not None:
            np.random.seed(n_seed)
            torch.manual_seed(n_seed)

        # Set constants
        W_kc_mbon_max = 0.05
        self.kc_mbon_min = 0.  # Minimum synaptic weight
        self.kc_mbon_max = W_kc_mbon_max  # Maximum synaptic weight
        self.W_kc_mbon_0 = Variable(torch.ones((n_mbon, n_kc)) * W_kc_mbon_max,
                                    requires_grad=False)
        self.tau_w = 5  # Time scale of KC->MBON LTD/LTP (plasticity)
        self.tau_r = 1  # Time scale of output circuitry activity
        self.n_int = 2  # Number of task intervals

        # Set the sizes of layers
        n_dan = n_mbon
        self.n_kc = n_kc
        self.n_mbon = n_mbon
        self.n_fbn = n_fbn
        self.n_dan = n_dan
        self.n_recur = n_mbon + n_fbn + n_dan
        self.n_ext = n_ext
        self.n_out = n_out
        self.n_ones = int(n_kc * f_ones)

        # Define network variables used to store data
        # Odors
        self.train_odors = None
        self.eval_odors = None
        # Training parameters (for continuation)
        self.train_rts = None
        self.train_Wts = None
        self.train_wts = None
        self.train_vts = None
        self.train_vt_opts = None
        self.train_CS_stim = None
        self.train_US_stim = None
        self.train_loss = None
        # Evaluation parameters (for plotting and analysis)
        self.eval_rts = None
        self.eval_Wts = None
        self.eval_wts = None
        self.eval_vts = None
        self.eval_vt_opts = None
        self.eval_CS_stim = None
        self.eval_US_stim = None
        self.eval_loss = None

        # Define updatable network parameters
        sqrt2 = torch.sqrt(torch.tensor(2, dtype=torch.float))
        mean_mbon = torch.zeros((self.n_recur, n_mbon))
        mean_fbn = torch.zeros((self.n_recur, n_fbn))
        mean_dan = torch.zeros((self.n_recur, n_dan))
        W_mbon = torch.normal(mean_mbon, torch.sqrt(1 / (sqrt2 * n_mbon)),
                              generator=n_seed)
        W_fbn = torch.normal(mean_fbn, torch.sqrt(1 / (sqrt2 * n_fbn)),
                             generator=n_seed)
        W_dan = torch.normal(mean_dan, torch.sqrt(1 / (sqrt2 * n_dan)),
                             generator=n_seed)
        self.W_recur = nn.Parameter(torch.cat((W_mbon, W_fbn, W_dan), dim=1),
                                    requires_grad=True)
        self.W_ext = nn.Parameter(torch.randn(n_fbn, n_ext),
                                  requires_grad=True)
        mean_readout = torch.zeros((n_out, n_mbon))
        std_readout = 1 / torch.sqrt(torch.tensor(n_mbon, dtype=torch.float))
        self.W_readout = nn.Parameter(torch.normal(mean_readout, std_readout,
                                                   generator=n_seed),
                                      requires_grad=True)
        self.bias = nn.Parameter(torch.ones(self.n_recur) * 0.1,
                                 requires_grad=True)

    def forward(self, r_kc, r_ext, time, n_batch=30, W0=None, r0=None, **kwargs):
        """ Defines the forward pass of the RNN

        The KC->MBON weights are constrained to the range [0, 0.05].
        MBONs receive external input from Kenyon cells (r_kc i.e. 'odors').
        Feedback neurons (FBNs) receive external input (r_ext i.e. 'context').
        DAN->MBON weights are permanently set to zero.
        DANs receive no external input.

        Inputs
            r_kc = activity of the Kenyon cell inputs (representing odors)
            r_ext = context inputs (representing the conditioning context)
            time = time vector for a single interval
            n_batch = number of trials in mini-batch
            W0 = initial weights for KC->MBON connections
            r0 = initial activities for output circuitry neurons

        Returns
            r_recur: list of torch.ndarray(batch_size, n_mbon + n_fbn + n_dan)
                = time series of activities in the output circuitry
            Wt: list of torch.ndarray(batch_size, n_recur, n_recur)
                = time series of KC->MBON weights (dopaminergic plasticity)
            readout: list of torch.ndarray(batch_size, 1)
                = time series of valence readouts (behaviour)
        """

        # Define the time step of the simulation
        dt = np.diff(time)[0]

        # Initialize output circuit firing rates for each trial
        if r0 is not None:
            r_init = r0
        else:
            r_init = torch.ones(n_batch, self.n_recur) * 0.1
            r_init[:, :self.n_mbon] = 0
        r_recur = [r_init]

        # Initialize the eligibility traces and readout
        r_bar_kc = r_kc[:, :, 0]
        r_bar_dan = r_recur[-1][:, -self.n_dan:]
        readout = [torch.einsum('bom, bm -> bo',
                                self.W_readout.repeat(n_batch, 1, 1),
                                r_recur[-1][:, :self.n_mbon]).squeeze()]

        # Set the weights DAN->MBON to zero
        W_recur = self.W_recur.clone()
        W_recur[:self.n_mbon, -self.n_dan:] = 0

        # Set the KC->MBON weights
        W_kc_mbon, wt = W0

        # Update activity for each time step
        for t in range(time.shape[0] - 1):
            # Define the input to the output circuitry
            I_kc_mbon = torch.einsum('bmk, bk -> bm',
                                     W_kc_mbon, r_kc[:, :, t])
            I_fbn = torch.einsum('bfe, be -> bf',
                                 self.W_ext.repeat(n_batch, 1, 1),
                                 r_ext[:, :, t])
            I_tot = torch.zeros((n_batch, self.n_recur))
            I_tot[:, :self.n_mbon] = I_kc_mbon
            I_tot[:, self.n_mbon:self.n_mbon + self.n_fbn] = I_fbn

            # Update the output circuitry activity (see Eq. 1)
            Wr_prod = torch.einsum('bsr, br -> bs',
                                   W_recur.repeat(n_batch, 1, 1),
                                   r_recur[-1])
            dr = (-r_recur[-1] + F.relu(Wr_prod + self.bias.repeat(n_batch, 1)
                                        + I_tot)) / self.tau_r
            r_recur.append(r_recur[-1] + dr * dt)

            # Update KC->MBON plasticity variables
            out = self.wt_update(W_kc_mbon, wt, dt, r_bar_kc, r_bar_dan,
                                 r_kc[:, :, t], r_recur[-1][:, -self.n_dan:],
                                 n_batch, **kwargs)
            W_kc_mbon, wt, r_bar_kc, r_bar_dan = out

            # Calculate the readout (see Eq. 2)
            readout.append(torch.einsum('bom, bm -> bo',
                                        self.W_readout.repeat(n_batch, 1, 1),
                                        r_recur[-1][:, :self.n_mbon]).squeeze())

        return r_recur, (W_kc_mbon.detach(), wt.detach()), readout

    def wt_update(self, W_kc_mbon, wt, dt, r_bar_kc, r_bar_dan, r_kc, r_dan,
                  n_batch, **kwargs):
        """ Updates the KC->MBON plasticity variables

        Synaptic weights from the Kenyon cells to the mushroom body output neurons
        (MBONs) are updated dynamically. All other weights are network parameters.
        The synaptic connections between Kenyon Cells (KCs) and MBONs are updated
        using a LTP/LTD rule (see Figure 1B of Jiang 2020), which models dopamine-
        gated neural plasticity on short time scale (behavioural learning).

        Parameters
            W_kc_mbon: list = KC->MBON weight matrices
            wt = dynamic plasticity update
            dt = time step of simulation
            r_bar_kc = eligibility trace of Kenyon cell activity
            r_bar_dan = eligibility trace of dopaminergic cell activity
            r_kc = current activity of Kenyon cells
            r_dan = current activity of dopamine cells
            n_batch = number of trials in mini-batch
        """

        # Calculate the eligibility traces (represent LTP/LTD)
        r_bar_kc = r_bar_kc + (r_kc - r_bar_kc) * dt / self.tau_w
        r_bar_dan = r_bar_dan + (r_dan - r_bar_dan) * dt / self.tau_w
        # Update the dynamic weight variable
        dw = self.calc_dw(r_bar_kc, r_bar_dan, r_kc, r_dan, n_batch, **kwargs)
        wt += dw * dt
        # Update the KC->MBON weights (see Eq. 8)
        dW = (-W_kc_mbon + wt) / self.tau_w
        W_tp1 = W_kc_mbon + dW * dt
        # Clip the KC->MBON weights to the range [0, 0.05]
        W_kc_mbon = torch.clamp(W_tp1, self.kc_mbon_min, self.kc_mbon_max)

        return W_kc_mbon, wt, r_bar_kc, r_bar_dan

    def calc_dw(self, r_bar_kc, r_bar_dan, r_kc, r_dan, n_batch, **kwargs):
        """ Calculates the dynamic weight update (see Eq 4).

        Parameters
            r_bar_kc = eligibility trace of Kenyon cell activity
            r_bar_dan = eligibility trace of dopaminergic cell activity
            r_kc = current activity of Kenyon cells
            r_dan = current activity of dopamine cells
            n_batch = number of trials in mini-batch

        Returns
            update to dynamic plasticity variables wt
        """

        # Calculate the LTD/LTP terms
        prod1 = torch.einsum('bd, bk -> bdk', r_bar_dan, r_kc)
        prod2 = torch.einsum('bd, bk -> bdk', r_dan, r_bar_kc)

        return prod1 - prod2

    def train_net(self, *, opti, T_int=30, T_stim=2, dt=0.5, n_epoch=5000,
                  n_batch=30, clip=0.001, **kwargs):
        """ Trains a network on classical conditioning tasks.

        Tasks include first-order or second-order conditioning, and extinction.
        Tasks consist of two (first-order) or three (second-order and extinction)
        intervals. Each task has its own input generating function. Stimuli are
        presented between 5-15s of each interval. Neuron activities are reset
        between intervals to prevent associations being represented through
        persistent activity.

        Parameters
            opti = RNN network optimizer
            T_int = length of task intervals
            T_stim = length of time each stimulus is presented
            dt = time step of simulations
            n_epoch = number of epochs to train over
            n_batch = number of trials in mini-batch
            clip = maximum gradient allowed during training

        Returns
            r_out_epoch = output circuit neuron activities for final epoch
            Wt_epoch = KC->MBON weights for final epoch
            vt_epoch = readout (i.e. valence) for final epoch
            vt_opt = target valence for final epoch
            loss_hist = list of losses for all epochs
            ls_stims = list of stimulus time series for plotting
        """

        # Interval time vector
        time_int = torch.arange(0, T_int + dt / 10, dt)

        # Generate a list of stimulus presentation times
        stim_times = self.gen_stim_times(T_stim, T_int, dt, n_epoch, n_batch)
        # Length of stimulus in indices
        stim_len = int(T_stim / dt)

        # List to store losses
        loss_hist = []

        # Initialize the KC-MBON weights
        W_kc_mbon = None

        for epoch in range(n_epoch):
            # Lists to store activities, weights, readouts and target valences
            r_outs = []
            # Wts = []
            vts = []

            # Set the intial KC->MBON weight values for each trial
            W_kc_mbon = self.init_w_kc_mbon(W_kc_mbon, n_batch, (epoch, n_epoch))

            # Generate odor (r_kc), context (r_ext), and target valence (vt_opt)
            st_epoch = stim_times[epoch]
            net_inputs = self.gen_inputs(st_epoch, stim_len, time_int.shape[0],
                                         n_batch, **kwargs)
            r_kc, r_ext, vt_opt, ls_stims = net_inputs

            # For each interval in the task
            for i in range(self.n_int):
                # Run the forward model
                net_outs = self(r_kc[i], r_ext[i], time_int, n_batch, W_kc_mbon)
                # Set the initial KC->MBON weights for the next interval
                W_kc_mbon = net_outs[1]

                # Append the interval outputs to lists
                r_outs += net_outs[0]
                # Wts += net_outs[1][0]
                vts += net_outs[2]

            # Concatenate the activities, weights and valences
            r_out_epoch = torch.stack(r_outs, dim=-1)
            # Wt_epoch = torch.stack(Wts, dim=-1)
            vt_epoch = torch.stack(vts, dim=-1)

            # Calculate the loss
            loss = cond_loss(vt_epoch, vt_opt, r_out_epoch[:, -self.n_dan:, :])

            # Update the network parameters
            opti.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            opti.step()

            # Print an update
            if epoch % 500 == 0:
                print(epoch, loss.item())
            loss_hist.append(loss.item())

        # return r_out_epoch, Wt_epoch, vt_epoch, vt_opt, loss_hist, ls_stims
        return r_out_epoch, vt_epoch, vt_opt, loss_hist, ls_stims

    def init_w_kc_mbon(self, W0, n_batch, e_tup):
        """ Initializes the KC->MBON weights for the task.

        KC->MBON weights are reset at the beginning of each epoch.

        Parameters
            W0 = specified initial weight values or None
            n_batch = number of trials in mini-batch
            e_tup: tuple = (current epoch, total training epochs)

        Returns
            tuple of initial KC->MBON and dynamic plasticity variables
        """

        wt0 = self.W_kc_mbon_0.repeat(n_batch, 1, 1)

        return wt0.clone(), wt0.clone()

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

        # Present the stimulus between 5-15s of each interval
        stim_min = 5
        stim_max = 15 - T_stim
        stim_range = int((stim_max - stim_min) / dt)
        stim_wts = torch.ones(n_epoch, stim_range)
        stim_offset = int(stim_min / dt)

        # Initialize stimulus presentation times array
        stim_times = torch.zeros(n_epoch, n_batch, self.n_int)

        for i in range(self.n_int):
            # Randomly determine the time of each stimulus presentation
            stim_times[:, :, i] = torch.multinomial(stim_wts, n_batch,
                                                    replacement=True)
            stim_times[:, :, i] += stim_offset

        return stim_times

    def gen_inputs(self, stim_times, stim_len, time_len, n_batch, p_omit=0.3):
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

        # Set stimulus presentation time dtype
        stim_times = stim_times.int()

        # # Conditioned stimuli (CS) = odors
        # r_kc = torch.zeros(n_batch, self.n_kc)
        # for b in range(n_batch):
        #     # Define an odor (CS) for each trial
        #     r_kc_inds = torch.multinomial(torch.ones(self.n_kc), self.n_ones)
        #     r_kc[b, r_kc_inds] = 1
        # # Unconditioned stimuli (US) = context
        # r_ext = torch.multinomial(torch.ones(n_batch, self.n_ext), self.n_ext)
        # Generate odors and context signals for each trial
        r_kc, r_ext = self.gen_r_kc_ext(n_batch)

        # Determine whether CS or US are randomly omitted
        omit_inds = torch.rand(n_batch) < p_omit
        # If omitted, determine which one is omitted
        p_omit_CS = 0.7
        x_omit_CS = torch.rand(n_batch)
        omit_CS_inds = torch.logical_and(omit_inds, x_omit_CS < p_omit_CS)
        omit_US_inds = torch.logical_and(omit_inds, x_omit_CS > p_omit_CS)

        # Initialize lists to store inputs, target valence and stimulus times
        r_kct_ls = []
        r_extt_ls = []
        vals = []
        ls_CS = []
        ls_US = []

        # For each interval
        for i in range(self.n_int):
            # Initialize time matrices
            time_CS = torch.zeros(n_batch, time_len)
            time_US = torch.zeros_like(time_CS)
            val_int = torch.zeros_like(time_CS)

            for b in range(n_batch):
                stim_inds = stim_times[b, i] + torch.arange(stim_len)
                # Set the CS input times
                if not omit_CS_inds[b]:
                    time_CS[b, stim_inds] = 1
                # Set the US input times
                if i == 0 and not omit_US_inds[b]:
                    time_US[b, (stim_inds + stim_len)] = 1
                # Set the target valence times
                if i == 1 and not omit_inds[b]:
                    if r_ext[b, 0] == 1:
                        val_int[b, (stim_inds + 1)] = 1
                    else:
                        val_int[b, (stim_inds + 1)] = -1

            # Calculate the stimulus time series (KC = CS, ext = US)
            r_kct = torch.einsum('bm, mbt -> bmt', r_kc,
                                 time_CS.repeat(self.n_kc, 1, 1))
            r_extt = torch.einsum('bm, mbt -> bmt', r_ext,
                                  time_US.repeat(self.n_ext, 1, 1))

            r_kct_ls.append(r_kct)
            r_extt_ls.append(r_extt)
            vals.append(val_int)
            ls_CS += time_CS
            ls_US += time_US

        # Concatenate target valences
        vt_opt = torch.cat((vals[0], vals[1]), dim=-1)

        # Make a list of stimulus times to plot
        ls_stims = [torch.cat(ls_CS), torch.cat(ls_US)]

        return r_kct_ls, r_extt_ls, vt_opt, ls_stims

    def gen_r_kc_ext(self, n_batch):
        """ Generates neuron activities for context and odor inputs.

        Parameters
            n_batch = number of trials in mini-batch

        Returns
            r_kc = odor (KC) inputs
            r_ext = context (ext) inputs
        """

        # Conditioned stimuli (CS) = odors
        r_kc = torch.zeros(n_batch, self.n_kc)
        # n_ones = int(self.n_kc * 0.1)
        for b in range(n_batch):
            # Define an odor (CS) for each trial
            r_kc_inds = torch.multinomial(torch.ones(self.n_kc), self.n_ones)
            r_kc[b, r_kc_inds] = 1
        # Unconditioned stimuli (US) = context
        r_ext = torch.multinomial(torch.ones(n_batch, self.n_ext), self.n_ext)

        return r_kc, r_ext
