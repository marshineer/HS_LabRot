# Import the required packages
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from common.common import *


class FirstOrderCondRNN(nn.Module):
    def __init__(self, *, n_kc=200, n_mbon=20, n_fbn=60, n_ext=2, n_out=1,
                 T_int=30, T_stim=2, dt=0.5, f_ones=0.1, n_seed=None):
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
        self.T_int = T_int  # Length of task interval [seconds]
        self.T_stim = T_stim  # Length of stimulus presentation [seconds]
        self.dt = dt  # Length of simulation time step [seconds]

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
        self.train_T_vars = None
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

        Parameters
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

        # Initialize the KC->MBON weights
        W_kc_mbon = [W0[0]]
        wt = [W0[1]]

        # Update activity for each time step
        for t in range(time.shape[0] - 1):
            # Define the input to the output circuitry
            I_kc_mbon = torch.einsum('bmk, bk -> bm',
                                     W_kc_mbon[-1], r_kc[:, :, t])
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
            wt_out = self.wt_update(W_kc_mbon, wt, dt, r_bar_kc, r_bar_dan,
                                    r_kc[:, :, t], r_recur[-1][:, -self.n_dan:],
                                    n_batch, **kwargs)
            # W_kc_mbon, wt, r_bar_kc, r_bar_dan = out
            r_bar_kc, r_bar_dan = wt_out

            # Calculate the readout (see Eq. 2)
            readout.append(torch.einsum('bom, bm -> bo',
                                        self.W_readout.repeat(n_batch, 1, 1),
                                        r_recur[-1][:, :self.n_mbon]).squeeze())

        return r_recur, (W_kc_mbon, wt), readout

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
        wt.append(wt[-1] + dw * dt)
        # Update the KC->MBON weights (see Eq. 8)
        dW = (-W_kc_mbon[-1] + wt[-1]) / self.tau_w
        W_tp1 = W_kc_mbon[-1] + dW * dt
        # Clip the KC->MBON weights to the range [0, 0.05]
        W_kc_mbon.append(torch.clamp(W_tp1, self.kc_mbon_min, self.kc_mbon_max))

        # return W_kc_mbon, wt, r_bar_kc, r_bar_dan
        return r_bar_kc, r_bar_dan

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

    def run_train(self, opti, *, T_int=None, T_stim=None, dt=None, n_epoch=5000,
                  n_batch=30, reset_wts=True, clip=0.001, **kwargs):
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
            reset_wts = indicates whether to reset weights between trials
            clip = maximum gradient allowed during training

        Returns
            r_out_epoch = output circuit neuron activities for final epoch
            Wt_epoch = KC->MBON weights for final epoch
            vt_epoch = readout (i.e. valence) for final epoch
            vt_opt = target valence for final epoch
            loss_hist = list of losses for all epochs
            ls_stims = list of stimulus time series for plotting
        """

        # Set the time variables
        if T_int is None:
            T_int = self.T_int
        if T_stim is None:
            T_stim = self.T_stim
        if dt is None:
            dt = self.dt
        time_int = torch.arange(0, T_int + dt / 10, dt)
        T_vars = (T_int, T_stim, dt, time_int.shape[0])
        self.train_T_vars = T_vars[:-1]

        # List to store losses
        loss_hist = []

        # Initialize the KC-MBON weights
        W_in = None

        for epoch in range(n_epoch):
            # Lists to store activities, weights, readouts and target valences
            rts = []
            vts = []

            # Set the intial KC->MBON weight values for each trial
            if reset_wts or (W_in is None):
                W_in = self.init_w_kc_mbon(None, n_batch, (epoch, n_epoch))
            else:
                W_in = (W_in[0][-1].detach(), W_in[1][-1].detach())

            # Generate odor (r_kc), context (r_ext), and target valence (vt_opt)
            net_inputs = self.gen_inputs(T_vars, n_batch, **kwargs)
            r_kc, r_ext, vt_opt, ls_stims = net_inputs

            # For each interval in the task
            for i in range(self.n_int):
                # Run the forward model
                net_out = self(r_kc[i], r_ext[i], time_int, n_batch, W_in)
                rt_int, (Wt_int, wt_int), vt_int = net_out
                # Pass the KC->MBON weights to the next interval
                W_in = (Wt_int[-1], wt_int[-1])

                # Append the interval outputs to lists
                rts += rt_int
                vts += vt_int

            # Convert the list of time point values to a tensor
            #  (time is the last dimension)
            rt_epoch = torch.stack(rts, dim=-1)
            vt_epoch = torch.stack(vts, dim=-1)

            # Calculate the loss
            loss = cond_loss(vt_epoch, vt_opt, rt_epoch[:, -self.n_dan:, :])

            # Update the network parameters
            opti.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            opti.step()

            # Print an update
            if epoch % 500 == 0:
                print(epoch, loss.item())
            loss_hist.append(loss.item())

        self.train_loss = loss_hist

        return loss_hist

    def init_w_kc_mbon(self, W_in, n_batch, e_tup):
        """ Initializes the KC->MBON weights for the task.

        KC->MBON weights are reset at the beginning of each epoch.

        Parameters
            W_in = specified initial weight values or None
            n_batch = number of trials in mini-batch
            e_tup: tuple = (current epoch, total training epochs)

        Returns
            tuple of initial KC->MBON and dynamic plasticity variables
        """

        if W_in is None:
            wt0 = self.W_kc_mbon_0.repeat(n_batch, 1, 1)
            W_in = (wt0.clone(), wt0.clone())

        return W_in

    def gen_inputs(self, T_vars, n_batch, p_omit=0.3):
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
        T_range = (5, 15)
        # Set the time variables
        T_stim, dt, time_len = T_vars[1:]

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

            # Calculate the stimulus presentation times and length
            st_times, st_len = gen_int_times(n_batch, dt, T_stim, T_range)

            for b in range(n_batch):
                stim_inds = st_times[b] + torch.arange(st_len)
                # Set the CS input times
                if not omit_CS_inds[b]:
                    time_CS[b, stim_inds] = 1
                # Set the US input times
                if i == 0 and not omit_US_inds[b]:
                    time_US[b, (stim_inds + st_len)] = 1
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

    def run_eval(self, trial_ls, *, T_int=None, T_stim=None, dt=None, n_trial=1,
                 n_batch=1, reset_wts=True, **kwargs):
        """ Runs an evaluation based on a series of input functions

        Parameters
            trial_ls = list of interval functions that compose a trial
            T_int = length of a task interval (in seconds)
            T_stim = length of time each stimulus is presented
            dt = time step of simulation (in seconds)
            n_trial = number of trials to run
            n_batch = number of parallel trials in a batch
            reset_wts = indicates whether to reset weights between trials
        """

        # Reset lists storing evaluation data
        self.eval_rts = []
        self.eval_Wts = []
        self.eval_wts = []
        self.eval_vts = []
        self.eval_vt_opts = []
        self.eval_CS_stim = []
        self.eval_US_stim = []
        self.eval_loss = []

        # Set the time variables
        if T_int is None:
            T_int = self.T_int
        if T_stim is None:
            T_stim = self.T_stim
        if dt is None:
            dt = self.dt
        time_int = torch.arange(0, T_int + dt / 10, dt)
        t_len = time_int.shape[0]
        n_int = len(trial_ls)

        # Initialize the KC-MBON weights and plasticity variable
        W_in = None

        # For each function in the list, run an interval
        # All intervals together compose a single trial
        for trial in range(n_trial):
            # Lists to store activities, weights, readouts and target valences
            rts = []
            Wts = []
            wts = []
            vts = []
            vt_opts = []
            time_CS = []
            time_US = []

            # Determine whether to reset KC->MBON weights between trials
            if reset_wts or (W_in is None):
                W_in = self.init_w_kc_mbon(None, n_batch, (trial, n_trial))
            else:
                W_in = (W_in[0][-1].detach(), W_in[1][-1].detach())

            # Generate odors and context (odor = KC = CS, context = ext = US)
            r_kc0, r_ext0 = self.gen_r_kc_ext(n_batch, **kwargs)
            trial_odors = [r_kc0]
            r_in = ([r_kc0], r_ext0)

            # Store the max number of CS stimuli across all intervals
            max_num_CS = 1
            max_num_US = 1

            for i in range(n_int):
                # Calculate the CS stimulus presentation times
                st_times, st_len = gen_int_times(n_batch, dt, T_stim, **kwargs)

                # Select the interval function to run
                int_fnc = trial_ls[i]
                # Calculate the interval inputs
                f_in = int_fnc(t_len, st_times, st_len, r_in, n_batch,
                               T_stim=T_stim, dt=dt, **kwargs)
                r_in, r_kct, r_extt, stim_ls, vt_opt = f_in

                # Run the forward pass
                net_out = self(r_kct, r_extt, time_int, n_batch, W_in)
                rt_int, (Wt_int, wt_int), vt_int = net_out
                # Pass the KC->MBON weights to the next interval
                W_in = (Wt_int[-1], wt_int[-1])

                # Append the interval outputs to lists
                rts += rt_int
                Wts += Wt_int
                wts += wt_int
                vts += vt_int
                vt_opts.append(vt_opt)
                time_CS.append(stim_ls[0])
                time_US.append(stim_ls[1])

                # Update max number of CS
                max_num_CS = max(max_num_CS, len(stim_ls[0]))
                max_num_US = max(max_num_US, len(stim_ls[1]))

            # # If the odors are not static for all trials, save odors
            # if not self.static_odors:
            #     # self.eval_odors.append(r_kc)
            #     self.eval_odors.append(trial_odors)

            # Convert the lists of time point values to a tensor,
            # time is the last dimension
            self.eval_rts.append(torch.stack(rts, dim=-1).detach())
            self.eval_Wts.append(torch.stack(Wts, dim=-1).detach())
            self.eval_wts.append(torch.stack(wts, dim=-1).detach())
            self.eval_vts.append(torch.stack(vts, dim=-1).detach())
            self.eval_vt_opts.append(torch.cat(vt_opts, dim=-1).detach())

            # TODO: This is messy, clean up this storage method
            # Concatenate time lists to store
            trial_CSs = []
            for i in range(max_num_CS):
                CS_vec = torch.zeros(n_batch, t_len * n_int)
                for j in range(n_int):
                    try:
                        CS_vec[:, j * t_len:(j + 1) * t_len] = time_CS[j][i]
                    except IndexError:
                        pass
                trial_CSs.append(CS_vec)

            trial_USs = []
            for i in range(max_num_US):
                US_vec = torch.zeros(n_batch, t_len * n_int)
                for j in range(n_int):
                    try:
                        US_vec[:, j * t_len:(j + 1) * t_len] = time_US[j][i]
                    except IndexError:
                        pass
                trial_USs.append(US_vec)

            # Store the time series lists
            self.eval_CS_stim.append(trial_CSs)
            self.eval_US_stim.append(trial_USs)

            # TODO: I think the loss function fails for batch sizes of 1
            # Calculate the loss
            # print(self.eval_vts[-1].shape)
            # print(self.eval_vt_opts[-1].shape)
            # print(self.eval_rts[-1][:, -self.n_dan:, :].shape)
            loss = cond_loss(self.eval_vts[-1], self.eval_vt_opts[-1],
                             self.eval_rts[-1][:, -self.n_dan:, :])
            self.eval_loss.append(loss.item())

    def gen_r_kc_ext(self, n_batch, pos_vt=None, **kwargs):
        """ Generates neuron activations for context and odor inputs.

        Parameters
            n_batch = number of trials in eval-batch
            pos_vt (kwarg) = indicates whether valence should be positive
                             None: random valence
                             True: positive valence
                             False: negative valence
        """

        # Determine the contextual input (r_ext)
        if pos_vt is None:
            r_ext = torch.multinomial(torch.ones(n_batch, self.n_ext),
                                      self.n_ext)
        elif pos_vt:
            r_ext = torch.tensor([1, 0]).repeat(n_batch, 1)
        elif not pos_vt:
            r_ext = torch.tensor([0, 1]).repeat(n_batch, 1)
        else:
            raise Exception('Not a valid value for pos_vt')

        # Determine odor input (r_kc)
        r_kc = torch.zeros(n_batch, self.n_kc)
        for b in range(n_batch):
            # Define an odor (CS) for each trial
            if self.train_odors is not None:
                n_odors = self.train_odors.shape[0]
                odor_select = torch.randint(n_odors, (1,))
                r_kc_inds = self.train_odors[odor_select, :]
            else:
                r_kc_inds = torch.multinomial(torch.ones(self.n_kc), self.n_ones)
            r_kc[b, r_kc_inds] = 1

        return r_kc, r_ext
