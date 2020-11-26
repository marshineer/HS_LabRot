# Import the required packages
from network_classes.paper_tasks.common import *


class RunNetwork:
    def __init__(self, *, net_instance):
        # Set the (trained) input network as a class variable
        self.net = net_instance
        # Copy parameters from the trained network
        self.n_kc = net_instance.n_kc
        self.n_dan = net_instance.n_dan
        self.n_ext = net_instance.n_ext
        self.n_ones = net_instance.n_ones
        if type(net_instance).__name__ == 'NoPlasticityRNN':
            self.eval_odors = net_instance.eval_odors
            self.static_odors = True
        else:
            self.eval_odors = []
            self.static_odors = False
        # Create variables to store parameters after each evaluation
        self.eval_rts = None
        self.eval_Wts = None
        self.eval_wts = None
        self.eval_vts = None
        self.eval_vt_opts = None
        self.eval_CS_stim = None
        self.eval_US_stim = None
        self.eval_loss = None
        # Create variables to store training parameters (for continuation)
        self.train_rts = None
        self.train_Wts = None
        self.train_wts = None
        self.train_vts = None
        self.train_vt_opts = None
        self.train_CS_stim = None
        self.train_US_stim = None
        self.train_loss = None

    def run_eval(self, *, trial_ls, T_int=30, dt=0.5, n_batch=1, n_trials=1,
                 reset_wts=True, **kwargs):
        """ Runs an evaluation based on a series of input functions

        Parameters
            trial_ls = list of interval functions that compose a trial
            T_int = length of a task interval (in seconds)
            dt = time step of simulation (in seconds)
            n_batch = number of parallel trials in a batch
            n_trials = number of trials to run
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

        # Interval time vector
        time_int = torch.arange(0, T_int + dt / 10, dt)
        t_len = time_int.shape[0]

        # Initialize the KC-MBON weights and plasticity variable
        W_in = None

        # For each function in the list, run an interval
        # All intervals together compose a single trial
        for tr in range(n_trials):
            # Lists to store activities, weights, readouts and target valences
            rts = []
            Wts = []
            wts = []
            vts = []
            vt_opts = []
            time_CS = []
            time_US = []

            # Determine whether to reset KC->MBON weights between trials
            if reset_wts:
                W_in = self.net.init_w_kc_mbon(self, W_in, n_batch,
                                               (tr, n_trials))
            # else:
            #     # Wt0 = (self.eval_Wts[-1][:, :, :, -1])
            #     # wt0 = (self.eval_wts[-1][:, :, :, -1])
            #     # W_in = (Wt0, wt0)
            #     W_in = (Wt_int[-1].detach(), wt_int[-1].detach())

            for i in range(len(trial_ls)):
                # Calculate the CS stimulus presentation times
                st_times, st_len = self.gen_st_times(dt, n_batch, **kwargs)

                # Generate odors and context (odor = KC = CS, context = ext = US)
                r_in = self.gen_r_kc_ext(n_batch, **kwargs)
                # r_kc, r_ext = self.gen_r_kc_ext(n_batch, **kwargs)
                if not self.static_odors:
                    # self.eval_odors.append(r_kc)
                    self.eval_odors.append(r_in[0])
                # r_in = (r_kc, r_ext)

                # Select the interval function to run
                int_fnc = trial_ls[i]
                # Calculate the interval inputs
                f_in = int_fnc(t_len, st_times, st_len, r_in, n_batch, **kwargs)
                r_kct, r_extt, stim_ls, vt_opt = f_in

                # Run the forward pass
                net_out = self.net(r_kct, r_extt, time_int, n_batch, W_in)
                rt_int, (Wt_int, wt_int), vt_int = net_out
                # Pass the KC->MBON weights to the next interval
                W_in = (Wt_int[-1].detach(), wt_int[-1].detach())

                # Append the interval outputs to lists
                rts += rt_int
                Wts += Wt_int
                wts += wt_int
                vts += vt_int
                vt_opts += vt_opt
                time_CS += stim_ls[0]
                time_US += stim_ls[1]

            # Concatenate the activities, weights and valences
            self.eval_rts.append(torch.stack(rts, dim=-1).detach())
            self.eval_Wts.append(torch.stack(Wts, dim=-1).detach())
            self.eval_wts.append(torch.stack(wts, dim=-1).detach())
            self.eval_vts.append(torch.stack(vts, dim=-1).detach())
            self.eval_vt_opts.append(torch.stack(vts, dim=-1).detach())
            self.eval_CS_stim.append(torch.stack(time_CS, dim=-1).detach())
            self.eval_US_stim.append(torch.stack(time_US, dim=-1).detach())

            # Calculate the loss
            loss = cond_loss(self.eval_vts[-1], self.eval_vt_opts[-1],
                             self.eval_rts[-1][:, -self.n_dan:, :])
            self.eval_loss.append(loss.item())

    def gen_r_kc_ext(self, n_batch, pos_val=None, **kwargs):
        """ Generates neuron activations for context and odor inputs.

        Parameters
            n_batch = number of trials in eval-batch
            pos_vt (kwarg) = indicates whether valence should be positive
                             None: random valence
                             True: positive valence
                             False: negative valence
        """

        # Determine the contextual input (r_ext)
        if pos_val is None:
            r_ext = torch.multinomial(torch.ones(n_batch, self.n_ext),
                                      self.n_ext)
        elif pos_val:
            r_ext = torch.tensor([1, 0]).repeat(n_batch, 1)
        elif not pos_val:
            r_ext = torch.tensor([0, 1]).repeat(n_batch, 1)
        else:
            print('Not a valid value for pos_val')

        # Determine odor input (r_kc)
        r_kc = torch.zeros(n_batch, self.n_kc)
        for b in range(n_batch):
            # Define an odor (CS) for each trial
            if self.static_odors is not None:
                odor_select = self.static_odors.shape[0]
                r_kc_inds = self.eval_odors[odor_select, :]
            else:
                r_kc_inds = torch.multinomial(torch.ones(self.n_kc), self.n_ones)
            r_kc[b, r_kc_inds] = 1

        return r_kc, r_ext
