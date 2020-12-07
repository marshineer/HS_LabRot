# Import the required packages
import numpy as np
from network_classes.paper_tasks.base_rnn import FirstOrderCondRNN
from network_classes.paper_tasks.continual_rnn import ContinualRNN
from network_classes.paper_tasks.all_conditioning_rnn import ExtendedCondRNN
from common.common import *


class NoRecurFirstO(FirstOrderCondRNN):
    def __init__(self, two_hop=True, **kwargs):
        super().__init__(**kwargs)
        self.two_hop = two_hop

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

        # Set the weights DAN->MBON, DAN->FBN, FBN->MBON, FBN->FBN to zero
        W_recur = self.W_recur.clone()
        if self.two_hop:
            W_recur[:(self.n_mbon + self.n_fbn), -(self.n_dan + self.n_fbn):] = 0
        else:
            W_recur[:self.n_mbon, -self.n_dan:] = 0
            W_recur[self.n_mbon:(self.n_mbon + self.n_fbn), :] = 0
            W_recur[:, self.n_mbon:(self.n_mbon + self.n_fbn)] = 0

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


class NoRecurExtended(NoRecurFirstO, ExtendedCondRNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set the number of task intervals
        self.n_int = 3


class NoRecurContinual(NoRecurFirstO, ContinualRNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
