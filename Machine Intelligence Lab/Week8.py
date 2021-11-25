import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(zip(self.emissions, list(range(self.M))))

    def init_nu_213 (self, nu_213, seq):
        for i in range(self.N):
            nu_213[0, i] = self.pi[i] * self.B[i, self.emissions_dict[seq[0]]]
        return nu_213

    def nu_rec_213(self, seq, nu_213):
        for i in range(1, len(seq)):
            for j in range(self.N):
                atTimeT_213_max = -1
                for k in range(self.N): 
                    if nu_213[i - 1, k] * self.A[k, j] * self.B[j, self.emissions_dict[seq[i]]] > atTimeT_213_max:
                        atTimeT_213_max = nu_213[i - 1, k] * self.A[k, j] * self.B[j, self.emissions_dict[seq[i]]]
                nu_213[i, j] = atTimeT_213_max
        return nu_213

    def maybe_nu_rec_213(self, seq, nu_213, maybe_nu_213):
        for i in range(1, len(seq)):
            for j in range(self.N):
                atTimeT_213_max = -1
                max_nu_213 = -1
                for k in range(self.N):
                    if nu_213[i - 1, k] * self.A[k, j] * self.B[j, self.emissions_dict[seq[i]]] > atTimeT_213_max:
                        atTimeT_213_max = nu_213[i - 1, k] * self.A[k, j] * self.B[j, self.emissions_dict[seq[i]]]
                        max_nu_213 = k
                maybe_nu_213[i, j] = max_nu_213
        return maybe_nu_213

    def termination_213(self, seq, nu_213, max_nu_213):
        atTimeT_213_max = -1
        for i in range(self.N):
            if nu_213[len(seq) - 1, i] > atTimeT_213_max:
                atTimeT_213_max = nu_213[len(seq) - 1, i]
                max_nu_213 = i
        return max_nu_213

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """

        nu_213 = np.zeros((len(seq), self.N))
        nu_213 = self.init_nu_213(nu_213, seq)

        maybe_nu_213 = np.zeros((len(seq), self.N), dtype=int)        
        for i in range(self.N):
            maybe_nu_213[0, i] = 0
        
        nu_213 = self.nu_rec_213(seq, nu_213)
        maybe_nu_213 = self.maybe_nu_rec_213(seq, nu_213, maybe_nu_213)
        
        max_nu_213 = -1
        max_nu_213 = self.termination_213(seq, nu_213, max_nu_213)
        st_213 = [max_nu_213]
        
        for i in range(len(seq) - 1, 0, -1):
            st_213.append(maybe_nu_213[i, st_213[-1]])
        st_213.reverse()

        hidden_states_sequence = {x : y for y, x in self.states_dict.items()}
        return [hidden_states_sequence[i] for i in st_213]
