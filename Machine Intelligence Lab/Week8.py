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
    
    
    def init_213 (self, t):
        return self.pi[t]
    
    def main_init_213 (self, s, t, seq):
        return self.pi[t] * self.B[self.states_dict[s[t]]][self.emissions_dict[seq[t]]]
    
    def rec_213 (self, s, t, seq):
        if (t == 1):
            return self.init_213(0) * self.B[self.states_dict[s[0]]][self.emissions_dict[seq[0]]] * self.A[self.states_dict[s[t]]][self.states_dict[s[t-1]]] * self.B[self.states_dict[s[t]]][self.emissions_dict[seq[t]]]
        
        else:
            return self.rec_213(s, t-1, seq) * self.A[self.states_dict[s[t]]][self.states_dict[s[t-1]]] * self.B[self.states_dict[s[t]]][self.emissions_dict[seq[t]]]
    
    def product_213(self, *args, reps):
        result = [[]]
        for i in [tuple(i) for i in args] * reps: result = [x+[y] for x in result for y in i]
        for prod in result: yield tuple(prod)
    
    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        # TODO
        
        hidden_states_sequence_213 = []
        nu_213 = 0

        for s in self.product_213(self.states, reps = len(seq)):
            maybe_nu_213 = 0
            for t in range(len(seq)):
                if (t == 0):
                    maybe_nu_213 = self.main_init_213(list(s), t, seq)
                else:
                    maybe_nu_213 = self.rec_213(list(s), t, seq)
            if(maybe_nu_213 > nu_213):
                hidden_states_sequence_213 = list(s)
                nu_213 = maybe_nu_213

        return hidden_states_sequence_213
        # DONE
