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

    def initialisation_213(self, i):
        pass

    def termination_213(self, i):
        pass

    def recursion_213(self, i):
        pass

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
        
        hidden_states_sequence_213 = np.zeros(len(seq) + 1)

        for i in len(seq):
            if (i == 1):
                hidden_states_sequence_213[i] = initialisation_213(i)
            elif (i == len(seq)):
                hidden_states_sequence_213[i] = termination_213(i)
            else:
                hidden_states_sequence_213[i] = recursion_213(i)

        return hidden_states_sequence_213[1:]

        # DONE
