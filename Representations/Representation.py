import logging
import numpy as np


class Representation(object):
    # the model used to predict the q value given s, a
    model = None
    #: The Domain that this Representation is modeling
    domain = None
    #: Number of features in the representation
    state_features_num = 0
    # A simple object that records the prints in a file
    logger = None
    # A seeded numpy random number generator
    random_state = None
    # state_base
    base = None

    def __init__(self, domain, seed=1):

        self.domain = domain
        self.logger = logging.getLogger("Representations." + self.__class__.__name__)
        self.random_state = np.random.RandomState(seed)

    def Qs(self, s):
        raise NotImplementedError("Implement Qs")

    def Q(self, s, aID):
        raise NotImplementedError("Implement the q-funciton")

    def phi_sa(self, s, aID):
        raise NotImplementedError("Implement phi_sa")

    def phi_s(self, s):
        raise NotImplementedError("Implement phi")

    def expand_state_space(self, s, state_limit, state_type):
        phi = np.zeros((s.shape[0], self.state_features_num))
        num_var = s.shape[1]
        for var_idx in range(0, num_var):
            if state_type[var_idx] == self.domain.categorical:
                for value_idx in range(int(state_limit[var_idx, 0]), int(state_limit[var_idx, 1])):
                    mask = np.where(s[:, var_idx] == (value_idx + state_limit[var_idx, 0]))
                    phi[mask, self.base[var_idx] + value_idx] = 1
            else:
                phi[:, self.base[var_idx]] = s[:, var_idx]
        return phi

