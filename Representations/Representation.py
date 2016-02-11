import logging
import numpy as np
import numpy.matlib
import time


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

    def Qs_phi_s(self, phi_s):
        raise NotImplementedError("Implement Qs_phi")

    def Q(self, s, aID):
        raise NotImplementedError("Implement the q-funciton")

    def Q_phi_sa(self, phi_sa):
        raise NotImplementedError("Implement the Q phisa")

    def phi_sa(self, s, aID):
        raise NotImplementedError("Implement phi_sa")

    def phi_a(self, aID):
        raise NotImplementedError("Implement phi_a")

    def phi_s(self, s):
        raise NotImplementedError("Implement phi")

    def phi_s_phi_a(self, phi_s, phi_a):
        raise NotImplementedError("Implement phi_s_phi_a")

    def expand_state_space(self, s, state_type):
        phi = np.zeros((s.shape[0], self.state_features_num))
        num_var = s.shape[1]
        row_starts = np.linspace(0.0, (s.shape[0]-1) * phi.shape[1], num=s.shape[0])
        all_indices = ((self.base[:-1] + s).T + row_starts).T
        all_indices = all_indices.astype(int)
        flat_phi = phi.flat
        for var_idx in xrange(num_var):
            if state_type[var_idx] == self.domain.categorical:
                indices = all_indices[:, var_idx]
                flat_phi[indices] = 1
            else:
                phi[:, self.base[var_idx]] = s[:, var_idx]
        return phi

