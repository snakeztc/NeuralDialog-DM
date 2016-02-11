import logging
import numpy as np
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

    def Q(self, s, aID):
        raise NotImplementedError("Implement the q-funciton")

    def phi_sa(self, s, aID):
        raise NotImplementedError("Implement phi_sa")

    def phi_s(self, s):
        raise NotImplementedError("Implement phi")

    def expand_state_space(self, s, state_type):
        #time1 = time.time()
        phi = np.zeros((s.shape[0], self.state_features_num))
        num_var = s.shape[1]
        for var_idx in xrange(0, num_var):
            if state_type[var_idx] == self.domain.categorical:
                indices = [self.base[var_idx]+v+i*phi.shape[1] for i, v, in enumerate(s[:, var_idx])]
                phi.flat[indices] = 1
            else:
                phi[:, self.base[var_idx]] = s[:, var_idx]
        #time2 = time.time()
        #print '%s function took %0.3f ms' % ("expand", (time2-time1)*1000.0)
        return phi

