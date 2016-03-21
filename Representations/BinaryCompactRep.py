import numpy as np
from Representation import Representation
from Utils.config import dqnConfig


class BinaryCompactRep(Representation):

    state_feature_base = None

    def __init__(self, domain, seed=1):
        super(BinaryCompactRep, self).__init__(domain, seed)
        # initialize the model
        self.model = None
        self.state_features_num = 0
        # get state_feature_num
        for idx, type in enumerate(self.domain.statespace_type):
            if type == self.domain.categorical:
                self.state_features_num += (self.domain.statespace_limits[idx, 1] - self.domain.statespace_limits[idx, 0])
            else:
                self.state_features_num += 1
        self.state_features_num = int(self.state_features_num)
        # get state base
        diff = self.domain.statespace_limits[:, 1] - self.domain.statespace_limits[:, 0]
        for row_idx in range(0, diff.size):
            if self.domain.statespace_type[row_idx] != self.domain.categorical:
                diff[row_idx] = 1
        self.base = np.append([0], np.cumsum(diff))

    # State Representation
    def phi_sa(self, s, aID):
        pass

    def phi_a(self, aID):
        pass

    def phi_s(self, s):
        phi = np.copy(s[0])
        return self.expand_state_space(phi, self.domain.statespace_type, self.domain.statespace_limits)

    def phi_s_phi_a(self, phi_s, phi_a):
        pass

    ### Value function Representation ###
    def Q(self, s, aID):
        pass

    def Qs(self, s):
        phi_s = self.phi_s(s)
        return self.Qs_phi_s(phi_s)

    def Q_phi_sa(self, phi_sa):
        pass

    def Qs_phi_s(self, phi_s):
        if self.model:
            return self.model.predict({'input':phi_s})
            #return self.model.predict(phi_s)
        else:
            result = {key: np.zeros((phi_s.shape[0], size)) for key, size in self.domain.policy_action_num.iteritems()}
            return result





