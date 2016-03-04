import numpy as np
from Representation import Representation
import time


class BinaryApproxRep(Representation):

    state_feature_base = None

    def __init__(self, domain, seed=1):
        super(BinaryApproxRep, self).__init__(domain, seed)
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

    ### State Representation ###
    def phi_sa(self, s, aID):
        """
        Get the feature vector for a subtask o at state s with action u
        :param s: the raw state vector
        :param aID: the action index
        :return: the feature vector
        """
        phi_s = self.phi_s(s)
        phi_a = self.phi_a(aID)
        return self.phi_s_phi_a(phi_s, phi_a)

    def phi_a(self, aID):
        if np.isscalar(aID):
            phi_a = np.zeros((1, self.domain.actions_num))
            phi_a[0, aID] = 1
        else:
            phi_a = np.zeros((aID.shape[0], self.domain.actions_num))
            indices = [v+i*phi_a.shape[1] for i, v, in enumerate(aID)]
            phi_a.flat[indices] = 1
        return phi_a

    def phi_s(self, s):
        phi = np.copy(s)
        return self.expand_state_space(phi, self.domain.statespace_type, self.domain.statespace_limits)

    def phi_s_phi_a(self, phi_s, phi_a):
        return np.column_stack((phi_s, phi_a))

    ### Value function Representation ###
    def Q(self, s, aID):
        phi_sa = self.phi_sa(s, aID)
        return self.Q_phi_sa(phi_sa)

    def Qs(self, s):
        phi_s = self.phi_s(s)
        return self.Qs_phi_s(phi_s)

    def Q_phi_sa(self, phi_sa):
        if self.model:
            q = self.model.predict(phi_sa).ravel()
        else:
            q = np.zeros(phi_sa.shape[0])
        return q

    def Qs_phi_s(self, phi_s):
        qs = np.zeros((phi_s.shape[0], self.domain.actions_num))
        actions = self.domain.possible_actions()
        base_aIDs = np.ones((phi_s.shape[0], 1))
        for idx, aID in enumerate(actions):
            temp_aIDs = base_aIDs * aID
            phi_sa = self.phi_s_phi_a(phi_s, self.phi_a(temp_aIDs))
            qs[:, idx] = self.Q_phi_sa(phi_sa)
        return qs





