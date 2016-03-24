import numpy as np
from Representation import Representation
import time


class TurnHistoryRep(Representation):

    # since we don't use state we create our own state limit and state type and feature base
    turn_state_limits = None
    turn_state_type = None

    def __init__(self, domain, seed=1):
        super(TurnHistoryRep, self).__init__(domain, seed)
        # initialize the model
        self.model = None
        self.state_features_num = 0
        # get state_feature_num
        # this state limit is different because we are not operating on the oracle state.
        # instead we are working on the turn output
        self.turn_state_limits = np.atleast_2d([[-1, self.domain.actions_num],
                                                [-1, len(self.domain.str_response)],
                                                [0, len(self.domain.corpus)]])
        self.turn_state_type = [self.domain.categorical, self.domain.categorical, self.domain.discrete]

        # get state feature number
        for idx, s_type in enumerate(self.turn_state_type):
            if s_type == self.domain.categorical:
                self.state_features_num += (self.turn_state_limits[idx, 1] - self.turn_state_limits[idx, 0])
            else:
                self.state_features_num += 1
        self.state_features_num = int(self.state_features_num)

        # get state base
        diff = self.turn_state_limits[:, 1] - self.turn_state_limits[:, 0]
        for row_idx in range(0, diff.size):
            if self.turn_state_type[row_idx] != self.domain.categorical:
                diff[row_idx] = 1
        self.base = np.append([0], np.cumsum(diff))

    # State Representation #
    def phi_sa(self, s, aID):
        pass

    def phi_a(self, aID):
        pass

    def phi_s(self, s):
        # !! we assume "s" is just one sample, can never be more than that
        phi_s = self.expand_state_space(s[2], self.turn_state_type, self.turn_state_limits)
        # convert from 2d to 3d
        phi_s = np.reshape(phi_s, (1,) + phi_s.shape)
        return phi_s

    def phi_s_phi_a(self, phi_s, phi_a):
        pass

    # Value function Representation ###
    def Q(self, s, aID):
        pass

    def Qs(self, s):
        phi_s = self.phi_s(s)
        return self.Qs_phi_s(phi_s)

    def Q_phi_sa(self, phi_sa):
        pass

    def Qs_phi_s(self, phi_s):
        # we assume that phi_s is in the format of num_sample * time_stamp * dimension
        if self.model:
            return self.model.predict({'input':phi_s})
        else:
            result = {key: np.zeros((phi_s.shape[0], size)) for key, size in self.domain.policy_action_num.iteritems()}
            return result





