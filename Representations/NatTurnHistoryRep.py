import numpy as np
from Representation import Representation
import time


class NatTurnHistoryRep(Representation):

    # since we don't use state we create our own state limit and state type and feature base
    turn_state_limits = None
    turn_state_type = None

    def __init__(self, domain, seed=1):
        super(NatTurnHistoryRep, self).__init__(domain, seed)
        # initialize the model
        self.model = None
        # user response ngram size + action number + computer response
        self.state_features_num = self.domain.ngram_size + self.domain.actions_num + 1

    # State Representation #
    def phi_sa(self, s, aID):
        pass

    def phi_a(self, aID):
        pass

    def phi_s(self, s):
        # !! we assume "s" is just one sample, can never be more than that
        phi_s = np.copy(s[2])
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




