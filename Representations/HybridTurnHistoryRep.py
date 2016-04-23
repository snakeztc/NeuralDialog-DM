import numpy as np
from Representation import Representation
import time


class HybridTurnHistoryRep(Representation):

    # since we don't use state we create our own state limit and state type and feature base
    turn_state_limits = None
    turn_state_type = None

    def __init__(self, domain, seed=1):
        super(HybridTurnHistoryRep, self).__init__(domain, seed)
        # initialize the model
        self.model = None
        # user response ngram size + action number + prev_hypothesis + prev_db_count
        self.state_features_num = self.domain.actions_num + 1 + self.domain.ngram_size + 1
        self.ngram_base = self.domain.actions_num + 1

    # State Representation #
    def phi_sa(self, s, aID):
        pass

    def phi_a(self, aID):
        pass

    def phi_s(self, s):
        # !! we assume "s" is just one sample, can never be more than that
        t_hist = s[2]
        phi_s = np.zeros((len(t_hist["sys"]), self.state_features_num))
        phi_s[:, self.ngram_base:-1] = t_hist["usr"].toarray()
        sys_indices = [int(v + idx*self.state_features_num) for idx, v in enumerate(t_hist["sys"])]
        phi_s.flat[sys_indices] = 1
        phi_s[:, -1] = t_hist["db"]

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
            return self.model.predict(phi_s)
        else:
            result = [np.zeros((phi_s.shape[0], size)) for size in self.domain.policy_action_num]
            result += [np.zeros((phi_s.shape[0], size)) for size in self.domain.spl_modality]
            return result





