import numpy as np
from Representation import Representation
import time


class StructTurnHistoryRep(Representation):

    # since we don't use state we create our own state limit and state type and feature base
    turn_state_limits = None
    turn_state_type = None

    def __init__(self, domain, seed=1):
        super(StructTurnHistoryRep, self).__init__(domain, seed)
        # initialize the model
        self.model = None
        # user response ngram size + action number + computer response
        self.state_features_num = self.domain.ngram_size
        self.sys_feature_num = self.domain.actions_num+1
        self.ngram_base = self.domain.actions_num + 1

    # State Representation #
    def phi_sa(self, s, aID):
        pass

    def phi_a(self, aID):
        pass

    def phi_s(self, s):
        # !! we assume "s" is just one sample, can never be more than that
        t_hist = s[2]
        usr_resp = t_hist["usr"].toarray()

        sys_resp = np.zeros((usr_resp.shape[0], self.sys_feature_num))
        indices = [int(v + idx*self.sys_feature_num) for idx, v in enumerate(t_hist["sys"])]
        sys_resp.flat[indices] = 1

        phi_s = {"usr": np.reshape(usr_resp, (1,) + usr_resp.shape),
                 "sys": np.reshape(sys_resp, (1,) + sys_resp.shape),
                 "cmp": np.atleast_3d(t_hist["cmp"])}
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
            result = {key: np.zeros((phi_s["usr"].shape[0], size)) for key, size in self.domain.policy_action_num.iteritems()}
            return result





