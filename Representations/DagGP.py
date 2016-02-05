import numpy as np
from Representation import Representation



class DagLinear(Representation):

    state_feature_base = None

    def __init__(self, domain, root, tree, terminals, seed=1):
        super(DagLinear, self).__init__(domain, root, tree, terminals, seed)
        # initialize the model
        self.models = {}
        self.state_features_num = int(np.sum(self.domain.statespace_limits[:, 1]
                                             - self.domain.statespace_limits[:, 0]))
        self.state_feature_base = np.append([0], np.cumsum(self.domain.statespace_limits[:-1, 1]
                                         - self.domain.statespace_limits[:-1, 0]))

    def phi_sa(self, o, s, u):
        """
        Get the feature vector for a subtask o at state s with action u
        :param o: the name of subtask
        :param s: the raw state vector
        :param u: the index!! of action in a np array
        :return: the feature vector
        """
        # feature vector
        phi = self.phi_s(s)
        phi_sa = np.hstack((phi, u))
        return phi_sa

    def phi_s(self, s):
        phi = s
        return phi

    def Q(self, o, s, u):
        phi_sa = self.phi_sa(o, s, u)
        model = self.models.get(o, None)
        if model:
            q = model.predict(phi_sa).ravel()
        else:
            q = np.zeros(s.shape[0])
        return q

    def Qs(self, o, s):
        if s.ndim < 2:
            s = np.matrix(s)
        qs = np.zeros((s.shape[0], len(self.tree.get(o))))
        for idx, u in enumerate(self.tree.get(o)):
            temp_uIDs = np.ones((s.shape[0], 1)) * idx
            qs[:, idx] = self.Q(o, s, temp_uIDs)
        return qs




