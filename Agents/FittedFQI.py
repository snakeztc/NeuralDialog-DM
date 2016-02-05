import numpy as np
from BatchAgent import BatchAgent
from sklearn.neighbors import KNeighborsRegressor


class FittedFQI(BatchAgent):
    def __init__(self, domain, representation, seed=1):
        super(FittedFQI, self).__init__(domain, representation, seed)

    def learn(self, experiences, max_iter=20):
        # experience is in (s, a, r, ns)
        states = experiences[:, 0:self.domain.state_space_dims]
        actions = experiences[:, self.domain.state_space_dims]
        rewards = experiences[:, self.domain.state_space_dims+1]
        next_states = experiences[:, self.domain.state_space_dims+2:]
        X = self.representation.phi_sa("root", states, actions)

        for i in range(0, max_iter):
            #old_qs = np.reshape(self.representation.Q("root", states, actions), (-1, 1))
            nqs = self.representation.Qs("root", next_states)
            best_nqs = np.reshape(np.amax(nqs, axis=1), (-1, 1))
            y = rewards+ self.domain.discount_factor * best_nqs
            #resd = np.mean(np.abs(y - old_qs))
            model = KNeighborsRegressor(n_neighbors=2, n_jobs=-1)
            model.fit(X, y)
            self.representation.models["root"] = model
            #print "Residual is " + str(resd)




