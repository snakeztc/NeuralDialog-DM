import numpy as np
from BatchAgent import BatchAgent
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import tree


class FQI(BatchAgent):
    show_resd = False
    mini_batch_size = 3000
    print "mini_batch size is " + str(mini_batch_size)

    def __init__(self, domain, representation, seed=1):
        super(FQI, self).__init__(domain, representation, seed)

    def learn(self, experiences, max_iter=20):
        if self.mini_batch_size:
            reduced_exp_table = experiences[self.random_state.choice(experiences.shape[0], self.mini_batch_size), :]
        else:
            reduced_exp_table = experiences

        # experience is in (s, a, r, ns)
        states = reduced_exp_table[:, 0:self.domain.statespace_dims]
        actions = reduced_exp_table[:, self.domain.statespace_dims]
        rewards = reduced_exp_table[:, self.domain.statespace_dims+1]
        next_states = reduced_exp_table[:, self.domain.statespace_dims+2:]
        X = self.representation.phi_sa(states, actions)

        for i in range(0, max_iter):
            if self.show_resd:
                old_qs = self.representation.Q(states, actions).ravel()

            nqs = self.representation.Qs(next_states)
            best_nqs = np.amax(nqs, axis=1).ravel()
            y = rewards+ self.domain.discount_factor * best_nqs

            if self.show_resd:
                resd = np.mean(np.abs(y - old_qs))
                print "Residual is " + str(resd)

            #model = KNeighborsRegressor(n_neighbors=2, n_jobs=-1)
            self.representation.model = tree.DecisionTreeRegressor()
            #if not self.representation.model:
            #    self.representation.model = linear_model.SGDRegressor(alpha=0.01, warm_start=True)
            self.representation.model.fit(X, y)
