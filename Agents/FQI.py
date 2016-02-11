import numpy as np
from BatchAgent import BatchAgent
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import tree


class FQI(BatchAgent):
    show_resd = False
    mini_batch_size = 1000
    print "mini_batch size is " + str(mini_batch_size)

    def __init__(self, domain, representation, seed=1):
        super(FQI, self).__init__(domain, representation, seed)

    def learn(self, experiences, max_iter=20):
        if self.mini_batch_size:
            reduced_exp_table = experiences[self.random_state.choice(experiences.shape[0], self.mini_batch_size), :]
        else:
            reduced_exp_table = experiences

        # experience is in (phi_sa, r, phi_ns)
        phi_sa_size = self.representation.state_features_num + self.domain.actions_num
        X = reduced_exp_table[:, 0:phi_sa_size]
        rewards = reduced_exp_table[:, phi_sa_size]
        phi_ns = reduced_exp_table[:,phi_sa_size+1:]

        for i in range(0, max_iter):
            if self.show_resd:
                old_qs = self.representation.Q_phi_sa(X).ravel()

            #nqs = self.representation.Qs(next_states)
            nqs = self.representation.Qs_phi_s(phi_ns)
            best_nqs = np.amax(nqs, axis=1).ravel()
            y = rewards+ self.domain.discount_factor * best_nqs

            if self.show_resd:
                resd = np.mean(np.abs(y - old_qs))
                print "Residual is " + str(resd)

            #self.representation.model = tree.DecisionTreeRegressor(random_state=self.random_state)
            if not self.representation.model:
                self.representation.model = linear_model.SGDRegressor(alpha=0.01)
            self.representation.model.partial_fit(X, y)
