from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy
from Agents.FQI import FQI
import numpy as np


class OnlineFQI(Agent):

    def learn(self, s, performance_run=False):
        Qs = self.representation.Qs(s)

        # choose an action
        if performance_run:
            aID = self.performance_policy.choose_action(Qs)
        else:
            aID = self.learning_policy.choose_action(Qs)

        (r, ns, terminal) = self.domain.step(s, aID)

        if not performance_run:
            phi_sa_size = self.representation.state_features_num + self.domain.actions_num
            experience = np.zeros((1, self.representation.state_features_num*2 + self.domain.actions_num + 1))
            experience[:, 0:phi_sa_size] = self.representation.phi_sa(s, aID)
            experience[:, phi_sa_size] = r
            experience[:, phi_sa_size+1:] = self.representation.phi_s(ns)
            if self.experience.shape[0] > 0:
                self.experience = np.vstack((self.experience, experience))
            else:
                self.experience = experience

            if (self.experience.shape[0] % self.update_frequency) == 0:
                self.learner.learn(self.experience, max_iter=self.max_iter)
                self.representation = self.learner.representation

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1, epsilon=0.3, update_frequency=10, max_iter=10):
        super(OnlineFQI, self).__init__(domain, representation, seed)
        epsilon = epsilon
        self.learning_policy = EpsilonGreedyPolicy(epsilon, seed)
        self.experience = np.zeros((0, self.domain.statespace_size*2+2))
        self.update_frequency = update_frequency
        self.learner = FQI(domain, representation, seed)
        self.max_iter = max_iter
        print "Using epsilon " + str(epsilon)
        print "update_frequency " + str(self.update_frequency)
        print "FQI max_iter is " + str(self.max_iter)
