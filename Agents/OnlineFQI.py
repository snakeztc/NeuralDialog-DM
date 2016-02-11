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
            experience = np.zeros((1, self.domain.statespace_size * 2 + 2))
            experience[:, 0:self.domain.statespace_size] = np.copy(s)
            experience[:, self.domain.statespace_size] = aID
            experience[:, self.domain.statespace_size+1] = r
            experience[:, self.domain.statespace_size+2:] = np.copy(ns)
            if self.experience.shape[0] > 0:
                self.experience = np.vstack((self.experience, experience))
            else:
                self.experience = experience

            if (self.experience.shape[0] % self.update_frequency) == 0:
                self.learner.learn(self.experience, max_iter=self.max_iter)
                self.representation = self.learner.representation

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1):
        super(OnlineFQI, self).__init__(domain, representation, seed)
        epsilon = 0.3
        self.learning_policy = EpsilonGreedyPolicy(epsilon, seed)
        self.experience = np.zeros((0, self.domain.statespace_size*2+2))
        self.update_frequency = 100
        self.learner = FQI(domain, representation, seed)
        self.max_iter = 20
        print "Using epsilon " + str(epsilon)
        print "update_frequency " + str(self.update_frequency)
        print "FQI max_iter is " + str(self.max_iter)
