from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy
from Utils.Tools import vec2id
import numpy as np


class QLearning(Agent):

    def learn(self, s, performance_run=False):
        Qs = self.representation.Qs(s)

        # choose an action
        if performance_run:
            aID = self.performance_policy.choose_action(Qs)
        else:
            aID = self.learning_policy.choose_action(Qs)

        (r, ns, terminal) = self.domain.step(s, aID)

        if not performance_run:
            self.logger.info("Learning here for tabular")

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1, verbose=False):
        super(QLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(0.1, seed)
        self.learning_rate = 0.1
        self.verbose = verbose
