from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy
from Utils.Tools import vec2id
import numpy as np

class QLearning(Agent):
    cache = None

    def learn(self, s, performance_run=False):
        root = self.representation.root
        s_id = vec2id(s, self.domain.statespace_limits)[1]

        if s_id not in self.cache:
            Qs = self.representation.Qs(root, s)
            self.cache[s_id] = Qs
        else:
            Qs = self.cache.get(s_id)

        # choose an action
        if performance_run:
            aID = self.performance_policy.choose_action(Qs)
        else:
            aID = self.learning_policy.choose_action(Qs)

        (r, ns, terminal) = self.domain.step(s, aID)

        if not performance_run:
            self.logger.info("Learning here!")

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1):
        super(QLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(0.1)
        self.cache = {}

