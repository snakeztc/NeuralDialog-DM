import logging
import numpy as np


class Policy(object):
    def __init__(self):
        self.logger = logging.getLogger("hrl.Policy." + self.__class__.__name__)
        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState()

    def choose_action(self, Qs):
        raise NotImplementedError("Implement the policy")


class RandomPolicy(Policy):
    def choose_action(self, Qs):
        return self.random_state.randint(0, len(Qs))


class GreedyPolicy(Policy):
    def choose_action(self, Qs):
        return np.argmax(Qs)


class EpsilonGreedyPolicy(Policy):
    epsilon = 0.1

    def choose_action(self, Qs):
        if self.random_state.random_sample() < self.epsilon:
            return self.random_state.randint(0, len(Qs))
        else:
            return np.argmax(Qs)

    def __init__(self, epsilon):
        super(EpsilonGreedyPolicy, self).__init__()
        self.epsilon = epsilon
