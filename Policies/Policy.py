import logging
import numpy as np


class Policy(object):
    def __init__(self, seed=1):
        self.logger = logging.getLogger("hrl.Policy." + self.__class__.__name__)
        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState(seed)

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
            random_a = self.random_state.randint(0, Qs.shape[1])
            return random_a
        else:
            return np.argmax(Qs)

    def __init__(self, epsilon, seed):
        super(EpsilonGreedyPolicy, self).__init__(seed)
        self.epsilon = epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
