import logging
import numpy as np


class Policy(object):
    def __init__(self, seed=1):
        self.logger = logging.getLogger("hrl.Policy." + self.__class__.__name__)
        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState(seed)

    def choose_action(self, Qs):
        raise NotImplementedError("Implement the policy")

    def rargmax(self, Qs):
        """ Argmax that chooses randomly among eligible maximum indices. """
        return self.random_state.choice(np.argwhere(Qs.ravel() == np.amax(Qs)).ravel())

    def boltzmann(self, Qs, T):
        exp_Qs = np.exp(Qs / T)
        prob = exp_Qs / np.sum(exp_Qs)
        sel = self.random_state.choice(np.arange(Qs.shape[1]), p=prob.ravel())
        return sel


class RandomPolicy(Policy):
    def choose_action(self, Qs):
        return self.random_state.randint(0, len(Qs))


class GreedyPolicy(Policy):
    def choose_action(self, Qs):
        return self.boltzmann(Qs, 0.5)
        #return self.rargmax(Qs)


class EpsilonGreedyPolicy(Policy):
    epsilon = 0.1

    def choose_action(self, Qs):
        if self.random_state.random_sample() < self.epsilon:
            return self.random_state.randint(0, Qs.shape[1])
        else:
            return self.boltzmann(Qs, 0.5)
            #return self.rargmax(Qs)

    def __init__(self, epsilon, seed):
        super(EpsilonGreedyPolicy, self).__init__(seed)
        self.epsilon = epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
