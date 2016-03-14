import logging
import numpy as np
from Utils.config import *

class Policy(object):
    def __init__(self, seed=1):
        self.logger = logging.getLogger("hrl.Policy." + self.__class__.__name__)
        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState(seed)

    def choose_action(self, Qs, mask=None):
        raise NotImplementedError("Implement the policy")

    #def rargmax(self, Qs):
    #    """ Argmax that chooses randomly among eligible maximum indices. """
    #    return self.random_state.choice(np.argwhere(Qs.ravel() == np.amax(Qs)).ravel())

    def argmax(self, Qs, mask=None):
        if mask is not None:
            valid_idxes = np.argwhere(mask).ravel()
            sel_idx = np.argmax(Qs[0, mask])
            return valid_idxes[sel_idx]
        else:
            return self.random_state.choice(np.argwhere(Qs.ravel() == np.amax(Qs)).ravel())

    def boltzmann(self, Qs, T, mask=None):
        if mask is not None:
            valid_idxes = np.argwhere(mask).ravel()
            valid_qs = Qs[0, mask]
            exp_Qs = np.exp(valid_qs / T)
            prob = exp_Qs / np.sum(exp_Qs)
            sel = self.random_state.choice(np.arange(sum(mask)), p=prob.ravel())
            return valid_idxes[sel]
        else:
            exp_Qs = np.exp(Qs / T)
            prob = exp_Qs / np.sum(exp_Qs)
            sel = self.random_state.choice(np.arange(Qs.shape[1]), p=prob.ravel())
            return sel


class RandomPolicy(Policy):
    def choose_action(self, Qs, mask=None):
        if mask is not None:
            valid_idxes = np.argwhere(mask).ravel()
            sel_idx = self.random_state.randint(0, sum(mask))
            return valid_idxes[sel_idx]
        else:
            return self.random_state.randint(0, Qs.shape[1])


class GreedyPolicy(Policy):
    def choose_action(self, Qs, mask=None):
        return self.boltzmann(Qs, generalConfig["greedy_temp"], mask)


class EpsilonGreedyPolicy(Policy):

    def choose_action(self, Qs, mask=None):
        if self.random_state.random_sample() < self.epsilon:
            return self.random_policy.choose_action(Qs, mask)
        else:
            return self.boltzmann(Qs, generalConfig["greedy_temp"], mask)

    def __init__(self, epsilon, seed):
        super(EpsilonGreedyPolicy, self).__init__(seed)
        self.epsilon = epsilon
        self.random_policy = RandomPolicy()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
