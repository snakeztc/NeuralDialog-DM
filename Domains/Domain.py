""" Domain Abstract Class """
import logging
import numpy as np


class Domain(object):
    # NL interface
    vocabs = None
    nb_words = 0

    categorical = 1
    discrete = 2
    continuous = 3
    #: The discount factor by which rewards are reduced
    discount_factor = .9
    #: The number of Actions the agent can perform
    actions_num = 0  # was None
    #: Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements
    # [min, max]
    statespace_limits = None  # was None
    #: Number of dimensions of the state space
    statespace_size = 0  # was None
    # a list of dimension type
    statespace_type = []
    #: The cap used to bound each episode (return to state 0 after)
    episode_cap = None
    #: A simple object that records the prints in a file
    logger = None
    # A seeded numpy random number generator
    random_state = None
    # Different type of actions
    action_types = None

    def __init__(self, seed=1):
        self.logger = logging.getLogger("hrl.Domains." + self.__class__.__name__)
        self.statespace_size = len(self.statespace_limits)
        # To make sure type of discount_factor is float. This will later on be used in
        self.discount_factor = float(self.discount_factor)
        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState(seed)

    def is_terminal(self, s):
        raise NotImplementedError("Implement initial state method")

    def possible_actions(self, s=None):
        return np.arange(self.actions_num)

    def s0(self):
        raise NotImplementedError("Implement initial state method")

    def step(self, s, aID):
        """
        :param s: the state vector
        :param aID: the action index
        :return: The tuple (r, ns, t, p_actions) =
            (Reward [value], next observed state, isTerminal [boolean], observation (string from environment))
        """
        raise NotImplementedError("Any domain needs to implement step function")

    def action_prune(self, s):
        """
        :param s: the current state
        :return: a boolean mask that indicates which action is avaliable
        """
        return None
