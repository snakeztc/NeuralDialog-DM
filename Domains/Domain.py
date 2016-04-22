""" Domain Abstract Class """
import logging
import numpy as np
from Utils.domainUtil import DomainUtil


class Domain(object):
    # terminal predicate
    end_idx = None
    end_success = 1

    # NL interface
    vocabs = None
    nb_words = 0

    categorical = 1
    discrete = 2
    continuous = 3
    #: The discount factor by which rewards are reduced
    discount_factor = None
    #: The number of Actions the agent can perform
    actions_num = None  # was None
    # Different type of actions
    action_types = None
    # The action hierarchy [action_name, ... ] num_actions
    action_to_policy = None
    # A dictionary action_name -> number of actions
    policy_action_num = None
    # tree node name
    policy_names = None
    # tree node string name
    policy_str_name = None
    # the global index base of each policy
    policy_bases = None
    # the string name of supervised learning signals
    spl_str_name = None
    # the index of supervised learning outputs
    spl_indexs = None
    # the modality of spl signals
    spl_modality = None
    # the variable type of spl
    spl_type = None
    # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
    statespace_limits = None  # was None
    #: Number of dimensions of the state space
    statespace_size = 0  # was None
    # a list of dimension type
    statespace_type = None
    #: The cap used to bound each episode (return to state 0 after)
    episode_cap = None
    #: A simple object that records the prints in a file
    logger = None
    # A seeded numpy random number generator
    random_state = None

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
        :return: reward, shape, (ns, n_w_hist, n_t_hist), self.is_terminal(ns)
        """
        raise NotImplementedError("Any domain needs to implement step function")

    def hybrid_step(self, s, aID, Qs):
        """
        :param s: the state vector
        :param aID: the action index
        :param Qs: the Qs contains supervised signal
        :return: reward, shape, (ns, n_w_hist, n_t_hist), self.is_terminal(ns), spl_targets
        """
        raise NotImplementedError("Any domain needs to implement step function")

    def action_prune(self, s):
        """
        :param s: the current state
        :return: a boolean mask that indicates which action is avaliable
        """
        return None