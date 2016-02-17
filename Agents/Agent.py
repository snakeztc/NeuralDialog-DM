import logging
import numpy as np
from Policies.Policy import GreedyPolicy


class Agent(object):
    logger = None
    domain = None
    representation = None
    random_state = None
    learning_policy = None
    performance_policy = None

    def __init__(self, domain, representation, seed = 1):
        self.logger = logging.getLogger("hrl.Agent." + self.__class__.__name__)
        self.domain = domain
        self.random_state = np.random.RandomState(seed)
        self.representation = representation
        self.performance_policy = GreedyPolicy(seed)

    def learn(self, s, performance_run=False):
        raise NotImplementedError("implement learning algorithm")