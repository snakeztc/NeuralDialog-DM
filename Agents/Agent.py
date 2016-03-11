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
        """
        Agent takes in a state and make step
        :param s: the state
        :param performance_run: if take exploration
        :return: r, ns, terminal (boolean), observation (response from env)
        """
        raise NotImplementedError("implement learning algorithm")

    def print_episode(self, hist):
        for i, idx in enumerate(hist):
            if (i+1) % 25 == 0:
                print self.domain.vocabs[idx-1]
            else:
                print self.domain.vocabs[idx-1],
        print ""