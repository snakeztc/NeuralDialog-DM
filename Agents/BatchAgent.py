import logging
import numpy as np


class BatchAgent(object):

    logger = None
    domain = None
    representation = None
    random_state = None

    def __init__(self, domain, representation, seed=1):
        self.logger = logging.getLogger("hrl.BatchAgent." + self.__class__.__name__)
        self.domain = domain
        self.random_state = np.random.RandomState(seed)
        self.representation = representation

    def learn(self, experiences, max_iter=20):
        raise NotImplementedError("Implement batch learning")



