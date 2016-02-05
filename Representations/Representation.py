import logging
import numpy as np


class Representation(object):

    #: dictionary of model, aID -> model
    models = None
    # tree structure aID -> and childrenID
    tree = None
    # a function from aID, s -> boolean
    terminals = None
    # the root: aID
    root = None
    #: The Domain that this Representation is modeling
    domain = None
    #: Number of features in the representation
    state_features_num = 0
    #: Number of actions in the representation
    actions_num = 0
    # number of options
    option_num = 0
    # A simple object that records the prints in a file
    logger = None
    # A seeded numpy random number generator
    random_state = None

    def __init__(self, domain, root, tree, terminals, seed=1):

        self.domain = domain
        self.actions_num = domain.actions_num
        self.root = root
        self.tree = tree
        self.terminals = terminals
        self.option_num = len(tree.keys())
        self.logger = logging.getLogger("hrl.Representations." + self.__class__.__name__)
        self.random_state = np.random.RandomState(seed)

    def Qs(self, o, s):
        raise NotImplementedError("Implement Qs")

    def Q(self, o, s, u):
        raise NotImplementedError("Implement the q-funciton")

    def phi_sa(self, o, s, u):
        raise NotImplementedError("Implement phi_sa")

    def phi_s(self, s):
        raise NotImplementedError("Implement phi")
