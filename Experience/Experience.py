import numpy as np


class Experiences (object):
    experience = None
    priority = None
    exp_head = None
    exp_size = None
    exp_actual_size = None
    use_priority = None
    random_state = None
    mini_batch_size = None
    alpha_priority = None
    max_priority = None

    def __init__(self, use_priority, alpha_priority, mini_batch_size, seed):
        self.use_priority = use_priority
        self.alpha_priority = alpha_priority
        self.mini_batch_size = mini_batch_size
        self.random_state = np.random.RandomState(seed)

    # used by the hybrid system to add supervised aux signals
    def get_spl_experience(self, sample_indices):
        return None

    def add_experience(self, phi_s, policy_s, a, r, phi_ns, policy_ns, spl_targets=None):
        raise NotImplementedError("add experience")

    def update_priority(self, sample_indices, td_error):
        raise NotImplementedError("update priority")

    def set_alpha(self, cur_alpha):
        self.alpha_priority = cur_alpha

    def sample_mini_batch(self):
        """
        :param mini_batch_size:
        :return: sampled (phi_s, a, r, phi_ns, policy_ns, batch_index)
        """
        raise NotImplementedError("sample mini_batch_size")

