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

    def __init__(self, use_priority, mini_batch_size, seed):
        self.use_priority = use_priority
        self.mini_batch_size = mini_batch_size
        self.random_state = np.random.RandomState(seed)

    def add_experience(self, phi_s, a, r, phi_ns, priority):
        raise NotImplementedError("add experience")

    def update_priority(self, sample_indices, td_error):
        raise NotImplementedError("update priority")

    def sample_mini_batch(self):
        """
        :param mini_batch_size:
        :return: sampled (phi_s, a, r, phi_ns, batch_index)
        """
        raise NotImplementedError("sample mini_batch_size")

