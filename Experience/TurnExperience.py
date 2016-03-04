import numpy as np
from Experience import Experiences


class TurnExperience (Experiences):
    phi_s_size = None
    max_len = None
    exp_ar = None

    def __init__(self, exp_size, phi_s_size, max_len, mini_batch_size, use_priority, seed):
        super(TurnExperience, self).__init__(use_priority=use_priority, mini_batch_size=mini_batch_size, seed=seed)
        # experience is a 3D tensor in shape num_sample * max_t * feature_dimension
        self.experience = np.zeros((exp_size, self.max_len, phi_s_size * 2 + 2))
        self.exp_ar = np.zeros((exp_size, 2))

        self.priority = np.zeros(exp_size)
        self.exp_size = exp_size
        self.max_len = max_len
        self.exp_head = 0
        self.exp_actual_size = 0
        self.phi_s_size = phi_s_size

    def add_experience(self, phi_s, a, r, phi_ns, priority):
        if self.exp_head >= self.exp_size:
            print "** reset exp header **"
            self.exp_head = 0

        # pad phi_s and phi_ns with 0 zeros in the front
        self.experience[self.exp_head, self.max_len-phi_s.shape[0]:self.max_len, 0:self.phi_s_size] = phi_s
        self.experience[self.exp_head, self.max_len-phi_ns.shape[0]:self.max_len, self.phi_s_size:] = phi_ns

        self.exp_ar[self.exp_head, 0] = a
        self.exp_ar[self.exp_head, 1] = r
        self.priority[self.exp_head] = 20.0

        # increment the write head
        self.exp_head += 1
        self.exp_actual_size += 1

    def sample_mini_batch(self):
        sample_size = np.min([self.exp_actual_size, self.exp_size])
        prob = self.priority[0:sample_size] / np.sum(self.priority[0:sample_size])
        sample_indices = self.random_state.choice(a=sample_size, size=self.mini_batch_size, p=prob, replace=False)

        mini_batch_exp = self.experience[sample_indices, :, :]
        phi_s = mini_batch_exp[:, :, 0:self.phi_s_size]
        actions = self.exp_ar[sample_indices, 0]
        rewards = self.exp_ar[sample_indices, 1]
        phi_ns = mini_batch_exp[:,:,self.phi_s_size:]
        return phi_s, actions, rewards, phi_ns, sample_indices

    def update_priority(self, sample_indices, td_error):
        self.priority[sample_indices] = np.clip(td_error, 0, 20) + 1.0





