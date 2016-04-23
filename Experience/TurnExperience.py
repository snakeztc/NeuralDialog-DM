import numpy as np
from Experience import Experiences


class TurnExperience (Experiences):
    phi_s_size = None
    max_len = None
    exp_ar = None

    def __init__(self, exp_size, phi_s_size, max_len, mini_batch_size, use_priority, alpha_priority, seed):
        super(TurnExperience, self).__init__(use_priority=use_priority, mini_batch_size=mini_batch_size,
                                             seed=seed, alpha_priority=alpha_priority)

        self.exp_ar = np.zeros((exp_size, 2))
        self.s_policies = [None] * exp_size
        self.ns_policies = [None] * exp_size

        self.priority = np.zeros(exp_size)
        self.exp_size = exp_size
        self.max_len = max_len + 1
        self.exp_head = 0
        self.exp_actual_size = 0
        self.phi_s_size = phi_s_size

        # experience is a 3D tensor in shape num_sample * max_t * feature_dimension
        self.experience = np.zeros((exp_size, self.max_len, phi_s_size * 2))

    def add_experience(self, phi_s, policy_s, a, r, phi_ns, policy_ns, priority, spl_targets=None):
        if self.exp_head >= self.exp_size:
            print "** reset exp header **"
            self.exp_head = 0

        # pad phi_s and phi_ns with 0 zeros in the front
        self.experience[self.exp_head, self.max_len-phi_s.shape[1]:self.max_len, 0:self.phi_s_size] = phi_s
        self.experience[self.exp_head, self.max_len-phi_ns.shape[1]:self.max_len, self.phi_s_size:] = phi_ns

        self.exp_ar[self.exp_head, 0] = a
        self.exp_ar[self.exp_head, 1] = r
        self.s_policies[self.exp_head] = policy_s
        self.ns_policies[self.exp_head] = policy_ns
        self.priority[self.exp_head] = 20.0

        # increment the write head
        self.exp_head += 1
        self.exp_actual_size += 1

    def sample_mini_batch(self):
        sample_size = np.min([self.exp_actual_size, self.exp_size])
        if self.use_priority:
            temp_prob = np.power(self.priority[0:sample_size], self.alpha_priority)
            prob = temp_prob / np.sum(temp_prob)
        else:
            prob = np.ones(sample_size) / sample_size

        sample_indices = self.random_state.choice(a=sample_size, size=self.mini_batch_size, p=prob, replace=False)

        mini_batch_exp = self.experience[sample_indices, :, :]
        phi_s = mini_batch_exp[:, :, 0:self.phi_s_size]
        actions = self.exp_ar[sample_indices, 0]
        rewards = self.exp_ar[sample_indices, 1]
        phi_ns = mini_batch_exp[:,:,self.phi_s_size:]
        policy_ns = [self.ns_policies[i] for i in sample_indices]
        policy_s = [self.s_policies[i] for i in sample_indices]
        return phi_s, policy_s, actions, rewards, phi_ns, policy_ns, sample_indices

    def update_priority(self, sample_indices, td_error):
        self.priority[sample_indices] = np.clip(td_error, 0, 20) + 1.0





