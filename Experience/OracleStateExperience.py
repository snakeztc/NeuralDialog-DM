import numpy as np
from Experience import Experiences


class OracleStateExperience (Experiences):
    phi_s_size = None

    def __init__(self, exp_size, phi_s_size, mini_batch_size, use_priority, seed):
        super(OracleStateExperience, self).__init__(use_priority=use_priority, mini_batch_size=mini_batch_size, seed=seed)
        self.experience = np.zeros((exp_size, phi_s_size * 2 + 2))
        self.priority = np.zeros(exp_size)
        self.ns_policies = [None] * exp_size
        self.exp_size = exp_size
        self.exp_head = 0
        self.exp_actual_size = 0
        self.phi_s_size = phi_s_size

    def add_experience(self, phi_s, a, r, phi_ns, policy_ns, priority):
        if self.exp_head >= self.exp_size:
            print "** reset exp header **"
            self.exp_head = 0

        self.experience[self.exp_head, 0:self.phi_s_size] = phi_s
        self.experience[self.exp_head, self.phi_s_size] = a
        self.experience[self.exp_head, self.phi_s_size+1] = r
        self.experience[self.exp_head, self.phi_s_size+2:] = phi_ns
        self.priority[self.exp_head] = 20.0
        self.ns_policies[self.exp_head] = policy_ns

        # increment the write head
        self.exp_head += 1
        self.exp_actual_size += 1

    def sample_mini_batch(self):
        sample_size = np.min([self.exp_actual_size, self.exp_size])
        prob = self.priority[0:sample_size] / np.sum(self.priority[0:sample_size])
        sample_indices = self.random_state.choice(a=sample_size, size=self.mini_batch_size, p=prob, replace=False)

        mini_batch_exp = self.experience[sample_indices, :]
        phi_s = mini_batch_exp[:, 0:self.phi_s_size]
        actions = mini_batch_exp[:, self.phi_s_size]
        rewards = mini_batch_exp[:, self.phi_s_size+1]
        phi_ns = mini_batch_exp[:,self.phi_s_size+2:]
        policy_ns = [self.ns_policies[i] for i in sample_indices]

        return phi_s, actions, rewards, phi_ns, policy_ns, sample_indices

    def update_priority(self, sample_indices, td_error):
        self.priority[sample_indices] = np.clip(td_error, 0, 20) + 1.0





