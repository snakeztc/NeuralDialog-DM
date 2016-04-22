import numpy as np
from Experience import Experiences
from scipy.sparse import coo_matrix
from keras.utils.np_utils import to_categorical


class HybridTurnExperience (Experiences):
    phi_s_size = None
    max_len = None
    exp_ar = None

    def __init__(self, exp_size, phi_s_size, max_len, mini_batch_size, use_priority, seed):
        super(HybridTurnExperience, self).__init__(use_priority=use_priority, mini_batch_size=mini_batch_size, seed=seed)

        self.exp_ar = np.zeros((exp_size, 2))
        self.s_policies = [None] * exp_size
        self.ns_policies = [None] * exp_size
        self.spl_targets = [np.zeros(exp_size) for i in range(32)]

        self.priority = np.zeros(exp_size)
        self.exp_size = exp_size
        self.max_len = max_len + 1
        self.exp_head = 0
        self.exp_actual_size = 0
        self.phi_s_size = phi_s_size

        # experience is a list of 2D sparse matrix
        self.experience = [None] * self.exp_size

    def add_experience(self, phi_s, policy_s, a, r, phi_ns, policy_ns, priority, spl_targets=None):
        if self.exp_head >= self.exp_size:
            print "** reset exp header **"
            self.exp_head = 0

        # pad phi_s and phi_ns with 0 zeros in the front
        self.experience[self.exp_head] = (coo_matrix(phi_s["input"][0, :, :]), coo_matrix(phi_ns["input"][0, :, :]))
        self.exp_ar[self.exp_head, 0] = a
        self.exp_ar[self.exp_head, 1] = r
        self.s_policies[self.exp_head] = policy_s
        self.ns_policies[self.exp_head] = policy_ns
        self.priority[self.exp_head] = 20.0
        for idx, target in enumerate(spl_targets):
            self.spl_targets[idx][self.exp_head] = target

        # increment the write head
        self.exp_head += 1
        self.exp_actual_size += 1

    def get_spl_experience(self, sample_indices):
        results = [None] * len(self.spl_targets)
        for i in range(len(self.spl_targets)):
            if i == 0:
                results[i] = self.spl_targets[i][sample_indices]
            else:
                results[i] = to_categorical(self.spl_targets[i][sample_indices], 4)
        return results

    def sample_mini_batch(self):
        sample_size = np.min([self.exp_actual_size, self.exp_size])
        prob = self.priority[0:sample_size] / np.sum(self.priority[0:sample_size])
        sample_indices = self.random_state.choice(a=sample_size, size=self.mini_batch_size, p=prob, replace=False)

        # allocate dense 3d tensor num_sample * max_len * dimension
        phi_s = np.zeros((self.mini_batch_size, self.max_len, self.phi_s_size))
        phi_ns = np.zeros((self.mini_batch_size, self.max_len, self.phi_s_size))

        # fill in the dense matrix
        for idx, sample_idx in enumerate(sample_indices):
            (dense_s, dense_ns) = self.experience[sample_idx]
            dense_s = dense_s.toarray()
            dense_ns = dense_ns.toarray()
            # to 3D tensor
            dense_s = np.reshape(dense_s, (1,) + dense_s.shape)
            dense_ns = np.reshape(dense_ns, (1,) + dense_ns.shape)

            phi_s[idx, self.max_len-dense_s.shape[1]:self.max_len, :] = dense_s
            phi_ns[idx, self.max_len-dense_ns.shape[1]:self.max_len, :] = dense_ns

        actions = self.exp_ar[sample_indices, 0]
        rewards = self.exp_ar[sample_indices, 1]
        policy_ns = [self.ns_policies[i] for i in sample_indices]
        policy_s = [self.s_policies[i] for i in sample_indices]
        return {"input": phi_s}, policy_s, actions, rewards, {"input": phi_ns}, policy_ns, sample_indices

    def update_priority(self, sample_indices, td_error):
        self.priority[sample_indices] = np.clip(td_error, 0, 20) + 1.0





