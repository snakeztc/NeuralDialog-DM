import numpy as np
from Experience import Experiences
from scipy.sparse import coo_matrix


class StructTurnExperience (Experiences):
    phi_s_size = None
    max_len = None
    exp_ar = None

    def __init__(self, exp_size, usr_size, sys_size, max_len, mini_batch_size, use_priority, seed):
        super(StructTurnExperience, self).__init__(use_priority=use_priority, mini_batch_size=mini_batch_size, seed=seed)

        self.exp_ar = np.zeros((exp_size, 2))
        self.s_policies = [None] * exp_size
        self.ns_policies = [None] * exp_size

        self.priority = np.zeros(exp_size)
        self.exp_size = exp_size
        self.max_len = max_len + 1
        self.exp_head = 0
        self.exp_actual_size = 0
        self.usr_size = usr_size
        self.sys_size = sys_size

        # sys and cmp are array num_sample * max_len * 2 (s and ns)
        self.sys = np.zeros((exp_size, self.max_len, 2 * self.sys_size))
        self.cmp = np.zeros((exp_size, self.max_len, 2))
        self.usr = [None] * exp_size

    def add_experience(self, phi_s, policy_s, a, r, phi_ns, policy_ns, priority):
        if self.exp_head >= self.exp_size:
            print "** reset exp header **"
            self.exp_head = 0

        s_len = len(phi_s["sys"])
        ns_len = len(phi_ns["sys"])

        sys_s = np.zeros((s_len, self.sys_size))
        sys_ns = np.zeros((ns_len, self.sys_size))
        # populate sys s
        indices_s = [int(v + idx*self.sys_size) for idx, v in enumerate(phi_s["sys"])]
        sys_s.flat[indices_s] = 1
        # populate ns
        indices_ns = [int(v + idx*self.sys_size) for idx, v in enumerate(phi_ns["sys"])]
        sys_ns.flat[indices_ns] = 1


        # pad phi_s and phi_ns with 0 zeros in the front
        self.usr[self.exp_head] = (phi_s["usr"], phi_ns["usr"])
        self.sys[self.exp_head, self.max_len-s_len:self.max_len, 0:self.sys_size] = sys_s
        self.sys[self.exp_head, self.max_len-ns_len:self.max_len, self.sys_size:] = sys_ns

        self.cmp[self.exp_head, self.max_len-s_len:self.max_len, 0] = phi_s["cmp"]
        self.cmp[self.exp_head, self.max_len-ns_len:self.max_len, 1] = phi_ns["cmp"]

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
        prob = self.priority[0:sample_size] / np.sum(self.priority[0:sample_size])
        sample_indices = self.random_state.choice(a=sample_size, size=self.mini_batch_size, p=prob, replace=False)

        phi_s = {"usr": np.zeros((self.mini_batch_size, self.max_len, self.usr_size)),
                 "sys": np.zeros((self.mini_batch_size, self.max_len, self.sys_size)),
                 "cmp": self.cmp[sample_indices, :, 0:1]}

        phi_ns = {"usr": np.zeros((self.mini_batch_size, self.max_len, self.usr_size)),
                  "sys": np.zeros((self.mini_batch_size, self.max_len, self.sys_size)),
                  "cmp": self.cmp[sample_indices, :, 1:2]}

        # fill in the dense matrix for usr
        for idx, sample_idx in enumerate(sample_indices):
            (dense_s, dense_ns) = self.usr[sample_idx]
            dense_s = dense_s.toarray()
            dense_ns = dense_ns.toarray()
            # to 3D tensor
            dense_s = np.reshape(dense_s, (1,) + dense_s.shape)
            dense_ns = np.reshape(dense_ns, (1,) + dense_ns.shape)

            phi_s["usr"][idx, self.max_len-dense_s.shape[1]:self.max_len, :] = dense_s
            phi_ns["usr"][idx, self.max_len-dense_ns.shape[1]:self.max_len, :] = dense_ns

            phi_s["sys"][idx, :, :] = self.sys[idx, :, 0:self.sys_size]
            phi_s["sys"][idx, :, :] = self.sys[idx, :, self.sys_size:]


        actions = self.exp_ar[sample_indices, 0]
        rewards = self.exp_ar[sample_indices, 1]
        policy_ns = [self.ns_policies[i] for i in sample_indices]
        policy_s = [self.s_policies[i] for i in sample_indices]
        return phi_s, policy_s, actions, rewards, phi_ns, policy_ns, sample_indices

    def update_priority(self, sample_indices, td_error):
        self.priority[sample_indices] = np.clip(td_error, 0, 20) + 1.0





