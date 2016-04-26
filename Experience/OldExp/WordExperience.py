import numpy as np
from Experience.Experience import Experiences
from keras.preprocessing import sequence



class WordExperience(Experiences):
    exp_s = None
    exp_ns = None
    exp_ar = None

    def __init__(self, exp_size, mini_batch_size, use_priority, seed):
        super(WordExperience, self).__init__(use_priority=use_priority, mini_batch_size=mini_batch_size, seed=seed)
        self.exp_s = [None] * exp_size
        self.exp_ns = [None] * exp_size
        self.exp_ar = np.zeros((exp_size, 2))
        self.priority = np.zeros(exp_size)
        self.exp_size = exp_size
        self.exp_head = 0
        self.exp_actual_size = 0

    def add_experience(self, phi_s, a, r, phi_ns, priority):
        if self.exp_head >= self.exp_size:
            print "** reset exp header **"
            self.exp_head = 0

        self.exp_s[self.exp_head] = phi_s
        self.exp_ns[self.exp_head] = phi_ns
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

        exp_s = [self.exp_s[i] for i in sample_indices]
        actions = self.exp_ar[sample_indices, 0]
        rewards = self.exp_ar[sample_indices, 1]
        exp_ns = [self.exp_ns[i] for i in sample_indices]

        phi_s = sequence.pad_sequences(exp_s)
        phi_ns = sequence.pad_sequences(exp_ns)

        return phi_s, actions, rewards, phi_ns, sample_indices

    def update_priority(self, sample_indices, td_error):
        self.priority[sample_indices] = np.clip(td_error, 0, 20) + 1.0





