from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy
import numpy as np
from DNNfqi import DNNfqi


class ExpQLearning(Agent):

    def learn(self, s, performance_run=False):
        Qs = self.representation.Qs(s)

        # choose an action
        if performance_run:
            aID = self.performance_policy.choose_action(Qs)
        else:
            aID = self.learning_policy.choose_action(Qs)

        (r, ns, terminal) = self.domain.step(s, aID)

        if not performance_run:
            # check if exp_head is larger than buffer size
            if self.exp_head >= self.exp_size:
                self.exp_head = 0

            phi_s_size = self.representation.state_features_num
            self.experience[self.exp_head, 0:phi_s_size] = self.representation.phi_s(s)
            self.experience[self.exp_head, phi_s_size] = aID
            self.experience[self.exp_head, phi_s_size+1] = r
            self.experience[self.exp_head, phi_s_size+2:] = self.representation.phi_s(ns)
            self.priority[self.exp_head] = np.min([np.abs(r),5.0]) + 1.0
            # increment the write head
            self.exp_head += 1

            if self.exp_head > self.mini_batch and (self.exp_head % self.update_frequency) == 0:
                prob = self.priority[0:self.exp_head] / np.sum(self.priority[0:self.exp_head])
                mini_batch_exp = self.experience[self.random_state.choice(a=self.exp_head,
                                                                          size=self.mini_batch, p=prob, replace=False), :]
                self.learner.learn(mini_batch_exp)

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1, epsilon=0.3, update_frequency=10, exp_size=10000, mini_batch=3000):
        super(ExpQLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(epsilon, seed)
        self.update_frequency = update_frequency
        self.mini_batch = mini_batch
        self.exp_size = exp_size
        self.exp_head = 0
        self.learner = DNNfqi(domain=domain, representation=representation)
        self.experience = np.zeros((self.exp_size, self.representation.state_features_num * 2 + 2))
        self.priority = np.zeros(self.exp_size)
        print "Using epsilon " + str(epsilon)
        print "Update_frequency " + str(self.update_frequency)
        print "Mini-batch size is " + str(self.mini_batch)
        print "Experience size is " + str(self.exp_size)
