from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy
from Representations.PartialObserveRep import PartialObserveRep
import numpy as np
from LstmDnnQ import LstmDnnQ


class LstmExpQLearning(Agent):

    def learn(self, s, performance_run=False):
        Qs = self.behavior_representation.Qs(s)

        # alternating between computer operation and human operation
        if self.domain.holding in s[0]:
            start = self.domain.actions_num - 3
            end = self.domain.actions_num
        else:
            start = 0
            end = self.domain.actions_num - 3

        # choose an action
        if performance_run:
            aID = self.performance_policy.choose_action(Qs[0:, start:end])
        else:
            aID = self.learning_policy.choose_action(Qs[0:, start:end])

        aID += start

        (r, ns, terminal) = self.domain.step(s, aID)

        if not performance_run:
            # check if exp_head is larger than buffer size
            if self.exp_head >= self.exp_size:
                self.exp_head = 0

            self.exp_s[self.exp_head] = s[1]
            self.exp_ns[self.exp_head] = ns[1]
            self.exp_ar[self.exp_head, 0] = aID
            self.exp_ar[self.exp_head, 1] = r
            self.priority[self.exp_head] = np.min([np.abs(r),5.0]) + 1.0
            # increment the write head
            self.exp_head += 1

            if self.exp_head > self.mini_batch and (self.exp_head % self.update_frequency) == 0:
                prob = self.priority[0:self.exp_head] / np.sum(self.priority[0:self.exp_head])
                indices = self.random_state.choice(a=self.exp_head, size=self.mini_batch, p=prob, replace=False)
                mini_batch_exp = ([self.exp_s[i] for i in indices], self.exp_ar[indices, :], [self.exp_ns[i] for i in indices])
                self.learner.learn(mini_batch_exp)
                # update target model
                self.update_cnt += 1

                if self.update_cnt % self.freeze_frequency == 0:
                    self.learner.update_target_model()

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1, epsilon=0.3, update_frequency=10,
                 exp_size=10000, mini_batch=3000, freeze_frequency=100, doubleDQN=False):
        super(LstmExpQLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(epsilon, seed)
        self.update_frequency = update_frequency
        self.mini_batch = mini_batch
        self.exp_size = exp_size
        self.exp_head = 0
        # freeze model
        self.freeze_frequency = freeze_frequency
        self.update_cnt = 0
        self.behavior_representation = PartialObserveRep(domain, seed=seed)
        # learner
        self.learner = LstmDnnQ(domain=domain, representation=representation,
                                behavior_representation=self.behavior_representation,
                                doubleDQN=doubleDQN)
        # experiences
        self.exp_s = [None] * self.exp_size
        self.exp_ns = [None] * self.exp_size
        self.exp_ar = np.zeros((self.exp_size, 2))
        self.priority = np.zeros(self.exp_size)

        print "Using epsilon " + str(epsilon)
        print "Update_frequency " + str(self.update_frequency)
        print "Mini-batch size is " + str(self.mini_batch)
        print "Experience size is " + str(self.exp_size)
