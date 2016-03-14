from Agent import Agent
from Agents.BatchAgents.DnnQ import DnnQ
from Experience.OracleStateExperience import OracleStateExperience
from Policies.Policy import EpsilonGreedyPolicy
from Representations.BinaryCompactRep import BinaryCompactRep


class ExpQLearning(Agent):
    fake_policy = []
    for i in range(100):
        if i < 62:
            if i % 2 == 0:
                fake_policy.append(i/2)
            else:
                fake_policy.append(32+i/2)
        else:
            fake_policy.append(31)

    def learn(self, s, performance_run=False):

        # choose an action. If in learning, we use behavior policy, If not use target policy
        mask = self.domain.action_prune(s)
        if performance_run:
            Qs = self.representation.Qs(s)
            aID = self.performance_policy.choose_action(Qs, mask)
        else:
            Qs = self.behavior_representation.Qs(s)
            aID = self.learning_policy.choose_action(Qs, mask)

        (r, ns, terminal) = self.domain.step(s, aID)

        if terminal and self.verbose:
            self.print_episode(ns[1])
            print "final reward is " + str(r)

        if not performance_run:
            # check if exp_head is larger than buffer size
            self.experience.add_experience(self.representation.phi_s(s), aID, r, self.representation.phi_s(ns), 20.0)

            if (self.experience.exp_actual_size > self.experience.mini_batch_size)\
                    and (self.experience.exp_actual_size % self.update_frequency) == 0:

                self.learner.learn(self.experience)

                # update target model
                self.update_cnt += 1

                if self.update_cnt % self.freeze_frequency == 0:
                    self.learner.update_target_model()

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1, epsilon=0.3, update_frequency=10, freeze_frequency=20,
                 exp_size=10000, mini_batch=3000, doubleDQN=False, verbose=False):
        super(ExpQLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(epsilon, seed)
        self.update_frequency = update_frequency
        self.experience = OracleStateExperience(exp_size=exp_size, phi_s_size=self.representation.state_features_num,
                                                mini_batch_size=mini_batch, use_priority=True, seed=seed)

        self.update_cnt = 0
        self.freeze_frequency = freeze_frequency
        self.behavior_representation = BinaryCompactRep(domain, seed)
        self.learner = DnnQ(domain=domain, representation=representation,
                            behavior_representation=self.behavior_representation,
                            seed=seed, doubleDQN=doubleDQN)
        self.verbose = verbose
        print "Using epsilon " + str(epsilon)
        print "Update_frequency " + str(self.update_frequency)
        print "Mini-batch size is " + str(self.experience.mini_batch_size)