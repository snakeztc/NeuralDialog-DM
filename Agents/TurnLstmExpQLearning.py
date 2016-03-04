from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy
from Agents.BatchAgents.TurnLstmDnnQ import TurnLstmDnnQ
from Experience.TurnExperience import TurnExperience
from Representations.TurnHistoryRep import TurnHistoryRep


class TurnLstmExpQLearning(Agent):

    def learn(self, s, performance_run=False):
        Qs = self.behavior_representation.Qs(s)

        # choose an action
        if performance_run:
            aID = self.performance_policy.choose_action(Qs)
        else:
            aID = self.learning_policy.choose_action(Qs)

        (r, ns, terminal) = self.domain.step(s, aID)

        if not performance_run:

            self.experience.add_experience(s[1], aID, r, ns[1], 20.0)

            if self.experience.exp_actual_size > self.experience.mini_batch_size\
                    and (self.experience.exp_actual_size % self.update_frequency) == 0:

                self.learner.learn(self.experience)
                # update target model
                self.update_cnt += 1

                if self.update_cnt % self.freeze_frequency == 0:
                    self.learner.update_target_model()

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1, epsilon=0.3, update_frequency=10,
                 exp_size=10000, mini_batch=3000, freeze_frequency=100, doubleDQN=False):

        super(TurnLstmExpQLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(epsilon, seed)
        self.update_frequency = update_frequency

        self.experience = TurnExperience(exp_size=exp_size,  mini_batch_size=mini_batch, use_priority=True, seed=seed)

        # freeze model
        self.freeze_frequency = freeze_frequency
        self.update_cnt = 0

        self.behavior_representation = TurnHistoryRep(domain, seed=seed)
        # learner
        self.learner = TurnLstmDnnQ(domain=domain, representation=representation,
                                    behavior_representation=self.behavior_representation,
                                    doubleDQN=doubleDQN, seed=seed)
        print "Using epsilon " + str(epsilon)
        print "Update_frequency " + str(self.update_frequency)
        print "Mini-batch size is " + str(self.experience.mini_batch_size)